import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from typing import Optional, Dict, Any
import numpy as np
import swanlab

class OrthogonalLoRALayer(nn.Module):
    """
    Orthogonal LoRA layer
    1. orthogonal method: (orthogonal_reg=True)
    2. QR: (orthogonal_reg=False, use_qr=True)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        lora_alpha: float = 16,
        lora_dropout: float = 0.0,
        orthogonal_reg: bool = False,
        use_qr: bool = True,
        qr_frequency: int = 1,  # QR freq
    ):
        super().__init__()
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        self.orthogonal_reg = orthogonal_reg
        self.use_qr = use_qr
        self.qr_frequency = qr_frequency
        self.step_counter = 0
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, in_features)
        """
        # LoRA 计算: x @ A^T @ B^T
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return lora_out * self.scaling
    
    def get_orthogonal_loss(self) -> torch.Tensor:
        """
        ||B^T B - I||_F^2
        """
        if not self.orthogonal_reg:
            return torch.tensor(0.0, device=self.lora_B.device)
        
        # B^T B
        BTB = self.lora_B.T @ self.lora_B  # (rank, rank)
        I = torch.eye(self.rank, device=self.lora_B.device, dtype=self.lora_B.dtype)
        ortho_loss = torch.norm(BTB - I, p='fro') ** 2
        
        return ortho_loss
    
    def orthogonalize_B(self):
        """
        QR->B(col)
        B: (out_features, rank)
        """
        if not self.use_qr:
            return
        
        with torch.no_grad():
            Q, R = torch.linalg.qr(self.lora_B, mode='reduced')
            self.lora_B.data.copy_(Q.to(self.lora_B.dtype))
    
    def step(self):
        self.step_counter += 1
        if self.use_qr and self.step_counter % self.qr_frequency == 0:
            self.orthogonalize_B()


class LinearWithOrthogonalLoRA(nn.Module):
    def __init__(
        self,
        linear_layer: nn.Linear,
        rank: int = 8,
        lora_alpha: float = 16,
        lora_dropout: float = 0.0,
        orthogonal_reg: bool = False,
        use_qr: bool = True,
        qr_frequency: int = 1,
    ):
        super().__init__()
        self.linear = linear_layer
        self.linear.weight.requires_grad = False
        
        self.lora = OrthogonalLoRALayer(
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features,
            rank=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            orthogonal_reg=orthogonal_reg,
            use_qr=use_qr,
            qr_frequency=qr_frequency,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)


class OrthogonalLoRATrainer(Trainer):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        self.orthogonal_lambda = kwargs.pop('orthogonal_lambda')
        self.use_qr = kwargs.pop('use_qr')
        self.qr_frequency = kwargs.pop('qr_frequency')
        super().__init__(*args, **kwargs)

    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        orthogonal_loss = torch.tensor(0.0, device=loss.device)
        lora_layer_count = 0
        
        for module in model.modules():
            if isinstance(module, OrthogonalLoRALayer):
                orthogonal_loss += module.get_orthogonal_loss()
                lora_layer_count += 1
        
        if lora_layer_count > 0:
            orthogonal_loss = orthogonal_loss / lora_layer_count
            total_loss = loss + self.orthogonal_lambda * orthogonal_loss
            
            if self.state.global_step % self.args.logging_steps == 0:
                swanlab.log({
                    'original_loss': loss.item(),
                    'orthogonal_loss': orthogonal_loss.item(),
                    'total_loss': total_loss.item(),
                })
        else:
            total_loss = loss
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Any], return_loss: bool = True) -> torch.Tensor:
        loss = super().training_step(model, inputs)
        
        if self.use_qr:
            for module in model.modules():
                if isinstance(module, OrthogonalLoRALayer):
                    module.step()
        
        return loss


def apply_orthogonal_lora_to_model(
    model: nn.Module,
    target_modules: list = None,
    rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    orthogonal_reg: bool = False,
    use_qr: bool = True,
    qr_frequency: int = 1,
):
    """
    Apply orthogonal lora to specify module
    
    Args:
        model: base model
        target_modules: module name list: ['q_proj', 'v_proj']
        rank: r
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        orthogonal_reg: whether to use orthogonal reg
        use_qr: whether to use qr
        qr_frequency: QR Freq
    """
    if target_modules is None:
        target_modules = ['q_lin', 'v_lin', 'k_lin']
    
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                
                lora_layer = LinearWithOrthogonalLoRA(
                    linear_layer=module,
                    rank=rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    orthogonal_reg=orthogonal_reg,
                    use_qr=use_qr,
                    qr_frequency=qr_frequency,
                )
                
                setattr(parent, child_name, lora_layer)
    
    return model


if __name__=='__main__':
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments

    model = AutoModelForSequenceClassification.from_pretrained("./distilbert-base-uncase")
    tokenizer = AutoTokenizer.from_pretrained("./distilbert-base-uncase")

    model = apply_orthogonal_lora_to_model(
        model,
        target_modules=['q_lin', 'k_lin', 'v_lin'],
        rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        orthogonal_reg=True,
        use_qr=True,
        qr_frequency=10,
    )
    print(model)
    # training_args = TrainingArguments(
    #     output_dir="./orthogonal_lora_output",
    #     num_train_epochs=3,
    #     per_device_train_batch_size=4,
    #     logging_steps=10,
    #     save_steps=100,
    # )

    # # 4. 创建 Trainer
    # trainer = OrthogonalLoRATrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     orthogonal_lambda=0.01,
    #     use_qr=True,
    #     qr_frequency=10,
    # )

    # trainer.train()
