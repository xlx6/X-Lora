import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from typing import Optional, Dict, Any, Literal
import numpy as np


class TriangularMatrix(nn.Module):
    """
    Triangle Wrapper
    """
    def __init__(
        self,
        rows: int,
        cols: int,
        triangular_type: Literal['lower', 'upper'] = 'lower',
        diagonal_init: float = 1.0,
    ):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.triangular_type = triangular_type
        
        self.weight = nn.Parameter(torch.zeros(rows, cols))
        
        # triangle mask
        self.register_buffer('mask', self._create_mask())
        
        self._initialize_weights(diagonal_init)
    
    def _create_mask(self) -> torch.Tensor:
        """triangle mask"""
        mask = torch.ones(self.rows, self.cols)
        
        if self.triangular_type == 'lower':
            mask = torch.tril(mask)
        else:
            mask = torch.triu(mask)
        
        return mask
    
    def _initialize_weights(self, diagonal_init: float):
        """init weights"""
        with torch.no_grad():
            nn.init.normal_(self.weight, mean=0.0, std=0.02)
            
            min_dim = min(self.rows, self.cols)
            for i in range(min_dim):
                self.weight[i, i] = diagonal_init
            
            self.weight.mul_(self.mask)
    
    def forward(self) -> torch.Tensor:
        """Return Triangle Matrix"""
        return self.weight * self.mask
    
    def get_num_params(self) -> int:
        """Compute Truth Params"""
        if self.triangular_type == 'lower':
            return int(self.mask.sum().item())
        else:
            return int(self.mask.sum().item())


class StructuredLoRALayer(nn.Module):
    """
    Structured LoRA Layer: ΔW = L * U
    L: low triangle (out_features x rank)
    U: upper triangle (rank x in_features)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        lora_alpha: float = 16,
        lora_dropout: float = 0.0,
        structure_type: Literal['lu', 'cholesky'] = 'lu',
        diagonal_init_L: float = 1.0,
        diagonal_init_U: float = 1.0,
    ):
        super().__init__()
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        self.structure_type = structure_type
        
        if structure_type == 'lu':
            self.L = TriangularMatrix(
                rows=out_features,
                cols=rank,
                triangular_type='lower',
                diagonal_init=diagonal_init_L,
            )
            self.U = TriangularMatrix(
                rows=rank,
                cols=in_features,
                triangular_type='upper',
                diagonal_init=diagonal_init_U,
            )
        elif structure_type == 'cholesky':
            # Cholesky: ΔW = L * L^T
            self.L = TriangularMatrix(
                rows=out_features,
                cols=rank,
                triangular_type='lower',
                diagonal_init=diagonal_init_L,
            )
            self.U = None
        else:
            raise ValueError(f"Unknown structure_type: {structure_type}")
        
        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, in_features)
        """
        L_matrix = self.L()  # (out_features, rank)
        
        if self.structure_type == 'lu':
            U_matrix = self.U()  # (rank, in_features)
            # ΔW = L * U, x @ ΔW^T = x @ U^T @ L^T
            lora_out = self.lora_dropout(x) @ U_matrix.T @ L_matrix.T
        else:  # cholesky
            # ΔW = L * L^T, x @ ΔW^T = x @ L @ L^T
            lora_out = self.lora_dropout(x) @ L_matrix @ L_matrix.T
        
        return lora_out * self.scaling
    
    def get_num_params(self) -> int:
        L_params = self.L.get_num_params()
        if self.structure_type == 'lu':
            U_params = self.U.get_num_params()
            return L_params + U_params
        else:
            return L_params
    
    def get_delta_weight(self) -> torch.Tensor:
        L_matrix = self.L()
        if self.structure_type == 'lu':
            U_matrix = self.U()
            return L_matrix @ U_matrix * self.scaling
        else:
            return L_matrix @ L_matrix.T * self.scaling


class StructuredLoRALayerV2(nn.Module):
    """
    Directly use mask constraint B and A
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        lora_alpha: float = 16,
        lora_dropout: float = 0.0,
        B_triangular: Literal['lower', 'upper', 'none'] = 'lower',
        A_triangular: Literal['lower', 'upper', 'none'] = 'upper',
    ):
        super().__init__()
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        
        # LoRA 矩阵
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # 创建 masks
        if B_triangular != 'none':
            self.register_buffer(
                'B_mask',
                self._create_triangular_mask(out_features, rank, B_triangular)
            )
        else:
            self.register_buffer('B_mask', torch.ones(out_features, rank))
        
        if A_triangular != 'none':
            self.register_buffer(
                'A_mask',
                self._create_triangular_mask(rank, in_features, A_triangular)
            )
        else:
            self.register_buffer('A_mask', torch.ones(rank, in_features))
        
        self._initialize_weights()

        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
    
    def _create_triangular_mask(
        self,
        rows: int,
        cols: int,
        triangular_type: str
    ) -> torch.Tensor:
        """build mask"""
        mask = torch.ones(rows, cols)
        if triangular_type == 'lower':
            mask = torch.tril(mask)
        elif triangular_type == 'upper':
            mask = torch.triu(mask)
        return mask
    
    def _initialize_weights(self):
        with torch.no_grad():
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)
            
            # 应用 mask
            self.lora_A.mul_(self.A_mask)
            self.lora_B.mul_(self.B_mask)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A_masked = self.lora_A * self.A_mask
        B_masked = self.lora_B * self.B_mask
        
        # x @ A^T @ B^T
        lora_out = self.lora_dropout(x) @ A_masked.T @ B_masked.T
        return lora_out * self.scaling
    
    def get_num_params(self) -> int:
        return int(self.A_mask.sum().item() + self.B_mask.sum().item())


class LinearWithStructuredLoRA(nn.Module):
    """
    Structured LoRA Linear Layer
    """
    def __init__(
        self,
        linear_layer: nn.Linear,
        rank: int = 8,
        lora_alpha: float = 16,
        lora_dropout: float = 0.0,
        structure_type: Literal['lu', 'cholesky'] = 'lu',
        use_v2: bool = False,
    ):
        super().__init__()
        self.linear = linear_layer
        self.linear.weight.requires_grad = False
        
        if use_v2:
            self.lora = StructuredLoRALayerV2(
                in_features=linear_layer.in_features,
                out_features=linear_layer.out_features,
                rank=rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                B_triangular='lower',
                A_triangular='upper',
            )
        else:
            self.lora = StructuredLoRALayer(
                in_features=linear_layer.in_features,
                out_features=linear_layer.out_features,
                rank=rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                structure_type=structure_type,
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)


class StructuredLoRATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        if self.state.global_step % self.args.logging_steps == 0:
            total_params = 0
            structured_params = 0
            
            for module in model.modules():
                if isinstance(module, (StructuredLoRALayer, StructuredLoRALayerV2)):
                    structured_params += module.get_num_params()
                    if isinstance(module, StructuredLoRALayer):
                        if module.structure_type == 'lu':
                            full_params = (
                                module.L.rows * module.rank +
                                module.rank * module.U.cols
                            )
                        else:
                            full_params = module.L.rows * module.rank
                        total_params += full_params
            
            if total_params > 0:
                sparsity = 1 - (structured_params / total_params)
                self.log({
                    'structured_params': structured_params,
                    'sparsity': sparsity,
                })
        
        return (loss, outputs) if return_outputs else loss


def apply_structured_lora_to_model(
    model: nn.Module,
    target_modules: list = None,
    rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    structure_type: Literal['lu', 'cholesky'] = 'lu',
    use_v2: bool = False,
):
    """
    Apply Structured LoRA to target module
    
    Args:
        model: base model
        target_modules: target module name list
        rank: r
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        structure_type: 'lu' or 'cholesky'
        use_v2: use mask implement structured lora
    """
    if target_modules is None:
        target_modules = ['q_lin', 'v_lin', 'k_lin']
    
    total_replaced = 0
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                
                lora_layer = LinearWithStructuredLoRA(
                    linear_layer=module,
                    rank=rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    structure_type=structure_type,
                    use_v2=use_v2,
                )
                
                setattr(parent, child_name, lora_layer)
                total_replaced += 1
    
    print(f"Applied Structured LoRA to {total_replaced} layers")
    return model


def compare_parameter_efficiency():
    in_features = 768
    out_features = 768
    rank = 8
    
    # Standard LoRA
    standard_params = rank * in_features + out_features * rank
    
    # L
    L_params = (rank * (rank + 1)) // 2 + (out_features - rank) * rank
    if out_features < rank:
        L_params = (out_features * (out_features + 1)) // 2
    
    # U
    U_params = (rank * (rank + 1)) // 2 + (in_features - rank) * rank
    if in_features < rank:
        U_params = (in_features * (in_features + 1)) // 2
    
    lu_params = L_params + U_params
    
    # Cholesky LoRA
    cholesky_params = L_params
    
    print(f"Standard LoRA parameters: {standard_params:,}")
    print(f"LU Structured LoRA parameters: {lu_params:,}")
    print(f"Cholesky Structured LoRA parameters: {cholesky_params:,}")
    print(f"\nParameter reduction (LU): {(1 - lu_params/standard_params)*100:.2f}%")
    print(f"Parameter reduction (Cholesky): {(1 - cholesky_params/standard_params)*100:.2f}%")

if __name__=='__main__':
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments

    model = AutoModelForSequenceClassification.from_pretrained("./distilbert-base-uncased", num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("./distilbert-base-uncased")
    model = apply_structured_lora_to_model(
        model,
        target_modules=['q_lin', 'k_lin', 'v_lin'],
        rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        structure_type='lu',
        use_v2=False,
    )

    compare_parameter_efficiency()
    print(model)

    training_args = TrainingArguments(
        output_dir="./structured_lora_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        logging_steps=10,
        save_steps=100,
    )

# trainer = StructuredLoRATrainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
# )

# trainer.train()
