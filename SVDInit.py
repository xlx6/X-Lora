import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel
from typing import Optional, Dict, List
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SVDLoraConfig(LoraConfig):
    """
    Extend LoraConfig to support SVD-init
    
    extra args:
        use_svd_init: use or no-use
        energy_threshold: Auto select rank
        auto_rank_selection: Use Auto select rank
    """
    def __init__(
        self,
        use_svd_init: bool = True,
        energy_threshold: float = 0.9,
        auto_rank_selection: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.use_svd_init = use_svd_init
        self.energy_threshold = energy_threshold
        self.auto_rank_selection = auto_rank_selection


def compute_optimal_rank(singular_values: torch.Tensor, energy_threshold: float = 0.9) -> int:
    """
    Compute Optimize Rank Acording Singular Value
    
    Args:
        singular_values: SV Tensor
        energy_threshold: from 0 to 1
    
    Returns:
        Optimize r
    """
    # Compute Accumulate Energy Ratio
    total_energy = torch.sum(singular_values ** 2)
    cumulative_energy = torch.cumsum(singular_values ** 2, dim=0)
    energy_ratio = cumulative_energy / total_energy
    
    rank = torch.searchsorted(energy_ratio, energy_threshold).item() + 1
    
    rank = max(1, rank)
    
    logger.info(f"SVD Analysis: Total singular values: {len(singular_values)}")
    logger.info(f"Optimal rank for {energy_threshold*100}% energy: {rank}")
    logger.info(f"Top 10 singular values: {singular_values[:10].tolist()}")
    
    return rank


def analyze_weight_spectrum(weight: torch.Tensor) -> Dict:
    """
    Analyzing the singular value spectrum of weight matrix
    
    Args:
        weight: weight matrix
    
    Returns:
        A dict containing analysis results
    """
    U, S, Vt = torch.linalg.svd(weight, full_matrices=False)
    
    energy = S ** 2
    total_energy = torch.sum(energy)
    cumulative_energy = torch.cumsum(energy, dim=0) / total_energy
    
    thresholds = [0.8, 0.85, 0.9, 0.95, 0.99]
    rank_suggestions = {}
    for threshold in thresholds:
        rank = compute_optimal_rank(S, threshold)
        rank_suggestions[f"{int(threshold*100)}%"] = rank
    
    return {
        "singular_values": S,
        "U": U,
        "Vt": Vt,
        "cumulative_energy": cumulative_energy,
        "rank_suggestions": rank_suggestions,
        "spectral_decay": (S[0] / S[-1]).item() if len(S) > 0 else float('inf')
    }


def initialize_lora_with_svd(
    original_weight: torch.Tensor,
    rank: int,
    include_sigma: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute A, B
    
    Args:
        original_weight: pretrained weights
        rank: r
        include_sigma: use singular value?
    
    Returns:
        (A_init, B_init, W_residual)
    """
    U, S, Vt = torch.linalg.svd(original_weight, full_matrices=False)
    
    U_r = U[:, :rank]  # (out_features, r)
    S_r = S[:rank]      # (r,)
    Vt_r = Vt[:rank, :] # (r, in_features)
    
    if include_sigma:
        B_init = U_r @ torch.diag(S_r)  # (out_features, r)
        A_init = Vt_r                    # (r, in_features)
    else:
        B_init = U_r
        A_init = torch.diag(S_r) @ Vt_r
    
    low_rank_approx = U_r @ torch.diag(S_r) @ Vt_r
    W_residual = original_weight - low_rank_approx
    
    reconstruction_error = torch.norm(original_weight - low_rank_approx, p='fro')
    relative_error = reconstruction_error / torch.norm(original_weight, p='fro')
    logger.info(f"SVD Initialization:")
    logger.info(f"  Rank: {rank}")
    logger.info(f"  Reconstruction error: {reconstruction_error.item():.6f}")
    logger.info(f"  Relative error: {relative_error.item():.4%}")
    
    return A_init, B_init, W_residual


# def apply_svd_initialization_to_model(
#     model: PeftModel,
#     original_model: nn.Module,
#     config: SVDLoraConfig,
#     target_modules: Optional[List[str]] = None
# ):
#     if not config.use_svd_init:
#         logger.info("SVD initialization is disabled.")
#         return
    
#     logger.info("\n" + "="*60)
#     logger.info("Applying SVD-based LoRA initialization...")
#     logger.info("="*60)
    
#     for name, module in model.named_modules():
#         if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
#             original_weight = None
            
#             name_parts = name.split('.')
#             current = original_model
#             for part in name_parts:
#                 if hasattr(current, part):
#                     current = getattr(current, part)
#                 else:
#                     break
            
#             if hasattr(current, 'weight'):
#                 original_weight = current.weight.data.clone()
#             else:
#                 continue
            
#             logger.info(f"\nInitializing module: {name}")
#             logger.info(f"  Original weight shape: {original_weight.shape}")
            
#             if config.auto_rank_selection:
#                 spectrum_analysis = analyze_weight_spectrum(original_weight)
#                 optimal_rank = compute_optimal_rank(
#                     spectrum_analysis['singular_values'],
#                     config.energy_threshold
#                 )
#                 logger.info(f"  Auto-selected rank: {optimal_rank}")
#                 rank = optimal_rank
#             else:
#                 rank = config.r
#                 logger.info(f"  Using configured rank: {rank}")
            
#             max_rank = min(original_weight.shape)
#             rank = min(rank, max_rank)
            
#             # A_init, B_init, W_residual = initialize_lora_with_svd(
#             #     original_weight,
#             #     rank,
#             #     include_sigma=True
#             # )
#             U, S, Vt = torch.linalg.svd(original_weight, full_matrices=False)
#             U_r = U[:, :rank]   # (out_features, r)
#             Vt_r = Vt[:rank, :] # (r, in_features)
            

#             for adapter_name in module.lora_A.keys():
#                 if rank <= module.lora_A[adapter_name].weight.shape[0]:
#                     module.lora_B[adapter_name].weight.data = U_r.to(
#                         device=module.lora_B[adapter_name].weight.device,
#                         dtype=module.lora_B[adapter_name].weight.dtype
#                     )
#                     torch.nn.init.zeros_(module.lora_A[adapter_name].weight)
#                     # if hasattr(module, 'base_layer') and hasattr(module.base_layer, 'weight'):
#                     #     module.base_layer.weight.data = W_residual
                    
#                     logger.info(f"  ✓ Applied SVD initialization to adapter: {adapter_name}")
#                 else:
#                     logger.info(f"  ✗ Rank mismatch for adapter {adapter_name}, skipping...")
    
#     logger.info("\n" + "="*60)
#     logger.info("SVD initialization completed!")
#     logger.info("="*60 + "\n")


def apply_svd_initialization_to_model(
    model: PeftModel,
    original_model: nn.Module,
    config: SVDLoraConfig,
    target_modules: Optional[List[str]] = None
):
    """
    Apply SVD-init to PEFT Model
    """
    if not config.use_svd_init:
        logger.info("SVD initialization is disabled.")
        return
    
    logger.info("\n" + "="*60)
    logger.info("Applying SVD-based LoRA initialization (Corrected)...")
    logger.info("="*60)
    
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            
            if not hasattr(module, 'base_layer') or not hasattr(module.base_layer, 'weight'):
                logger.warning(f"Skipping module {name}: Cannot find base_layer or its weight.")
                continue
            
            original_weight = module.base_layer.weight.data.clone()
            # -------------------------------------------------

            logger.info(f"\nInitializing module: {name}")
            logger.info(f"   Original weight shape: {original_weight.shape}")
            
            if config.auto_rank_selection:
                logger.info(f"   pass Auto Rank Selection!!!!!!!!!!!!!")
                pass
            else:
                rank = config.r
                logger.info(f"   Using configured rank: {rank}")
            
            max_rank = min(original_weight.shape)
            rank = min(rank, max_rank)
            
            A_init, B_init, W_residual = initialize_lora_with_svd(
                original_weight,
                rank,
                include_sigma=True
            )
            
            for adapter_name in module.lora_A.keys():
                if rank <= module.lora_A[adapter_name].weight.shape[0]:
                    module.lora_A[adapter_name].weight.data = A_init[:rank, :].contiguous()
                    module.lora_B[adapter_name].weight.data = B_init[:, :rank].contiguous()
                    
                    if hasattr(module, 'base_layer') and hasattr(module.base_layer, 'weight'):
                        logger.info(f"   base_layer.weight.data is not updated.")
                        module.base_layer.weight.data = W_residual
                    
                    logger.info(f"   ✓ Applied SVD initialization to adapter: {adapter_name}")
                else:
                    logger.info(f"   ✗ Rank mismatch for adapter {adapter_name}, skipping...")
    
    logger.info("\n" + "="*60)
    logger.info("SVD initialization completed!")
    logger.info("="*60 + "\n")

def create_svd_lora_model(
    base_model: nn.Module,
    rank: int = 8,
    alpha: int = 16,
    target_modules: Optional[List[str]] = None,
    use_svd_init: bool = True,
    auto_rank_selection: bool = False,
    energy_threshold: float = 0.9,
    task_type: Optional[str] = None,
    **kwargs
):
    """
    Build Lora Model Contain SVD-Init
    
    Args:
        base_model: Pretrained Model
        rank: r
        target_modules: target module name list
        use_svd_init: use svd init?
        auto_rank_selection: use auto select rank?
        energy_threshold: auto select rank threshold
        task_type:
    
    Returns:
        SVD-LoRA PEFT model
    """
    if task_type is None:
        model_class_name = base_model.__class__.__name__
        if 'ForCausalLM' in model_class_name:
            task_type = "CAUSAL_LM"
        elif 'ForSequenceClassification' in model_class_name:
            task_type = "SEQ_CLS"
        elif 'ForTokenClassification' in model_class_name:
            task_type = "TOKEN_CLS"
        elif 'ForQuestionAnswering' in model_class_name:
            task_type = "QUESTION_ANS"
        elif 'ForSeq2SeqLM' in model_class_name or 'ForConditionalGeneration' in model_class_name:
            task_type = "SEQ_2_SEQ_LM"
        else:
            task_type = "FEATURE_EXTRACTION"
            logger.info(f"Warning: Could not auto-detect task type for {model_class_name}, using FEATURE_EXTRACTION")
    
    logger.info(f"Using task_type: {task_type}")
    
    config = SVDLoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0,
        bias="none",
        task_type=task_type,
        use_svd_init=use_svd_init,
        auto_rank_selection=auto_rank_selection,
        energy_threshold=energy_threshold,
        **kwargs
    )
    
    original_model = base_model
    
    model = get_peft_model(base_model, config)
    
    if use_svd_init:
        apply_svd_initialization_to_model(
            model,
            original_model,
            config,
            target_modules
        )
    
    return model


def analyze_model_spectrum(model: nn.Module, target_modules: Optional[List[str]] = None):
    """
    Analyze target module singular value spectrum
    """
    logger.info("\n" + "="*60)
    logger.info("Analyzing Model Weight Spectrum")
    logger.info("="*60)
    
    for name, module in model.named_modules():
        if target_modules and not any(target in name for target in target_modules):
            continue
        
        if hasattr(module, 'weight'):
            weight = module.weight.data
            if weight.dim() == 2:
                logger.info(f"\n{name}:")
                logger.info(f"  Shape: {weight.shape}")
                
                analysis = analyze_weight_spectrum(weight)
                logger.info(f"  Spectral decay: {analysis['spectral_decay']:.2e}")
                logger.info(f"  Rank suggestions:")
                for threshold, rank in analysis['rank_suggestions'].items():
                    logger.info(f"    {threshold} energy: rank {rank}")


def get_target_modules_for_model(model: nn.Module) -> List[str]:
    target_modules = set()
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module_name = name.split('.')[-1]
            target_modules.add(module_name)
    
    common_attention_modules = [
        'q_lin', 'k_lin', 'v_lin',
        'query', 'key', 'value',
    ]
    
    attention_modules = [m for m in target_modules if any(attn in m for attn in common_attention_modules)]
    
    if attention_modules:
        logger.info(f"Detected attention modules: {attention_modules}")
        return attention_modules
    else:
        logger.info(f"No standard attention modules found. Using all Linear layers: {list(target_modules)[:10]}...")
        return list(target_modules)


if __name__ == "__main__":
    logger.info("SVD-based LoRA Implementation for PEFT")
    logger.info("="*60)
    
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "./distilbert-base-uncased",
        num_labels=2
    )
    target_modules = get_target_modules_for_model(base_model)
    logger.info(f"\nTarget modules: {target_modules}")
    
    analyze_model_spectrum(base_model, target_modules=target_modules)
    
    logger.info("\n\nCreating SVD-LoRA model...")
    svd_lora_model = create_svd_lora_model(
        base_model,
        rank=8,
        target_modules=target_modules,
        use_svd_init=True,
        auto_rank_selection=True,
        energy_threshold=0.9
    )
    
    logger.info("\nModel created successfully!")
    logger.info(svd_lora_model)
    
    logger.info("\n" + "="*60)
    logger.info("Usage example for real models:")
    logger.info("="*60)