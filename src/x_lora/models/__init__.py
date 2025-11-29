"""
LoRA model implementations.
"""

from .orthogonal_lora import (
    OrthogonalLoRALayer,
    LinearWithOrthogonalLoRA,
    OrthogonalLoRATrainer,
    apply_orthogonal_lora_to_model,
)
from .structured_lora import (
    TriangularMatrix,
    StructuredLoRALayer,
    StructuredLoRALayerV2,
    LinearWithStructuredLoRA,
    StructuredLoRATrainer,
    apply_structured_lora_to_model,
)
from .svd_lora import (
    SVDLoraConfig,
    compute_optimal_rank,
    create_svd_lora_model,
    get_target_modules_for_model,
)

__all__ = [
    # Orthogonal LoRA
    "OrthogonalLoRALayer",
    "LinearWithOrthogonalLoRA",
    "OrthogonalLoRATrainer",
    "apply_orthogonal_lora_to_model",
    # Structured LoRA
    "TriangularMatrix",
    "StructuredLoRALayer",
    "StructuredLoRALayerV2",
    "LinearWithStructuredLoRA",
    "StructuredLoRATrainer",
    "apply_structured_lora_to_model",
    # SVD LoRA
    "SVDLoraConfig",
    "compute_optimal_rank",
    "create_svd_lora_model",
    "get_target_modules_for_model",
]

