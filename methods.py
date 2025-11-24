from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

from SVDInit import create_svd_lora_model, get_target_modules_for_model
from OrthogonalLoRA import apply_orthogonal_lora_to_model, OrthogonalLoRATrainer
from StructuredLoRA import apply_structured_lora_to_model


DEFAULT_TARGETS_DISTILBERT = ["q_lin", "k_lin", "v_lin"]


def load_base_model(model_name_or_path: str,
                    num_labels: int = 2,
                    label2id: Optional[dict] = None,
                    id2label: Optional[dict] = None):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )
    return model


def build_baseline_lora(model: nn.Module,
                        r: int,
                        alpha: int = 16,
                        dropout: float = 0.0,
                        targets: Optional[List[str]] = None):
    targets = targets or DEFAULT_TARGETS_DISTILBERT
    # print(targets)
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=targets,
    )
    model = get_peft_model(model, config)
    return model


def build_svd_lora(model: nn.Module,
                   r: int,
                   alpha: int,
                   targets: Optional[List[str]] = None,
                   use_svd_init: bool = True,
                   auto_rank_selection: bool = False,
                   energy_threshold: float = 0.9):
    targets = targets or DEFAULT_TARGETS_DISTILBERT
    svd_model = create_svd_lora_model(
        base_model=model,
        rank=r,
        alpha=alpha,
        target_modules=targets,
        use_svd_init=use_svd_init,
        auto_rank_selection=auto_rank_selection,
        energy_threshold=energy_threshold,
    )
    return svd_model


def build_orthogonal_lora(model: nn.Module,
                          r: int,
                          alpha: int = 16,
                          dropout: float = 0.0,
                          targets: Optional[List[str]] = None,
                          orthogonal_reg: bool = True,
                          use_qr: bool = True,
                          qr_frequency: int = 1):
    targets = DEFAULT_TARGETS_DISTILBERT
    model = apply_orthogonal_lora_to_model(
        model,
        target_modules=targets,
        rank=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        orthogonal_reg=orthogonal_reg,
        use_qr=use_qr,
        qr_frequency=qr_frequency,
    )
    return model


def build_structured_lora(model: nn.Module,
                          r: int,
                          alpha: int = 16,
                          dropout: float = 0.0,
                          targets: Optional[List[str]] = None,
                          structure_type: str = "lu",
                          use_v2: bool = False):
    targets = targets or DEFAULT_TARGETS_DISTILBERT
    model = apply_structured_lora_to_model(
        model,
        target_modules=targets,
        rank=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        structure_type=structure_type,
        use_v2=use_v2,
    )
    return model
