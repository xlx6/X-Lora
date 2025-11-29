import os
from typing import Dict, Tuple, Optional
import numpy as np
import torch
from datasets import load_dataset


def load_sst2(tokenizer, max_length: int = 128, cache_dir: Optional[str] = None):
    """Load GLUE/SST-2 and tokenize with the provided tokenizer.

    Returns (train_dataset, eval_dataset, label2id, id2label)
    """
    raw = load_dataset('./sst2')

    def preprocess(ex):
        return tokenizer(ex["sentence"], truncation=True, padding=False, max_length=max_length)

    encoded = raw.map(preprocess, batched=True, remove_columns=["sentence"])
    encoded = encoded.rename_column("label", "labels")
    encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    label2id = {"negative": 0, "positive": 1}
    id2label = {0: "negative", 1: "positive"}
    return encoded["train"], encoded["validation"], label2id, id2label


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Compute accuracy and F1 for binary classification without external deps."""
    logits, labels = eval_pred
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)

    correct = (preds == labels).sum().item() if isinstance(labels, torch.Tensor) else (preds == labels).sum()
    total = labels.shape[0]
    acc = correct / max(1, total)

    # Binary F1 (positive class = 1)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    # Convert possible numpy scalars
    return {
        "accuracy": float(acc),
        "f1": float(f1),
    }


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

