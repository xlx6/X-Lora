import argparse
import json
import os
from datetime import datetime
from typing import List
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)

from src.x_lora.data import load_sst2, compute_metrics, count_trainable_params
from src.x_lora.train import (
    load_base_model,
    build_baseline_lora,
    build_svd_lora,
    build_orthogonal_lora,
    build_structured_lora,
    DEFAULT_TARGETS_DISTILBERT,
    LoRAMonitorCallback,
)
from src.x_lora.models import OrthogonalLoRATrainer
    def __init__(self, output_dir):
        self.output_file = os.path.join(output_dir, "lora_metrics.jsonl")
        self.prev_weights = {}
        os.makedirs(output_dir, exist_ok=True)
        open(self.output_file, 'w').close()
        self._logged_debug = False

    def _get_lora_params(self, model):
        """
        Target structure: ...layer.0...query.lora_A.default.weight
        """
        lora_groups = {}
        pattern = re.compile(r"(.*)\.(lora_[AB])\.default\.weight$")
        
        found_count = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            match = pattern.match(name)
            if match:
                prefix = match.group(1)
                type_name = match.group(2)
                
                if prefix not in lora_groups:
                    lora_groups[prefix] = {}
                
                lora_groups[prefix][type_name] = param
                found_count += 1
                
                if not self._logged_debug:
                    print(f"[LoRAMonitor] Detected: {name}")
                    print(f"    -> Group: {prefix}")
                    print(f"    -> Type:  {type_name}")
                    print(f"    -> Shape: {param.shape}")

        if found_count > 0:
            self._logged_debug = True
        
        return {k: v for k, v in lora_groups.items() if "lora_A" in v and "lora_B" in v}

    @torch.no_grad()
    def on_step_end(self, args, state, control, model=None, **kwargs):
        current_groups = self._get_lora_params(model)
        
        metrics = {
            "step": state.global_step,
            "epoch": state.epoch
        }

        for module_name, params in current_groups.items():
            A = params["lora_A"]
            B = params["lora_B"]
            
            short_name = module_name.split("bert.")[-1] if "bert." in module_name else module_name

            metrics[f"{short_name}/norm_A"] = torch.norm(A).item()
            metrics[f"{short_name}/norm_B"] = torch.norm(B).item()

            if A.grad is not None:
                metrics[f"{short_name}/grad_norm_A"] = torch.norm(A.grad).item()
            if B.grad is not None:
                metrics[f"{short_name}/grad_norm_B"] = torch.norm(B.grad).item()

            if module_name in self.prev_weights:
                device = A.device
                A_old = self.prev_weights[module_name]["A"].to(device)
                B_old = self.prev_weights[module_name]["B"].to(device)

                Delta_A = A - A_old
                Delta_B = B - B_old

                metrics[f"{short_name}/delta_A_norm"] = torch.norm(Delta_A).item()
                metrics[f"{short_name}/delta_B_norm"] = torch.norm(Delta_B).item()
                
                def matmul_lora(mat_B, mat_A):
                    if mat_B.shape[1] == mat_A.shape[0]:
                        return mat_B @ mat_A
                    elif mat_B.shape[1] == mat_A.shape[1]:
                        return mat_B @ mat_A.T
                    else:
                        return mat_B @ mat_A # Fallback

                # Term 1: || (Delta B) A ||
                term1 = matmul_lora(Delta_B, A)
                metrics[f"{short_name}/eff_deltaB_A"] = torch.norm(term1).item()

                # Term 2: || B (Delta A) ||
                term2 = matmul_lora(B, Delta_A)
                metrics[f"{short_name}/eff_B_deltaA"] = torch.norm(term2).item()

                # Term 3: || Delta(BA) || = || B_new A_new - B_old A_old ||
                BA_new = matmul_lora(B, A)
                BA_old = matmul_lora(B_old, A_old)
                diff = BA_new - BA_old
                metrics[f"{short_name}/eff_delta_BA"] = torch.norm(diff).item()

            self.prev_weights[module_name] = {
                "A": A.detach().cpu().clone(),
                "B": B.detach().cpu().clone()
            }

        if len(metrics) > 2:
            with open(self.output_file, "a") as f:
                f.write(json.dumps(metrics) + "\n")

def parse_args():
    p = argparse.ArgumentParser(description="Train DistilBERT on SST-2 with LoRA variants")
    p.add_argument("--method", type=str, required=True,
                   choices=["baseline", "svd", "ortho", "struct"],
                   help="Which method to train")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--targets", type=str, nargs="*", default=DEFAULT_TARGETS_DISTILBERT,
                   help="Target module names for LoRA application")
    p.add_argument('--exp_name', type=str)

    # Model and data
    p.add_argument("--model_name_or_path", type=str, default="distilbert-base-uncased")
    p.add_argument("--hf_cache", type=str, default=None)
    p.add_argument("--max_length", type=int, default=128)

    # Training hyperparameters (fixed per README unless user overrides)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--bf16", type=bool, default=True)
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--scheduler", type=str, default="linear")

    # Orthogonal-specific
    p.add_argument("--orthogonal_lambda", type=float, default=1e-1)
    p.add_argument("--use_qr", type=bool, default=False)
    p.add_argument("--qr_frequency", type=int, default=1)

    # Structured-specific
    p.add_argument("--structure_type", type=str, default="lu", choices=["lu", "cholesky"]) \
        
    # SVD-specific
    p.add_argument("--svd_auto_rank", type=bool, default=False)
    p.add_argument("--svd_energy_threshold", type=float, default=0.9)
    p.add_argument("--mode", type=str, default='standard')

    # Outputs
    p.add_argument("--output_dir", type=str, default=None)
    return p.parse_args()




def main():
    args = parse_args()
    set_seed(args.seed)

    run_name = f"{args.method}_r{args.rank}"
    if args.method == "ortho":
        run_name += f"_lam{args.orthogonal_lambda:g}"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or os.path.join("outputs", args.method, run_name + f"_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    if args.report_to == 'swanlab':
        import swanlab
        swanlab.login(api_key=os.environ['SWANLAB_API_KEY'])
        swanlab.init(
            project="LoRA-SVD",
            experiment_name=args.exp_name
        )
    # Tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.hf_cache)
    train_ds, eval_ds, label2id, id2label = load_sst2(tokenizer, max_length=args.max_length, cache_dir=args.hf_cache)

    # Base model
    model = load_base_model(args.model_name_or_path, num_labels=2, label2id=label2id, id2label=id2label)
    # freeze base model
    for param in model.parameters():
        param.requires_grad = False
    # print(model)
    # Build method-specific model
    if args.method == "baseline":
        model = build_baseline_lora(model, r=args.rank, alpha=args.lora_alpha, dropout=args.lora_dropout, targets=args.targets)
        trainer_cls = Trainer
        trainer_kwargs = {}
    elif args.method == "svd":
        model = build_svd_lora(model, r=args.rank, alpha=args.lora_alpha, targets=args.targets,
                               use_svd_init=True, auto_rank_selection=args.svd_auto_rank,
                               energy_threshold=args.svd_energy_threshold, mode=args.mode)
        trainer_cls = Trainer
        trainer_kwargs = {}
    elif args.method == "ortho":
        model = build_orthogonal_lora(model, r=args.rank, alpha=args.lora_alpha, dropout=args.lora_dropout,
                                      targets=args.targets, orthogonal_reg=True, use_qr=args.use_qr,
                                      qr_frequency=args.qr_frequency)
        trainer_cls = OrthogonalLoRATrainer
        trainer_kwargs = {
            "orthogonal_lambda": args.orthogonal_lambda,
            "use_qr": args.use_qr,
            "qr_frequency": args.qr_frequency,
        }
    elif args.method == "struct":
        model = build_structured_lora(model, r=args.rank, alpha=args.lora_alpha, dropout=args.lora_dropout,
                                      targets=args.targets, structure_type=args.structure_type, use_v2=False)
        trainer_cls = Trainer
        trainer_kwargs = {}
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer)
    print(model)
    # Unfreeze classifier
    for name, param in model.named_parameters():
        if "classifier" in name or "pre_classifier" in name:
            param.requires_grad = True

    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"Trainable params: {trainable} / {total} ({100*trainable/total:.2f}%)")

    # TrainingArguments per README
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=50,
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        save_total_limit=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=args.bf16,
        report_to=args.report_to,
        seed=args.seed,
        lr_scheduler_type=args.scheduler
    )
    # print(model)
    # Trainer
    monitor_callback = LoRAMonitorCallback(output_dir=f'./svd_outputs/{args.mode}')
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[monitor_callback],
        **trainer_kwargs,
    )

    # Train + evaluate
    train_result = trainer.train()
    metrics = trainer.evaluate()

    # Count trainable params
    trainable_params = count_trainable_params(model)

    # Persist summary
    summary = {
        "method": args.method,
        "rank": args.rank,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "orthogonal_lambda": args.orthogonal_lambda if args.method == "ortho" else None,
        "structure_type": args.structure_type if args.method == "struct" else None,
        "svd_auto_rank": args.svd_auto_rank if args.method == "svd" else None,
        "svd_energy_threshold": args.svd_energy_threshold if args.method == "svd" else None,
        "trainable_params": int(trainable_params),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "output_dir": output_dir,
    }
    with open(os.path.join(output_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Run Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.report_to == 'swanlab':
        swanlab.finish()

if __name__ == "__main__":
    main()
