"""
Training callbacks for monitoring LoRA training.
"""
import os
import json
import re
import torch
from transformers import TrainerCallback


class LoRAMonitorCallback(TrainerCallback):
    """Callback to monitor LoRA parameter dynamics during training."""
    
    def __init__(self, output_dir):
        self.output_file = os.path.join(output_dir, "lora_metrics.jsonl")
        self.prev_weights = {}
        os.makedirs(output_dir, exist_ok=True)
        open(self.output_file, 'w').close()
        self._logged_debug = False

    def _get_lora_params(self, model):
        """
        Extract LoRA parameters from model.
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
                        return mat_B @ mat_A

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

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        groups = self._get_lora_params(model)
        for name, params in groups.items():
            self.prev_weights[name] = {
                "A": params["lora_A"].detach().cpu().clone(),
                "B": params["lora_B"].detach().cpu().clone()
            }

