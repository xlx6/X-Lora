import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os

def get_model_spectrum_data(model, target_modules=None, max_rank=100):
    layers_s = []
    layer_names = []
    
    print("Extracting singular values...")
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        
        if target_modules:
            if not any(t in name for t in target_modules):
                continue
        
        w = module.weight.data.float()
        
        try:
            S = torch.linalg.svdvals(w)
        except:
            _, S, _ = torch.linalg.svd(w, full_matrices=False)
            
        s_top = S[:max_rank].cpu().numpy()
        
        if len(s_top) < max_rank:
            s_top = np.pad(s_top, (0, max_rank - len(s_top)))
            
        layers_s.append(s_top)
        short_name = name.replace("base_model.model.", "").replace("distilbert.", "")
        layer_names.append(short_name)
        
    return np.array(layers_s), layer_names

def plot_dual_energy_heatmap(S_data, layer_names, top_r=50, save_dir="visual", filename="energy_heatmap_dual.png"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    S_normalized = S_data / (S_data[:, 0:1] + 1e-9)
    
    energy = S_data ** 2
    cumulative_energy = np.cumsum(energy, axis=1)
    total_energy = np.sum(energy, axis=1, keepdims=True) + 1e-9
    cumulative_ratio = cumulative_energy / total_energy

    fig, axes = plt.subplots(1, 2, figsize=(24, len(layer_names) * 0.5))
    
    sns.heatmap(
        S_normalized[:, :top_r], 
        ax=axes[0],
        cmap="viridis",
        vmin=0.0, vmax=1.0,
        cbar_kws={'label': 'Normalized Singular Value'}
    )
    axes[0].set_title(f'Normalized Singular Value Spectrum (Top {top_r})', fontsize=16)
    axes[0].set_xlabel('Rank (r)', fontsize=14)
    axes[0].set_ylabel('Layer Name', fontsize=14)
    axes[0].set_yticks(np.arange(len(layer_names)) + 0.5)
    axes[0].set_yticklabels(layer_names, rotation=0, fontsize=10)

    sns.heatmap(
        cumulative_ratio[:, :top_r], 
        ax=axes[1],
        cmap="magma",
        vmin=0.0, vmax=1.0,
        cbar_kws={'label': 'Cumulative Energy Ratio'}
    )
    axes[1].set_title(f'Cumulative Energy Ratio (How fast info is captured)', fontsize=16)
    axes[1].set_xlabel('Rank (r)', fontsize=14)
    axes[1].set_yticks(np.arange(len(layer_names)) + 0.5)
    axes[1].set_yticklabels(layer_names, rotation=0, fontsize=10)

    plt.tight_layout()
    
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Dual Heatmap saved to: {save_path}")
    
    plt.close()
from transformers import AutoModelForSequenceClassification

base_model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')


S_data, names = get_model_spectrum_data(base_model, target_modules=['q_lin', 'v_lin', 'k_lin'])

plot_dual_energy_heatmap(S_data, names, top_r=64, filename="energy_heatmap_dual_distil.png")