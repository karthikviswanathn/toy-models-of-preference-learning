#!/usr/bin/env python3
"""Plot Fourier spectra of W_E and W_U for PT and POST overlaid.

Two subplots (W_E, W_U), each with PT (circles) and POST (x markers) overlaid.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.analyzer import ModelAnalyzer, load_model

device = "cuda"
out_dir = PROJECT_ROOT / "writeup" / "figs"

def find_model(variant_dir, pattern):
    matches = sorted(PROJECT_ROOT.glob(f"outputs/runs-p106/{variant_dir}/{pattern}/model.pt"))
    if not matches:
        raise FileNotFoundError(f"No model found for {variant_dir}/{pattern}")
    return matches[0]

pt_model = load_model(find_model("pt", "pt_wd0.15_bs1024_ms1236_ds42_*"))
post_model = load_model(find_model("sft", "sft_wd0.15_bs1024_ms1236_ds42_*"))

pt_a = ModelAnalyzer(pt_model, task="pt", device=device, label="PT")
post_a = ModelAnalyzer(post_model, task="ptg", device=device, label="POST")

fig, axes = plt.subplots(2, 1, figsize=(16, 8))

for row_idx, kind in enumerate(["embedding", "unembed"]):
    ax = axes[row_idx]

    pt_fd = pt_a.fourier_embedding() if kind == "embedding" else pt_a.fourier_unembed()
    post_fd = post_a.fourier_embedding() if kind == "embedding" else post_a.fourier_unembed()

    pt_power = pt_fd["power_per_freq"]
    post_power = post_fd["power_per_freq"]
    if isinstance(pt_power, torch.Tensor):
        pt_power = pt_power.cpu().numpy()
    if isinstance(post_power, torch.Tensor):
        post_power = post_power.cpu().numpy()
    names = pt_fd["fourier_names"]

    x = np.arange(len(pt_power))
    ax.scatter(x, pt_power, marker="o", s=20, color="tab:blue", zorder=3, label="PT")
    ax.scatter(x, post_power, marker="x", s=20, color="tab:red", zorder=4, label="POST")
    ax.vlines(x, 0, np.maximum(pt_power, post_power), colors="gray", alpha=0.3, linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=5)
    ax.set_ylabel("Power", fontsize=11)
    w_name = "$W_E$" if kind == "embedding" else "$W_U$"
    ax.set_title(f"Fourier power spectrum of {w_name}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)

fig.suptitle("PT vs POST Fourier spectra", fontsize=14, fontweight="bold")
fig.tight_layout()
out_path = out_dir / "fourier_pt_post.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
