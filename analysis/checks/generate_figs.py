#!/usr/bin/env python3
"""Generate individual PNG figures for a given hyperparameter config.

Saves to writeup/figs/wd{wd}_bs{bs}_ms{ms}_ds{ds}/.

Usage:
    sbatch run_job.sh analysis/checks/generate_figs.py --wd 0.15 --bs 1024 --ms 1234 --ds 42
    python analysis/checks/generate_figs.py --wd 0.15 --bs 1024 --ms 1234 --ds 42 --device cuda
"""

import argparse
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
from sklearn.decomposition import PCA
from trainer.utils import get_fourier_basis, get_fourier_basis_names


def find_model(variant_dir, pattern):
    """Find the model.pt matching a glob pattern under variant_dir."""
    matches = sorted(PROJECT_ROOT.glob(f"outputs/runs-p106/{variant_dir}/{pattern}/model.pt"))
    if not matches:
        raise FileNotFoundError(f"No model found for {variant_dir}/{pattern}")
    return matches[0]


def save_pca_stages(out_dir, a_pt, a_sft, a_ptg):
    """PCA stages parity — PT, POST, PT-G with row labels."""
    fig = ModelAnalyzer.compare_pca([a_pt, a_sft, a_ptg], color_by="parity", n_components=2)
    fig.savefig(out_dir / "pca_stages_parity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_fourier_embedding(out_dir, a_pt, a_sft, a_ptg):
    """Fourier spectrum of W_E. PT+POST overplotted, PT-G separate."""
    p = a_pt.p
    names = get_fourier_basis_names(p)
    step = max(1, len(names) // 20)
    ticks = list(range(0, len(names), step))
    tick_labels = [names[t] for t in ticks]

    def get_power(analyzer):
        data = analyzer.fourier_embedding()
        power = data["power_per_freq"]
        return power.cpu().numpy() if isinstance(power, torch.Tensor) else power

    pow_pt = get_power(a_pt)
    pow_sft = get_power(a_sft)
    pow_ptg = get_power(a_ptg)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: PT + POST overplotted
    ax = axes[0]
    m1, s1, _ = ax.stem(range(len(pow_pt)), pow_pt, markerfmt="o", basefmt="k-", label="PT")
    m1.set_markersize(4); m1.set_color("tab:blue"); s1.set_color("tab:blue"); s1.set_alpha(0.5)
    m2, s2, _ = ax.stem(range(len(pow_sft)), pow_sft, markerfmt="o", basefmt="k-", label="POST")
    m2.set_markersize(4); m2.set_color("tab:red"); s2.set_color("tab:red"); s2.set_alpha(0.5)
    ax.set_xticks(ticks); ax.set_xticklabels(tick_labels, rotation=90, fontsize=10)
    ax.set_ylabel("Power", fontsize=14); ax.set_title("PT + POST", fontsize=16, fontweight="bold")
    ax.tick_params(axis='y', labelsize=11)
    ax.legend(fontsize=13)

    # Right: PT-G
    ax = axes[1]
    m3, s3, _ = ax.stem(range(len(pow_ptg)), pow_ptg, markerfmt="o", basefmt="k-")
    m3.set_markersize(4); m3.set_color("tab:green"); s3.set_color("tab:green"); s3.set_alpha(0.5)
    ax.set_xticks(ticks); ax.set_xticklabels(tick_labels, rotation=90, fontsize=10)
    ax.set_ylabel("Power", fontsize=14); ax.set_title("PT-G", fontsize=16, fontweight="bold")
    ax.tick_params(axis='y', labelsize=11)

    fig.suptitle("Fourier Power Spectrum of $W_E$ (Number Tokens)", fontsize=17, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fourier_embedding.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_fourier_neuron_logit(out_dir, a_pt, a_sft, a_ptg):
    """Fourier spectrum of W_logit = W_out @ W_U, normalized and overplotted (2 panels)."""
    p = a_pt.p
    fb = get_fourier_basis(p, "cpu")
    names = get_fourier_basis_names(p)
    step = max(1, len(names) // 20)
    ticks = list(range(0, len(names), step))
    tick_labels = [names[t] for t in ticks]

    def get_power(analyzer):
        model = analyzer.model
        W_out = model.blocks[0].mlp.W_out.detach().cpu()
        W_U = model.unembed.W_U.detach().cpu()[:, :p]
        W_logit = W_out @ W_U
        coeffs = W_logit @ fb.T
        return (coeffs ** 2).sum(dim=0).numpy()

    pow_pt = get_power(a_pt)
    pow_sft = get_power(a_sft)
    pow_ptg = get_power(a_ptg)

    # Normalize each by its max
    pow_pt = pow_pt / pow_pt.max()
    pow_sft = pow_sft / pow_sft.max()
    pow_ptg = pow_ptg / pow_ptg.max()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: PT + POST overplotted
    ax = axes[0]
    m1, s1, _ = ax.stem(range(len(pow_pt)), pow_pt, markerfmt="o", basefmt="k-", label="PT")
    m1.set_markersize(4); m1.set_color("tab:blue"); s1.set_color("tab:blue"); s1.set_alpha(0.5)
    m2, s2, _ = ax.stem(range(len(pow_sft)), pow_sft, markerfmt="o", basefmt="k-", label="POST")
    m2.set_markersize(4); m2.set_color("tab:red"); s2.set_color("tab:red"); s2.set_alpha(0.5)
    ax.set_xticks(ticks); ax.set_xticklabels(tick_labels, rotation=90, fontsize=10)
    ax.set_ylabel("Normalized Power", fontsize=14); ax.set_title("PT + POST", fontsize=16, fontweight="bold")
    ax.tick_params(axis='y', labelsize=11)
    ax.legend(fontsize=13)

    # Right: PT-G
    ax = axes[1]
    m3, s3, _ = ax.stem(range(len(pow_ptg)), pow_ptg, markerfmt="o", basefmt="k-")
    m3.set_markersize(4); m3.set_color("tab:green"); s3.set_color("tab:green"); s3.set_alpha(0.5)
    ax.set_xticks(ticks); ax.set_xticklabels(tick_labels, rotation=90, fontsize=10)
    ax.set_ylabel("Normalized Power", fontsize=14); ax.set_title("PT-G", fontsize=16, fontweight="bold")
    ax.tick_params(axis='y', labelsize=11)

    fig.suptitle(
        "Fourier Power Spectrum of $W_{\\mathrm{logit}} = W_{\\mathrm{out}} W_U$ (normalized)",
        fontsize=17, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fourier_neuron_logit.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_pca_per_head(out_dir, a_pt, a_sft, a_ptg):
    """Per-head PCA: 4 rows (heads) x 3 cols (PT, POST, PT-G), colored by parity."""
    analyzers = [a_pt, a_sft, a_ptg]
    n_heads = a_pt.model.cfg.n_heads  # 4
    eq_pos = 3

    fig, axes = plt.subplots(n_heads, 3, figsize=(12, 3 * n_heads), squeeze=False)

    for col, a in enumerate(analyzers):
        # hook_z shape: (batch, seq, n_heads, d_head)
        z = a.cache["blocks.0.attn.hook_z"][:, eq_pos].cpu().numpy()  # (p*p, n_heads, d_head)
        for h in range(n_heads):
            head_act = z[:, h, :]  # (p*p, d_head)
            pca = PCA(n_components=2)
            proj = pca.fit_transform(head_act)
            var = pca.explained_variance_ratio_ * 100

            ax = axes[h, col]
            even = a.parity_labels
            ax.scatter(proj[even, 0], proj[even, 1],
                       c="tab:blue", s=2, alpha=0.4, rasterized=True)
            ax.scatter(proj[~even, 0], proj[~even, 1],
                       c="tab:red", s=2, alpha=0.4, rasterized=True)
            ax.set_xlabel(f"PC1 ({var[0]:.1f}%)", fontsize=10)
            ax.set_ylabel(f"PC2 ({var[1]:.1f}%)", fontsize=10)
            ax.tick_params(labelsize=8)
            ax.locator_params(axis="both", nbins=5)
            if h == 0:
                ax.set_title(a.label, fontsize=13, fontweight="bold", pad=8)

    # Row labels
    for h in range(n_heads):
        axes[h, 0].annotate(
            f"Head {h}", xy=(0, 0.5), xytext=(-0.38, 0.5),
            xycoords="axes fraction", textcoords="axes fraction",
            fontsize=12, fontweight="bold", rotation=90, ha="center", va="center",
        )

    # Legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markersize=10, label="even"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:red", markersize=10, label="odd"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=13, frameon=True, bbox_to_anchor=(0.5, -0.01))

    fig.subplots_adjust(hspace=0.4, wspace=0.35, left=0.12, top=0.95)
    fig.savefig(out_dir / "pca_per_head.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate analysis figure PNGs")
    parser.add_argument("--wd", type=float, required=True, help="Weight decay")
    parser.add_argument("--bs", type=int, required=True, help="Batch size (-1 for full)")
    parser.add_argument("--ms", type=int, required=True, help="Model seed")
    parser.add_argument("--ds", type=int, required=True, help="Data seed")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    suffix = f"wd{args.wd}_bs{args.bs}_ms{args.ms}_ds{args.ds}"
    out_dir = PROJECT_ROOT / "writeup" / "figs" / suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config: {suffix}")
    print(f"Device: {args.device}")
    print(f"Output: {out_dir}")

    # Find models
    pt_path = find_model("pt", f"pt_{suffix}_*")
    ptg_path = find_model("pt-g", f"ptg_{suffix}_*")
    sft_path = find_model("sft", f"sft_{suffix}_*")
    print(f"PT:   {pt_path.parent.name}")
    print(f"PT-G: {ptg_path.parent.name}")
    print(f"SFT:  {sft_path.parent.name}")

    # Load models
    print("\nLoading models...")
    pt_model = load_model(pt_path)
    ptg_model = load_model(ptg_path)
    sft_model = load_model(sft_path)

    # Create analyzers
    print("Creating analyzers...")
    a_pt = ModelAnalyzer(pt_model, task="pt", device=args.device, label="PT")
    a_sft = ModelAnalyzer(sft_model, task="ptg", device=args.device, label="POST")
    a_ptg = ModelAnalyzer(ptg_model, task="ptg", device=args.device, label="PT-G")

    # Generate figures
    print("\n[1/4] PCA stages parity...")
    save_pca_stages(out_dir, a_pt, a_sft, a_ptg)

    print("[2/4] Fourier embedding spectrum...")
    save_fourier_embedding(out_dir, a_pt, a_sft, a_ptg)

    print("[3/4] Neuron-logit Fourier spectrum...")
    save_fourier_neuron_logit(out_dir, a_pt, a_sft, a_ptg)

    print("[4/4] Per-head PCA...")
    save_pca_per_head(out_dir, a_pt, a_sft, a_ptg)

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
