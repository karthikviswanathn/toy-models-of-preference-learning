#!/usr/bin/env python3
"""Generate individual PNG figures for a given hyperparameter config.

Saves to writeup/figs/wd{wd}_bs{bs}_ms{ms}_ss{ss}_sh{sh}/.

Usage:
    sbatch run_job.sh analysis/checks/generate_figs.py --wd 0.15 --bs 1024 --ms 1234 --ss 42 --sh 44
    python analysis/checks/generate_figs.py --wd 0.15 --bs 1024 --ms 1234 --ss 42 --sh 44 --device cuda
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
from trainer.utils import get_fourier_basis, get_fourier_basis_names


def find_model(variant_dir, pattern):
    """Find the model.pt matching a glob pattern under variant_dir."""
    matches = sorted(PROJECT_ROOT.glob(f"outputs/runs/{variant_dir}/{pattern}/model.pt"))
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
    p = 113
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Left: PT + POST overplotted
    ax = axes[0]
    m1, s1, _ = ax.stem(range(len(pow_pt)), pow_pt, markerfmt="o", basefmt="k-", label="PT")
    m1.set_markersize(3); m1.set_color("tab:blue"); s1.set_color("tab:blue"); s1.set_alpha(0.5)
    m2, s2, _ = ax.stem(range(len(pow_sft)), pow_sft, markerfmt="o", basefmt="k-", label="POST")
    m2.set_markersize(3); m2.set_color("tab:red"); s2.set_color("tab:red"); s2.set_alpha(0.5)
    ax.set_xticks(ticks); ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_ylabel("Power", fontsize=11); ax.set_title("PT + POST", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    # Right: PT-G
    ax = axes[1]
    m3, s3, _ = ax.stem(range(len(pow_ptg)), pow_ptg, markerfmt="o", basefmt="k-")
    m3.set_markersize(3); m3.set_color("tab:green"); s3.set_color("tab:green"); s3.set_alpha(0.5)
    ax.set_xticks(ticks); ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_ylabel("Power", fontsize=11); ax.set_title("PT-G", fontsize=13, fontweight="bold")

    fig.suptitle("Fourier Power Spectrum of $W_E$ (Number Tokens)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fourier_embedding.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_fourier_neuron_logit(out_dir, a_pt, a_sft, a_ptg):
    """Fourier spectrum of W_logit = W_out @ W_U, 3 separate panels."""
    p = 113
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

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    for ax, power, label, color in [
        (axes[0], pow_pt, "PT", "tab:blue"),
        (axes[1], pow_sft, "POST", "tab:red"),
        (axes[2], pow_ptg, "PT-G", "tab:green"),
    ]:
        markerline, stemlines, baseline = ax.stem(
            range(len(power)), power, markerfmt="o", basefmt="k-"
        )
        markerline.set_markersize(3)
        markerline.set_color(color)
        stemlines.set_color(color)
        stemlines.set_alpha(0.5)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
        ax.set_ylabel("Power", fontsize=11)
        ax.set_title(label, fontsize=13, fontweight="bold")

    fig.suptitle(
        "Fourier Power Spectrum of $W_{\\mathrm{logit}} = W_{\\mathrm{out}} W_U$",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fourier_neuron_logit.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate analysis figure PNGs")
    parser.add_argument("--wd", type=float, required=True, help="Weight decay")
    parser.add_argument("--bs", type=int, required=True, help="Batch size (-1 for full)")
    parser.add_argument("--ms", type=int, required=True, help="Model seed")
    parser.add_argument("--ss", type=int, required=True, help="Split seed")
    parser.add_argument("--sh", type=int, required=True, help="Shuffle seed")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    suffix = f"wd{args.wd}_bs{args.bs}_ms{args.ms}_ss{args.ss}_sh{args.sh}"
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
    print("\n[1/3] PCA stages parity...")
    save_pca_stages(out_dir, a_pt, a_sft, a_ptg)

    print("[2/3] Fourier embedding spectrum...")
    save_fourier_embedding(out_dir, a_pt, a_sft, a_ptg)

    print("[3/3] Neuron-logit Fourier spectrum...")
    save_fourier_neuron_logit(out_dir, a_pt, a_sft, a_ptg)

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
