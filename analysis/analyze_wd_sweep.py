#!/usr/bin/env python3
"""PCA of '=' token across weight decay sweep, colored by preference.

Usage:
    python toy_models_of_preference_training/code/analyze_wd_sweep.py \
        toy_models_of_preference_training/runs/ptg_..._wd0.1_... \
        toy_models_of_preference_training/runs/ptg_..._wd0.2_... \
        ...
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.analyzer import (
    STAGES, generate_preference_gated_inputs, extract_stage_activations, load_model,
)
from trainer.config import ModelConfig, DataConfig
from trainer.utils import generate_preference_gated_data, split_data, eval_model
from trainer.tokenizer import ModularAdditionTokenizer


def main():
    parser = argparse.ArgumentParser(description="PCA of '=' token across weight decay sweep")
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories to analyze")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    mc = ModelConfig()
    dc = DataConfig()
    tokenizer = ModularAdditionTokenizer(mc.p)
    all_inputs, result_labels = generate_preference_gated_inputs(tokenizer, "cpu", unsafe_threshold=dc.unsafe_threshold)
    print(f"Generated {len(all_inputs)} inputs (p={mc.p})")

    # Prepare test split for computing loss
    inputs, labels, loss_mask, is_preferred = generate_preference_gated_data(tokenizer, device=args.device, unsafe_threshold=dc.unsafe_threshold)
    with open(args.runs[0] / "config.json") as f:
        train_frac = json.load(f)["train_frac"]
    rng = np.random.default_rng(dc.seed)
    _, _, _, _, test_x, test_y, test_m, test_preferred = split_data(
        inputs, labels, loss_mask, is_preferred, train_frac, rng,
    )

    # Load metadata and activations for each run
    entries = []
    for run_dir in args.runs:
        with open(run_dir / "config.json") as f:
            wd = json.load(f)["weight_decay"]
        model = load_model(run_dir / "model.pt")
        model.to(args.device)
        test_loss, test_acc, _, _ = eval_model(model, test_x, test_y, test_m, test_preferred)
        print(f"  wd={wd}, test_loss={test_loss:.4f}, test_acc={test_acc:.4f}, arch={model.cfg.n_layers}L{model.cfg.n_heads}H")
        activations = extract_stage_activations(model, all_inputs, args.device)
        entries.append((wd, test_loss, activations))

    # Sort by weight decay
    entries.sort(key=lambda e: e[0])

    # Plot: rows = stages, columns = weight decay values
    preferred = result_labels < dc.unsafe_threshold
    unpreferred = ~preferred
    row_labels = [s[0] for s in STAGES]
    nrows = len(row_labels)
    ncols = len(entries)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows + 0.8),
                             squeeze=False)

    for col_idx, (wd, test_loss, activations) in enumerate(entries):
        for row_idx, row_label in enumerate(row_labels):
            ax = axes[row_idx, col_idx]
            data = activations[row_label]

            pca = PCA(n_components=2)
            proj = pca.fit_transform(data)

            ax.scatter(proj[preferred, 0], proj[preferred, 1],
                       c="tab:blue", s=2, alpha=0.4, rasterized=True)
            ax.scatter(proj[unpreferred, 0], proj[unpreferred, 1],
                       c="tab:red", s=2, alpha=0.4, rasterized=True)

            ax.tick_params(labelsize=9)
            ax.locator_params(axis="both", nbins=5)

            if row_idx == 0:
                if test_loss < 1e-4:
                    loss_str = f"{test_loss:.2e}"
                else:
                    loss_str = f"{test_loss:.4f}"
                ax.set_title(f"wd={wd}\n(loss={loss_str})", fontsize=10, fontweight="bold", pad=8)

            var = pca.explained_variance_ratio_ * 100
            ax.set_xlabel(f"PC1 ({var[0]:.1f}%)", fontsize=11)
            ax.set_ylabel(f"PC2 ({var[1]:.1f}%)", fontsize=11)

    # Row labels on the left margin
    for row_idx, row_label in enumerate(row_labels):
        axes[row_idx, 0].annotate(
            row_label, xy=(0, 0.5), xytext=(-0.38, 0.5),
            xycoords="axes fraction", textcoords="axes fraction",
            fontsize=12, fontweight="bold", rotation=90,
            ha="center", va="center",
        )

    # Legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue",
                    markersize=6, label="preferred"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:red",
                    markersize=6, label="unpreferred"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=11,
               frameon=True, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle('Residual stream PCA of the "=" token — weight decay sweep',
                 fontsize=14, fontweight="bold")
    fig.subplots_adjust(hspace=0.45, wspace=0.4, left=0.07)

    output_dir = args.output or (PROJECT_ROOT / "outputs" / "figs")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "pca_wd_sweep.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
