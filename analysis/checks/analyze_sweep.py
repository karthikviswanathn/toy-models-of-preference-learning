#!/usr/bin/env python3
"""PCA of '=' token across a sweep of runs, colored by parity.

Usage:
    python analysis/analyze_sweep.py \
        outputs/runs-p106/pt-g/ptg_wd0.3_bs256_... \
        outputs/runs-p106/pt-g/ptg_wd0.5_bs512_... \
        ...
        --filename pca_top_ptg.png
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.analyzer import (
    STAGES, generate_parity_gated_inputs, extract_stage_activations, load_model,
)
from trainer.config import ModelConfig, DataConfig
from trainer.data import generate_all_data, train_test_split
from trainer.utils import generate_parity_gated_data, split_data, eval_model
from trainer.tokenizer import ModularAdditionTokenizer


def main():
    parser = argparse.ArgumentParser(description='PCA of "=" token across runs')
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories to analyze")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--filename", default="pca_sweep.png")
    args = parser.parse_args()

    mc = ModelConfig()
    dc = DataConfig()
    tokenizer = ModularAdditionTokenizer(mc.p)
    all_inputs, result_labels = generate_parity_gated_inputs(tokenizer, "cpu")
    print(f"Generated {len(all_inputs)} inputs (p={mc.p})")

    # Load metadata and activations for each run
    entries = []
    for run_dir in args.runs:
        with open(run_dir / "config.json") as f:
            cfg = json.load(f)
        wd = cfg["weight_decay"]
        bs = cfg["batch_size"]
        variant = cfg.get("variant", "PT-G")
        train_frac = cfg["train_frac"]
        data_seed = cfg.get("data_seed", cfg.get("split_seed", 42))

        rng = np.random.default_rng(data_seed)
        if variant == "PT":
            inputs, labels = generate_all_data(tokenizer, device=args.device)
            _, _, test_x, test_y = train_test_split(inputs, labels, train_frac, rng)
            test_m = torch.zeros(len(test_x), 5, device=args.device)
            test_m[:, 3:5] = 1.0
            test_even = (test_x[:, 1] + test_x[:, 2]) % mc.p % 2 == 0
        else:
            inputs, labels, loss_mask, is_even = generate_parity_gated_data(tokenizer, device=args.device)
            _, _, _, _, test_x, test_y, test_m, test_even = split_data(
                inputs, labels, loss_mask, is_even, train_frac, rng,
            )

        model = load_model(run_dir / "model.pt")
        model.to(args.device)
        test_loss, test_acc, _, _ = eval_model(model, test_x, test_y, test_m, test_even)
        print(f"  {variant} wd={wd}, bs={bs}, test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")
        activations = extract_stage_activations(model, all_inputs, args.device)
        entries.append((wd, bs, test_loss, activations))

    # Plot: rows = stages, columns = runs (in input order)
    even = result_labels % 2 == 0
    odd = ~even
    row_labels = [s[0] for s in STAGES]
    nrows = len(row_labels)
    ncols = len(entries)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows + 0.8),
                             squeeze=False)

    for col_idx, (wd, bs, test_loss, activations) in enumerate(entries):
        for row_idx, row_label in enumerate(row_labels):
            ax = axes[row_idx, col_idx]
            data = activations[row_label]

            pca = PCA(n_components=2)
            proj = pca.fit_transform(data)

            ax.scatter(proj[even, 0], proj[even, 1],
                       c="tab:blue", s=2, alpha=0.4, rasterized=True)
            ax.scatter(proj[odd, 0], proj[odd, 1],
                       c="tab:red", s=2, alpha=0.4, rasterized=True)

            ax.tick_params(labelsize=9)
            ax.locator_params(axis="both", nbins=5)

            if row_idx == 0:
                if test_loss < 1e-4:
                    loss_str = f"{test_loss:.2e}"
                else:
                    loss_str = f"{test_loss:.4f}"
                bs_str = "full" if bs <= 0 else str(bs)
                ax.set_title(f"wd={wd}, bs={bs_str}\n(loss={loss_str})", fontsize=10, fontweight="bold", pad=8)

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
                    markersize=6, label="even"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:red",
                    markersize=6, label="odd"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=11,
               frameon=True, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle('Residual stream PCA of the "=" token',
                 fontsize=14, fontweight="bold")
    fig.subplots_adjust(hspace=0.45, wspace=0.4, left=0.07)

    output_dir = args.output or (PROJECT_ROOT / "outputs" / "figs")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / args.filename
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
