#!/usr/bin/env python3
"""Generate all poster figures (POST vs PT-G only, poster-optimized).

Usage:
    sbatch run_job.sh poster/poster_figs.py --device cuda
    python poster/poster_figs.py --device cpu
    python poster/poster_figs.py --device cuda --fig pca
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.analyzer import ModelAnalyzer, load_model
from sklearn.decomposition import PCA

# ---- Poster style ----
POSTER_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 20,
    "axes.titlesize": 26,
    "axes.labelsize": 22,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 18,
    "lines.linewidth": 2.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}
matplotlib.rcParams.update(POSTER_RC)

# Colors
EVEN_COLOR = "#1f77b4"   # blue
ODD_COLOR = "#d62728"    # red
POST_COLOR = "#E8734A"   # orange
PTG_COLOR = "#6BCB77"    # green
PT_COLOR = "#9CA3AF"     # gray (reference)

OUT_DIR = PROJECT_ROOT / "poster" / "figs"


def find_model(variant_dir, pattern):
    matches = sorted(PROJECT_ROOT.glob(f"outputs/runs/{variant_dir}/{pattern}/model.pt"))
    if not matches:
        raise FileNotFoundError(f"No model found for {variant_dir}/{pattern}")
    return matches[0]


def load_analyzers(device):
    suffix = "wd0.15_bs1024_ms1234_ss42_sh44"
    pt_path = find_model("pt", f"pt_{suffix}_*")
    sft_path = find_model("sft", f"sft_{suffix}_*")
    ptg_path = find_model("pt-g", f"ptg_{suffix}_*")

    print(f"PT:   {pt_path.parent.name}")
    print(f"POST: {sft_path.parent.name}")
    print(f"PT-G: {ptg_path.parent.name}")

    a_pt = ModelAnalyzer(load_model(pt_path), task="pt", device=device, label="PT")
    a_sft = ModelAnalyzer(load_model(sft_path), task="ptg", device=device, label="POST")
    a_ptg = ModelAnalyzer(load_model(ptg_path), task="ptg", device=device, label="PT-G")
    return a_pt, a_sft, a_ptg


# ============================================================
# Figure 1: Ensemble Evidence (paper fig 2)
# ============================================================
def save_ensemble_evidence(out_dir):
    """Two-panel: (a) probe accuracy bar, (b) CDF of test loss."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Panel (a): Probe accuracy at Post-Attn & Post-MLP ---
    csv_path = PROJECT_ROOT / "outputs/runs/parity_probes_sweep.csv"
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    locations = ["Post-Attn", "Post-MLP"]
    loc_keys = ["post_attn", "post_mlp"]
    variants = ["PT", "POST", "PT-G"]
    var_keys = ["pt", "post", "ptg"]
    colors = [PT_COLOR, POST_COLOR, PTG_COLOR]

    means = {v: [] for v in variants}
    stds = {v: [] for v in variants}
    for loc_key in loc_keys:
        for var, vk in zip(variants, var_keys):
            col = f"{loc_key}_{vk}"
            vals = [float(r[col]) for r in rows]
            means[var].append(np.mean(vals))
            stds[var].append(np.std(vals))

    ax = axes[0]
    x = np.arange(len(locations))
    width = 0.25
    for i, (var, color) in enumerate(zip(variants, colors)):
        offset = (i - 1) * width
        ax.bar(x + offset, means[var], width, yerr=stds[var],
               label=var, color=color, edgecolor="white", linewidth=0.5,
               capsize=4, error_kw={"linewidth": 1.5})

    ax.set_xticks(x)
    ax.set_xticklabels(locations)
    ax.set_ylabel("Probe Accuracy")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylim(0.3, 1.1)
    ax.legend(loc="upper left")
    ax.set_title("(a) Parity probe accuracy", fontweight="bold")

    # --- Panel (b): CDF of test loss ---
    merged_path = PROJECT_ROOT / "outputs/runs/merged_summary.csv"
    with open(merged_path) as f:
        mrows = list(csv.DictReader(f))

    sft_loss = np.array(sorted([float(r["sft_best_test_loss"]) for r in mrows]))
    ptg_loss = np.array(sorted([float(r["ptg_best_test_loss"]) for r in mrows]))

    ax = axes[1]
    cdf = np.arange(1, len(sft_loss) + 1) / len(sft_loss)
    ax.step(np.log10(sft_loss), cdf, where="post", color=POST_COLOR, linewidth=2.5, label="POST")
    ax.step(np.log10(ptg_loss), cdf, where="post", color=PTG_COLOR, linewidth=2.5, label="PT-G")
    ax.set_xlabel("log$_{10}$(best test loss)")
    ax.set_ylabel("CDF")
    ax.set_title("(b) Test loss across sweep", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout(w_pad=3)
    fig.savefig(out_dir / "ensemble_evidence.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "ensemble_evidence.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved ensemble_evidence")


# ============================================================
# Figure 2: PCA Comparison (2x2: POST vs PT-G × Post-Attn vs Post-MLP)
# ============================================================
def save_pca_comparison(out_dir, a_sft, a_ptg):
    """2x2 PCA grid: rows=stages, cols=POST/PT-G, colored by parity."""
    stages = [
        ("blocks.0.hook_resid_mid", "Post-Attention"),
        ("blocks.0.hook_resid_post", "Post-MLP"),
    ]
    analyzers = [a_sft, a_ptg]
    eq_pos = 3

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)

    for row, (hook, stage_name) in enumerate(stages):
        for col, a in enumerate(analyzers):
            act = a.cache[hook][:, eq_pos].cpu().numpy()
            pca = PCA(n_components=2)
            proj = pca.fit_transform(act)
            var = pca.explained_variance_ratio_ * 100

            ax = axes[row, col]
            even = a.parity_labels
            ax.scatter(proj[even, 0], proj[even, 1],
                       c=EVEN_COLOR, s=4, alpha=0.4, rasterized=True, label="even")
            ax.scatter(proj[~even, 0], proj[~even, 1],
                       c=ODD_COLOR, s=4, alpha=0.4, rasterized=True, label="odd")
            ax.set_xlabel(f"PC1 ({var[0]:.1f}%)", fontsize=14)
            ax.set_ylabel(f"PC2 ({var[1]:.1f}%)", fontsize=14)
            ax.tick_params(labelsize=11)
            ax.locator_params(axis="both", nbins=5)

            if row == 0:
                ax.set_title(a.label, fontsize=20, fontweight="bold", pad=10)

        # Row label
        axes[row, 0].annotate(
            stage_name, xy=(0, 0.5), xytext=(-0.35, 0.5),
            xycoords="axes fraction", textcoords="axes fraction",
            fontsize=16, fontweight="bold", rotation=90, ha="center", va="center",
        )

    # Legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=EVEN_COLOR, markersize=10, label="even"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=ODD_COLOR, markersize=10, label="odd"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=15,
               frameon=True, bbox_to_anchor=(0.5, -0.06))

    fig.subplots_adjust(hspace=0.35, wspace=0.35, left=0.15, top=0.93)
    fig.savefig(out_dir / "pca_comparison.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "pca_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved pca_comparison")


# ============================================================
# Figure 3: Probe Bars (canonical, POST + PT-G only)
# ============================================================
def save_probe_bars(out_dir):
    """Linear probe accuracy at 6 locations, POST vs PT-G."""
    # Hardcoded canonical data from canonical_probe_bar.py
    canonical_row = "0.15,1024,1234,42,44,0.4902,0.4832,0.4952,0.4954,0.4850,0.6763,0.4811,0.4779,0.4894,0.4855,0.6171,0.7940,0.4879,0.4842,0.4855,0.4853,0.6395,0.6996,0.4858,0.4960,0.4889,0.4931,0.4832,0.6434,0.4886,0.4910,0.4842,0.5038,0.8880,0.9982,0.5158,0.5513,0.5730,0.9616,0.9997,1.0000"
    vals = [float(v) for v in canonical_row.split(",")[5:]]

    LOCATIONS = ["Head 0", "Head 1", "Head 2", "Head 3", "Post-Attn", "Post-MLP"]

    # Parse: each location has 6 values: pt_lin, pt_aug, post_lin, post_aug, ptg_lin, ptg_aug
    post_linear = []
    ptg_linear = []
    for i in range(6):
        offset = i * 6
        post_linear.append(vals[offset + 2])
        ptg_linear.append(vals[offset + 4])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(LOCATIONS))
    width = 0.35

    ax.bar(x - width/2, post_linear, width, label="POST", color=POST_COLOR, edgecolor="white", linewidth=0.5)
    ax.bar(x + width/2, ptg_linear, width, label="PT-G", color=PTG_COLOR, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(LOCATIONS, rotation=30, ha="right")
    ax.set_ylabel("Parity Probe Accuracy", fontsize=18)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_dir / "probe_bars.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "probe_bars.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved probe_bars")


# ============================================================
# Figure 4: Augmented Probes (linear vs PC^2, POST + PT-G)
# ============================================================
def save_augmented_probes(out_dir):
    """2-row: top=POST (Post-Attn, Post-MLP), bottom=PT-G (Head 0-3). Linear vs PC^2."""
    csv_path = PROJECT_ROOT / "outputs/runs/modified_probes_sweep.csv"
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    # Top row: POST at Post-Attn, Post-MLP
    post_locs = ["Post-Attn", "Post-MLP"]
    post_keys = ["post_attn", "post_mlp"]
    # Bottom row: PT-G heads (sorted descending per config)
    head_keys = ["head_0", "head_1", "head_2", "head_3"]

    def get_stats(loc_keys, variant_key):
        lin_means, lin_stds, aug_means, aug_stds = [], [], [], []
        for lk in loc_keys:
            lin_vals = [float(r[f"{lk}_{variant_key}"]) for r in rows]
            aug_vals = [float(r[f"{lk}_{variant_key}_aug"]) for r in rows]
            lin_means.append(np.mean(lin_vals))
            lin_stds.append(np.std(lin_vals))
            aug_means.append(np.mean(aug_vals))
            aug_stds.append(np.std(aug_vals))
        return lin_means, lin_stds, aug_means, aug_stds

    def get_sorted_head_stats(variant_key):
        """Sort 4 head scores descending within each config, then aggregate."""
        lin_sorted = []  # shape: (n_configs, 4)
        aug_sorted = []
        for r in rows:
            lin_vals = [float(r[f"{hk}_{variant_key}"]) for hk in head_keys]
            aug_vals = [float(r[f"{hk}_{variant_key}_aug"]) for hk in head_keys]
            # Sort by augmented score descending (consistent ordering)
            order = np.argsort(aug_vals)[::-1]
            lin_sorted.append([lin_vals[i] for i in order])
            aug_sorted.append([aug_vals[i] for i in order])
        lin_sorted = np.array(lin_sorted)  # (126, 4)
        aug_sorted = np.array(aug_sorted)
        return (lin_sorted.mean(0).tolist(), lin_sorted.std(0).tolist(),
                aug_sorted.mean(0).tolist(), aug_sorted.std(0).tolist())

    ptg_locs = ["Head 0*", "Head 1*", "Head 2*", "Head 3*"]

    POST_DARK = "#B5380A"  # darker orange for scatter dots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8),
                             gridspec_kw={"height_ratios": [1, 1]})

    # --- Top row: POST scatter (linear vs augmented) ---
    for col, (loc_name, lk) in enumerate(zip(post_locs, post_keys)):
        ax = axes[0, col]
        lin_vals = np.array([float(r[f"{lk}_post"]) for r in rows])
        aug_vals = np.array([float(r[f"{lk}_post_aug"]) for r in rows])

        ax.scatter(lin_vals, aug_vals, c=POST_DARK, s=25, alpha=0.6,
                   edgecolors="white", linewidth=0.3)
        lo = min(lin_vals.min(), aug_vals.min()) - 0.02
        hi = max(lin_vals.max(), aug_vals.max()) + 0.02
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, alpha=0.5)
        ax.set_xlabel("Linear")
        ax.set_title(loc_name, fontweight="bold", fontsize=20)
        ax.set_aspect("equal")
    axes[0, 0].set_ylabel("Augmented")
    # Shared suptitle for top row
    fig.text(0.5, 0.97, "POST", fontsize=28, fontweight="bold",
             color=POST_COLOR, ha="center")

    # --- Bottom row: PT-G sorted heads (merge into single wide axes) ---
    # Remove individual bottom axes and create a spanning one
    axes[1, 0].remove()
    axes[1, 1].remove()
    ax = fig.add_subplot(2, 1, 2)

    width = 0.3
    x = np.arange(len(ptg_locs))
    lm, ls, am, astd = get_sorted_head_stats("ptg")
    ax.bar(x - width/2, lm, width, yerr=ls,
           label="Linear", color=PTG_COLOR, alpha=0.3, edgecolor=PTG_COLOR, linewidth=1.5,
           capsize=5, error_kw={"linewidth": 1.5})
    ax.bar(x + width/2, am, width, yerr=astd,
           label="+ Squared PCs", color=PTG_COLOR, alpha=1.0, edgecolor="white", linewidth=0.5,
           capsize=5, error_kw={"linewidth": 1.5})
    ax.set_title("PT-G", fontweight="bold", color=PTG_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(ptg_locs)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylim(0.3, 1.15)
    ax.set_ylabel("Probe Accuracy", fontsize=18)
    ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout(h_pad=1)
    fig.savefig(out_dir / "augmented_probes.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "augmented_probes.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved augmented_probes")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Generate poster figures")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fig", type=str, default="all",
                        choices=["all", "ensemble", "pca", "probes", "augmented"])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy existing diagrams
    for src, dst in [
        ("writeup/figs/diagram.pdf", "setup_diagram.pdf"),
    ]:
        src_path = PROJECT_ROOT / src
        dst_path = OUT_DIR / dst
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            print(f"  Copied {dst}")

    # CSV-only figures (no model loading needed)
    if args.fig in ("all", "ensemble"):
        print("\n[1] Ensemble evidence...")
        save_ensemble_evidence(OUT_DIR)

    if args.fig in ("all", "probes"):
        print("\n[3] Probe bars (canonical)...")
        save_probe_bars(OUT_DIR)

    if args.fig in ("all", "augmented"):
        print("\n[4] Augmented probes...")
        save_augmented_probes(OUT_DIR)

    # Model-dependent figures
    if args.fig in ("all", "pca"):
        print("\nLoading models...")
        a_pt, a_sft, a_ptg = load_analyzers(args.device)

        print("\n[2] PCA comparison...")
        save_pca_comparison(OUT_DIR, a_sft, a_ptg)

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
