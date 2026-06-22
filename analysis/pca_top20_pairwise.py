"""PCA pairwise plots (PC1–PC7) for top 20 LUMI models, colored by preference.

One PDF page per model, each page: 21 rows (PC pairs) x 2 cols (stages).
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.analyzer import ModelAnalyzer, STAGES, load_model

TOP20_JSON = PROJECT_ROOT / ".ai-code" / "wandb-analysis" / "top20_runs.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figs" / "top20_pca"


def plot_pairwise_grid(analyzer, pc_pairs, title):
    """Plot pairwise PCs as a grid: rows = PC pairs, cols = stages."""
    stage_names = [s[0] for s in STAGES]
    n_pairs = len(pc_pairs)
    max_pc = max(max(i, j) for i, j in pc_pairs) + 1
    pca_results = analyzer.pca(n_components=max_pc)

    ncols = len(stage_names)
    nrows = n_pairs
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    preferred = analyzer.preference_labels

    for col_idx, stage_name in enumerate(stage_names):
        pr = pca_results[stage_name]
        var = pr.explained_variance_ratio * 100

        for row_idx, (i, j) in enumerate(pc_pairs):
            ax = axes[row_idx, col_idx]
            ax.scatter(pr.projections[preferred, i], pr.projections[preferred, j],
                       c="tab:blue", s=2, alpha=0.4, rasterized=True)
            ax.scatter(pr.projections[~preferred, i], pr.projections[~preferred, j],
                       c="tab:red", s=2, alpha=0.4, rasterized=True)

            ax.set_xlabel(f"PC{i+1} ({var[i]:.1f}%)", fontsize=11)
            ax.set_ylabel(f"PC{j+1} ({var[j]:.1f}%)", fontsize=11)
            ax.tick_params(labelsize=9)
            ax.locator_params(axis="both", nbins=5)

            if row_idx == 0:
                ax.set_title(stage_name, fontsize=13, fontweight="bold", pad=10)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markersize=6, label="preferred"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:red", markersize=6, label="unpreferred"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=11,
               frameon=True, bbox_to_anchor=(0.5, -0.005))

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.001)
    fig.tight_layout()
    return fig


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pcs", type=int, default=7)
    args = parser.parse_args()

    pc_pairs = list(combinations(range(args.pcs), 2))

    with open(TOP20_JSON) as f:
        runs = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "pca_pairwise_preference.pdf"

    with PdfPages(pdf_path) as pdf:
        for i, r in enumerate(runs):
            jid = r["name"].split("_")[-1]
            model_path = PROJECT_ROOT / "outputs" / "runs" / f"ptg_lumi_{jid}" / "model.pt"
            label = f"wd={r['weight_decay']} ms={r['model_seed']} ds={r['data_seed']} bs={r['batch_size']} (loss={r['best_test_loss']:.2e})"

            print(f"[{i+1}/20] {label}")
            model = load_model(model_path)
            analyzer = ModelAnalyzer(model, device=args.device, label=label)
            fig = plot_pairwise_grid(analyzer, pc_pairs, label)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
