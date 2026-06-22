"""PCA pairwise plots (PC1–PC7) for best pretrained (PT) model, colored by preference.

Layout: 2 rows (stages) x 21 cols (PC pairs), saved as PNG.
"""
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.analyzer import ModelAnalyzer, load_model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pcs", type=int, default=7)
    args = parser.parse_args()

    pc_pairs = list(combinations(range(args.pcs), 2))

    model_path = PROJECT_ROOT / "outputs" / "runs" / "pt_lumi_16428523" / "model.pt"
    label = "PT wd=0.5 ms=1234 ds=42 bs=512 (loss=1.55e-07)"

    print(f"Loading {model_path}")
    model = load_model(model_path)
    analyzer = ModelAnalyzer(model, task="pt", device=args.device, label=label)

    output_dir = PROJECT_ROOT / "outputs" / "figs" / "top20_pca"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = analyzer.plot_pca(color_by="preference", pc_pairs=pc_pairs)
    out_path = output_dir / "pca_pairwise_preference_pretrained.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
