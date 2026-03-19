#!/usr/bin/env python3
"""PCA analysis of '=' token — generates a multi-page PDF with all colorings.

Usage (single model):
    python toy_models_of_preference_training/code/analyze_embeddings.py \
        --model path/to/model.pt

Usage (multi-model comparison):
    python toy_models_of_preference_training/code/analyze_embeddings.py \
        --model path/to/ptg.pt --label "PT-G" \
        --model path/to/sft.pt --label "SFT"

Outputs: pca_analysis.pdf with pages for parity, result, a, b colorings.
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.analyzer import ModelAnalyzer, load_model

COLORINGS = ["parity", "result", "a", "b"]


def main():
    parser = argparse.ArgumentParser(
        description='PCA analysis of "=" token — all colorings as PDF')
    parser.add_argument("--model", type=Path, action="append", required=True,
                        help="Model path (can specify multiple)")
    parser.add_argument("--label", action="append", default=None,
                        help="Label for each model (same order as --model)")
    parser.add_argument("--pcs", type=int, default=2,
                        help="Number of PCs (2 = just PC1 vs PC2, >2 = pairwise)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    labels = args.label or [None] * len(args.model)
    if len(labels) < len(args.model):
        labels.extend([None] * (len(args.model) - len(labels)))

    analyzers = []
    for model_path, label in zip(args.model, labels):
        model = load_model(model_path)
        print(f"Loaded {model_path.name}: {model.cfg.n_layers}L{model.cfg.n_heads}H d={model.cfg.d_model}")
        analyzers.append(ModelAnalyzer(model, device=args.device, label=label))

    output_dir = args.output or (PROJECT_ROOT / "outputs" / "figs")
    output_dir.mkdir(parents=True, exist_ok=True)

    pc_pairs = list(combinations(range(args.pcs), 2)) if args.pcs > 2 else None

    if len(analyzers) == 1:
        a = analyzers[0]
        pdf_path = output_dir / "pca_analysis.pdf"
        with PdfPages(pdf_path) as pdf:
            for color_by in COLORINGS:
                fig = a.plot_pca(color_by=color_by, pc_pairs=pc_pairs)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                print(f"  Added page: {color_by}")
        print(f"Saved: {pdf_path}")
    else:
        pdf_path = output_dir / "pca_comparison.pdf"
        with PdfPages(pdf_path) as pdf:
            for color_by in COLORINGS:
                fig = ModelAnalyzer.compare_pca(analyzers, color_by=color_by)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                print(f"  Added page: {color_by}")
        print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
