#!/usr/bin/env python3
"""CCDF plot of hybrid model accuracy on standard modular addition."""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def main():
    with open(PROJECT_ROOT / "outputs/runs-p106/hybrid_summary.csv") as f:
        rows = list(csv.DictReader(f))

    accs = np.array(sorted([float(r["hybrid_acc"]) for r in rows], reverse=True))
    cdf = np.arange(1, len(accs) + 1) / len(accs)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(accs, cdf, where="post", color="tab:blue", linewidth=2)
    ax.fill_between(accs, cdf, step="post", alpha=0.15, color="tab:blue")

    ax.set_xlabel("Hybrid accuracy on standard modular addition", fontsize=12)
    ax.set_ylabel("Fraction of runs", fontsize=12)
    ax.set_title("POST MLP activations + PT neuron-logit map\nevaluated on standard modular addition", fontsize=13, fontweight="bold")
    ax.set_xlim(1.02, -0.02)  # inverted: 1 on left, 0 on right
    ax.set_ylim(0, 1.05)
    ax.axvline(0.99, color="gray", linestyle="--", alpha=0.5, label="99% acc")
    ax.axvline(0.90, color="gray", linestyle=":", alpha=0.5, label="90% acc")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    out_path = PROJECT_ROOT / "writeup" / "figs" / "hybrid_cdf.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
