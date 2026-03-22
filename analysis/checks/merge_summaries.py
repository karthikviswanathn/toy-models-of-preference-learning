#!/usr/bin/env python3
"""Merge per-variant summary CSVs into a single merged_summary.csv.

Joins pt/summary.csv, pt-g/summary.csv, sft/summary.csv, and hybrid_summary.csv
on the shared key columns (weight_decay, batch_size, model_seed, split_seed, shuffle_seed).

Usage:
    python analysis/checks/merge_summaries.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS = PROJECT_ROOT / "outputs" / "runs"

KEYS = ["weight_decay", "batch_size", "model_seed", "split_seed", "shuffle_seed"]

VARIANTS = [
    ("pt",  RUNS / "pt"  / "summary.csv"),
    ("ptg", RUNS / "pt-g" / "summary.csv"),
    ("sft", RUNS / "sft" / "summary.csv"),
]


def main():
    # Load and prefix each variant's summary
    merged = None
    for prefix, path in VARIANTS:
        df = pd.read_csv(path)
        rename = {c: f"{prefix}_{c}" for c in df.columns if c not in KEYS}
        df = df.rename(columns=rename)
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=KEYS, how="outer")

    # Load hybrid summary
    hybrid = pd.read_csv(RUNS / "hybrid_summary.csv")
    # Drop sft_best_test_loss (redundant with sft_best_test_loss from sft/summary.csv)
    hybrid = hybrid.drop(columns=["sft_best_test_loss"], errors="ignore")
    merged = merged.merge(hybrid, on=KEYS, how="outer")

    # Keep only the 10 columns we care about
    cols = KEYS + [
        "pt_best_test_loss", "ptg_best_test_loss", "sft_best_test_loss",
        "hybrid_test_loss", "hybrid_acc",
    ]
    merged = merged[cols]

    out = RUNS / "merged_summary.csv"
    merged.to_csv(out, index=False)
    print(f"Wrote {len(merged)} rows x {len(merged.columns)} cols to {out}")


if __name__ == "__main__":
    main()
