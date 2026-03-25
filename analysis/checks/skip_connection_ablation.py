#!/usr/bin/env python3
"""Run skip-connection ablation on all 126 x 3 models and output CSV.

For each model (PT, POST, PT-G), computes:
  - original loss/acc (resid_mid + mlp_out)
  - zero-ablated loss/acc (mlp_out only, skip connection zeroed)
  - mean-ablated loss/acc (mlp_out + mean(resid_mid))

Usage:
    python analysis/checks/skip_connection_ablation.py --device cuda \
        --csv outputs/runs-p106/skip_ablation.csv
"""

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.analyzer import ModelAnalyzer, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "outputs/runs-p106/skip_ablation.csv")
    args = parser.parse_args()

    variants = [
        ("PT", "pt", "pt_*", "pt"),
        ("POST", "sft", "sft_*", "ptg"),
        ("PT-G", "pt-g", "ptg_*", "ptg"),
    ]

    rows = []
    for variant_name, subdir, pattern, task in variants:
        run_dirs = sorted(glob.glob(str(PROJECT_ROOT / f"outputs/runs-p106/{subdir}/{pattern}/")))
        print(f"\n=== {variant_name}: {len(run_dirs)} runs ===")

        for run_dir in run_dirs:
            run_dir = Path(run_dir)
            config_path = run_dir / "config.json"
            model_path = run_dir / "model.pt"
            if not config_path.exists() or not model_path.exists():
                continue

            with open(config_path) as f:
                cfg = json.load(f)

            wd = cfg["weight_decay"]
            bs = cfg["batch_size"]
            ms = cfg.get("model_seed", 1234)
            ds = cfg.get("data_seed", cfg.get("split_seed", 42))

            model = load_model(model_path)
            analyzer = ModelAnalyzer(model, task=task, device=args.device)
            result = analyzer.test_skip_connection_ablation()

            print(f"  {variant_name} wd={wd} bs={bs} ms={ms} ds={ds} "
                  f"orig_acc={result['original_acc']:.4f} "
                  f"zero_acc={result['zero_ablated_acc']:.4f} "
                  f"mean_acc={result['mean_ablated_acc']:.4f}")

            rows.append({
                "variant": variant_name,
                "weight_decay": wd,
                "batch_size": bs,
                "model_seed": ms,
                "data_seed": ds,
                "original_loss": result["original_loss"],
                "original_acc": result["original_acc"],
                "zero_ablated_loss": result["zero_ablated_loss"],
                "zero_ablated_acc": result["zero_ablated_acc"],
                "mean_ablated_loss": result["mean_ablated_loss"],
                "mean_ablated_acc": result["mean_ablated_acc"],
                "loss_ratio_zero": result["loss_ratio_zero"],
                "loss_ratio_mean": result["loss_ratio_mean"],
            })

            # Free memory
            del model, analyzer

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
