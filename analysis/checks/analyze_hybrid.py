#!/usr/bin/env python3
"""Hybrid model analysis: replace POST unembedding with PT unembedding.

Evaluates three models on standard modular addition:
  1. PT (base pretrained model)
  2. POST (SFT model)
  3. Hybrid (POST body + PT unembedding)

If the hybrid recovers PT-level accuracy, it confirms POST preserves
the pretrained computation and only patches the output projection.

Usage:
    python analysis/analyze_hybrid.py \
        --pairs pt_dir1 sft_dir1 pt_dir2 sft_dir2 ...
"""

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.analyzer import load_model
from trainer.config import ModelConfig, DataConfig
from trainer.data import generate_all_data, train_test_split
from trainer.utils import eval_model
from trainer.tokenizer import ModularAdditionTokenizer


def create_hybrid_model(pretrained, sft_model):
    """Create hybrid: SFT base + PT's W_out and W_U.

    Copies W_out (MLP output projection) and W_U (unembedding) from the
    pretrained model into a deep copy of the SFT model.
    """
    hybrid = copy.deepcopy(sft_model)

    with torch.no_grad():
        # W_out: (d_mlp, d_model)
        hybrid.blocks[0].mlp.W_out.copy_(pretrained.blocks[0].mlp.W_out)
        hybrid.blocks[0].mlp.b_out.copy_(pretrained.blocks[0].mlp.b_out)

        # W_U: (d_model, vocab)
        hybrid.unembed.W_U.copy_(pretrained.unembed.W_U)
        if hybrid.unembed.b_U is not None and pretrained.unembed.b_U is not None:
            hybrid.unembed.b_U.copy_(pretrained.unembed.b_U)

    return hybrid


def main():
    parser = argparse.ArgumentParser(description="Hybrid model: POST body + PT unembedding")
    parser.add_argument("--pairs", nargs="+", type=Path, required=True,
                        help="Alternating PT and SFT run directories: pt1 sft1 pt2 sft2 ...")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--csv", type=Path, default=None, help="Output CSV path")
    args = parser.parse_args()

    if len(args.pairs) % 2 != 0:
        parser.error("--pairs must have an even number of arguments (pt1 sft1 pt2 sft2 ...)")

    mc = ModelConfig()
    tokenizer = ModularAdditionTokenizer(mc.p)

    # Header
    print(f"{'wd':>6} {'bs':>6} {'ms':>6} {'ss':>4} {'sh':>4} {'sft_best_loss':>14} {'hyb_loss':>12} {'hyb_acc':>8} {'hyb_even':>8} {'hyb_odd':>8}")
    print("-" * 90)

    rows = []
    for i in range(0, len(args.pairs), 2):
        pt_dir = args.pairs[i]
        sft_dir = args.pairs[i + 1]

        # Read config for split seed and params
        with open(pt_dir / "config.json") as f:
            cfg = json.load(f)
        split_seed = cfg.get("split_seed", cfg.get("seed", 42))
        train_frac = cfg["train_frac"]
        wd = cfg["weight_decay"]
        bs = cfg["batch_size"]
        ms = cfg.get("model_seed", 1234)
        sh = cfg.get("shuffle_seed", 43)

        # Get SFT best test loss from its training history
        sft_history = torch.load(sft_dir / "history.pt", map_location="cpu", weights_only=False)
        sft_best_loss = min(sft_history["test_loss"])

        # Generate standard modular addition test data with matching split
        rng = np.random.default_rng(split_seed)
        inputs, labels = generate_all_data(tokenizer, device=args.device)
        _, _, test_x, test_y = train_test_split(inputs, labels, train_frac, rng)
        test_m = torch.zeros(len(test_x), 5, device=args.device)
        test_m[:, 3:5] = 1.0
        test_even = (test_x[:, 1] + test_x[:, 2]) % mc.p % 2 == 0

        # Load models
        pt_model = load_model(pt_dir / "model.pt")
        pt_model.to(args.device)
        sft_model = load_model(sft_dir / "model.pt")
        sft_model.to(args.device)

        # Create and evaluate hybrid (SFT base + PT's W_out and W_U)
        hybrid = create_hybrid_model(pt_model, sft_model)
        hybrid.to(args.device)
        hyb_loss, hyb_acc, hyb_even, hyb_odd = eval_model(hybrid, test_x, test_y, test_m, test_even)

        bs_str = "full" if bs <= 0 else str(bs)
        print(f"{wd:>6} {bs_str:>6} {ms:>6} {split_seed:>4} {sh:>4} {sft_best_loss:>14.2e} {hyb_loss:>12.2e} {hyb_acc:>8.4f} {hyb_even:>8.4f} {hyb_odd:>8.4f}")
        rows.append({
            "weight_decay": wd, "batch_size": bs,
            "model_seed": ms, "split_seed": split_seed, "shuffle_seed": sh,
            "sft_best_test_loss": sft_best_loss,
            "hybrid_test_loss": hyb_loss, "hybrid_acc": hyb_acc,
            "hybrid_acc_even": hyb_even, "hybrid_acc_odd": hyb_odd,
        })

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
