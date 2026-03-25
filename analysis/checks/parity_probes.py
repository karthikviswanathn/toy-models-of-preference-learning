#!/usr/bin/env python3
"""Train linear parity probes at 6 network locations for PT, POST, PT-G.

Probes predict even/odd parity from activations at the = token position.
Saves results as CSV and prints a table to stdout.

Usage (single config):
    sbatch run_job.sh analysis/checks/parity_probes.py --wd 0.15 --bs 1024 --ms 1234 --ds 42 --device cuda

Usage (sweep over all 126 configs):
    sbatch run_job.sh analysis/checks/parity_probes.py --sweep --device cuda
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.analyzer import ModelAnalyzer, load_model
from trainer.config import ModelConfig

LOCATIONS = ["Head 0", "Head 1", "Head 2", "Head 3", "Post-Attn", "Post-MLP"]
KEYS = ["weight_decay", "batch_size", "model_seed", "data_seed"]


def find_model(variant_dir, pattern):
    matches = sorted(PROJECT_ROOT.glob(f"outputs/runs-p106/{variant_dir}/{pattern}/model.pt"))
    if not matches:
        return None
    return matches[0]


def train_parity_probe(X, parity_labels, seed=42, train_frac=0.7):
    """Train logistic regression probe, return test accuracy."""
    assert X.ndim == 2, f"Expected 2D input, got shape {X.shape}"
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    split = int(n * train_frac)
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y = parity_labels.astype(int)
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)


def extract_and_probe(analyzer, p, d_model, n_heads, d_head):
    """Extract activations at 6 locations and train parity probes. Returns list of accuracies."""
    eq_pos = 3
    accs = []

    # Per-head activations from hook_z
    z = analyzer.cache["blocks.0.attn.hook_z"][:, eq_pos].cpu().numpy()
    assert z.shape == (p * p, n_heads, d_head), f"hook_z shape: {z.shape}, expected {(p * p, n_heads, d_head)}"

    for h in range(n_heads):
        head_act = z[:, h, :]
        assert head_act.shape == (p * p, d_head), f"head {h} shape: {head_act.shape}"
        accs.append(train_parity_probe(head_act, analyzer.parity_labels))

    # Post-attention residual stream
    resid_mid = analyzer.cache["blocks.0.hook_resid_mid"][:, eq_pos].cpu().numpy()
    assert resid_mid.shape == (p * p, d_model), f"resid_mid shape: {resid_mid.shape}, expected {(p * p, d_model)}"
    accs.append(train_parity_probe(resid_mid, analyzer.parity_labels))

    # Post-MLP residual stream
    resid_post = analyzer.cache["blocks.0.hook_resid_post"][:, eq_pos].cpu().numpy()
    assert resid_post.shape == (p * p, d_model), f"resid_post shape: {resid_post.shape}, expected {(p * p, d_model)}"
    accs.append(train_parity_probe(resid_post, analyzer.parity_labels))

    return accs


def probe_single(args):
    """Run probes for a single hyperparameter config."""
    suffix = f"wd{args.wd}_bs{args.bs}_ms{args.ms}_ds{args.ds}"
    out_dir = PROJECT_ROOT / "writeup" / "figs" / suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config: {suffix}")
    print(f"Device: {args.device}")

    pt_path = find_model("pt", f"pt_{suffix}_*")
    ptg_path = find_model("pt-g", f"ptg_{suffix}_*")
    sft_path = find_model("sft", f"sft_{suffix}_*")
    print(f"PT:   {pt_path.parent.name}")
    print(f"PT-G: {ptg_path.parent.name}")
    print(f"SFT:  {sft_path.parent.name}")

    print("\nLoading models...")
    a_pt = ModelAnalyzer(load_model(pt_path), task="pt", device=args.device, label="PT")
    a_sft = ModelAnalyzer(load_model(sft_path), task="ptg", device=args.device, label="POST")
    a_ptg = ModelAnalyzer(load_model(ptg_path), task="ptg", device=args.device, label="PT-G")

    p = a_pt.p
    d_model = a_pt.model.cfg.d_model
    n_heads = a_pt.model.cfg.n_heads
    d_head = d_model // n_heads

    results = {}
    for label, analyzer in [("PT", a_pt), ("POST", a_sft), ("PT-G", a_ptg)]:
        print(f"\nProbing {label}...")
        accs = extract_and_probe(analyzer, p, d_model, n_heads, d_head)
        results[label] = accs
        for loc, acc in zip(LOCATIONS, accs):
            print(f"  {loc:10s}: {acc:.4f}")

    # Print table
    print("\n" + "=" * 50)
    print(f"{'Location':10s}  {'PT':>8s}  {'POST':>8s}  {'PT-G':>8s}")
    print("-" * 50)
    for i, loc in enumerate(LOCATIONS):
        print(f"{loc:10s}  {results['PT'][i]:8.4f}  {results['POST'][i]:8.4f}  {results['PT-G'][i]:8.4f}")
    print("=" * 50)

    # Save CSV
    csv_path = out_dir / "parity_probes.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["location", "PT", "POST", "PT-G"])
        for i, loc in enumerate(LOCATIONS):
            writer.writerow([loc, f"{results['PT'][i]:.4f}", f"{results['POST'][i]:.4f}", f"{results['PT-G'][i]:.4f}"])
    print(f"\nCSV saved: {csv_path}")


def probe_sweep(device):
    """Run probes across all 126 sweep configs and save a single CSV."""
    # Read configs from PT summary
    summary_path = PROJECT_ROOT / "outputs" / "runs-p106" / "pt" / "summary.csv"
    import pandas as pd
    df = pd.read_csv(summary_path)
    configs = df[KEYS].values
    n_configs = len(configs)
    print(f"Found {n_configs} configs from {summary_path}")
    print(f"Device: {device}")

    p = ModelConfig().p
    # Columns: 4 keys + 6 locations x 3 variants = 22 columns
    variants = ["pt", "post", "ptg"]
    loc_cols = []
    for loc in LOCATIONS:
        loc_tag = loc.lower().replace(" ", "_").replace("-", "_")
        for var in variants:
            loc_cols.append(f"{loc_tag}_{var}")

    out_path = PROJECT_ROOT / "outputs" / "runs-p106" / "parity_probes_sweep.csv"
    # Write header
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(KEYS + loc_cols)

    for i, (wd, bs, ms, ds) in enumerate(configs):
        bs = int(bs); ms = int(ms); ds = int(ds)
        suffix = f"wd{wd}_bs{bs}_ms{ms}_ds{ds}"
        print(f"\n[{i+1}/{n_configs}] {suffix}")

        pt_path = find_model("pt", f"pt_{suffix}_*")
        sft_path = find_model("sft", f"sft_{suffix}_*")
        ptg_path = find_model("pt-g", f"ptg_{suffix}_*")

        if not pt_path or not sft_path or not ptg_path:
            print(f"  SKIP — missing model(s)")
            continue

        row = [wd, bs, ms, ds]
        for variant, path, task in [("PT", pt_path, "pt"), ("POST", sft_path, "ptg"), ("PT-G", ptg_path, "ptg")]:
            model = load_model(path)
            analyzer = ModelAnalyzer(model, task=task, device=device, label=variant)
            d_model = analyzer.model.cfg.d_model
            n_heads = analyzer.model.cfg.n_heads
            d_head = d_model // n_heads
            accs = extract_and_probe(analyzer, p, d_model, n_heads, d_head)
            # Store accs in correct column order (interleaved by location)
            if variant == "PT":
                pt_accs = accs
            elif variant == "POST":
                post_accs = accs
            else:
                ptg_accs = accs
            # Free memory
            del model, analyzer

        # Interleave: for each location, write pt, post, ptg
        for j in range(len(LOCATIONS)):
            row.extend([f"{pt_accs[j]:.4f}", f"{post_accs[j]:.4f}", f"{ptg_accs[j]:.4f}"])

        # Append row
        with open(out_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        print(f"  PT:   heads={[f'{a:.2f}' for a in pt_accs[:4]]}  post_attn={pt_accs[4]:.2f}  post_mlp={pt_accs[5]:.2f}")
        print(f"  POST: heads={[f'{a:.2f}' for a in post_accs[:4]]}  post_attn={post_accs[4]:.2f}  post_mlp={post_accs[5]:.2f}")
        print(f"  PTG:  heads={[f'{a:.2f}' for a in ptg_accs[:4]]}  post_attn={ptg_accs[4]:.2f}  post_mlp={ptg_accs[5]:.2f}")

    print(f"\n\nSweep CSV saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train parity probes at 6 network locations")
    parser.add_argument("--sweep", action="store_true", help="Run across all 126 sweep configs")
    parser.add_argument("--wd", type=float)
    parser.add_argument("--bs", type=int)
    parser.add_argument("--ms", type=int)
    parser.add_argument("--ds", type=int)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.sweep:
        probe_sweep(args.device)
    else:
        if not all([args.wd, args.bs, args.ms, args.ds]):
            parser.error("Provide --wd --bs --ms --ds, or use --sweep")
        probe_single(args)


if __name__ == "__main__":
    main()
