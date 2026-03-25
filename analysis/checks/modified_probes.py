#!/usr/bin/env python3
"""Modified parity probes: augment activations with squared PC features.

Test whether parity becomes decodable when we add nonlinear features
(pc1^2, pc2^2, ..., pc_k^2) to the activation vector.

Usage (single canonical config):
    sbatch run_job.sh analysis/checks/modified_probes.py --device cuda

Usage (sweep over all 126 configs):
    sbatch run_job.sh analysis/checks/modified_probes.py --sweep --device cuda
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.analyzer import ModelAnalyzer, load_model

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


def augment_with_squared_pcs(X, n_pcs=10):
    """PCA-project X, then append pc_1^2, ..., pc_k^2 to original features."""
    pca = PCA(n_components=n_pcs)
    projections = pca.fit_transform(X)  # (N, n_pcs)
    squared = projections ** 2           # (N, n_pcs)
    return np.hstack([X, squared])       # (N, d + n_pcs)


def extract_activations(analyzer):
    """Extract activations at 6 locations, return dict."""
    p = analyzer.p
    eq_pos = 3
    d_model = analyzer.model.cfg.d_model
    n_heads = analyzer.model.cfg.n_heads
    d_head = d_model // n_heads

    z = analyzer.cache["blocks.0.attn.hook_z"][:, eq_pos].cpu().numpy()
    assert z.shape == (p * p, n_heads, d_head)
    resid_mid = analyzer.cache["blocks.0.hook_resid_mid"][:, eq_pos].cpu().numpy()
    assert resid_mid.shape == (p * p, d_model)
    resid_post = analyzer.cache["blocks.0.hook_resid_post"][:, eq_pos].cpu().numpy()
    assert resid_post.shape == (p * p, d_model)

    acts = {}
    for h in range(n_heads):
        acts[f"Head {h}"] = z[:, h, :]
    acts["Post-Attn"] = resid_mid
    acts["Post-MLP"] = resid_post
    return acts


def probe_model(analyzer, n_pcs):
    """Run linear and augmented probes at all 6 locations. Returns (linear_accs, augmented_accs)."""
    acts = extract_activations(analyzer)
    parity = analyzer.parity_labels
    linear_accs = []
    aug_accs = []
    for loc in LOCATIONS:
        X = acts[loc]
        linear_accs.append(train_parity_probe(X, parity))
        X_aug = augment_with_squared_pcs(X, n_pcs=n_pcs)
        aug_accs.append(train_parity_probe(X_aug, parity))
    return linear_accs, aug_accs


def run_single(args):
    """Run on canonical config, print table."""
    suffix = "wd0.15_bs1024_ms1234_ds42"
    print(f"Config: {suffix}")
    print(f"Device: {args.device}")
    print(f"Squared PCs: {args.n_pcs}")

    results = {}
    for variant, vdir, task in [("PT", "pt", "pt"), ("POST", "sft", "ptg"), ("PT-G", "pt-g", "ptg")]:
        path = find_model(vdir, f"{'ptg' if vdir == 'pt-g' else vdir.replace('-','') if vdir != 'pt' else 'pt'}_{suffix}_*")
        print(f"\n{variant}: {path.parent.name}")
        analyzer = ModelAnalyzer(load_model(path), task=task, device=args.device, label=variant)
        linear, aug = probe_model(analyzer, args.n_pcs)
        results[variant] = (linear, aug)
        del analyzer

    print(f"\n{'Location':10s}  {'PT':>8s} {'PT+PC²':>8s}  {'POST':>8s} {'POST+PC²':>9s}  {'PT-G':>8s} {'PTG+PC²':>8s}")
    print("-" * 75)
    for i, loc in enumerate(LOCATIONS):
        pt_l, pt_a = results["PT"][0][i], results["PT"][1][i]
        po_l, po_a = results["POST"][0][i], results["POST"][1][i]
        pg_l, pg_a = results["PT-G"][0][i], results["PT-G"][1][i]
        print(f"{loc:10s}  {pt_l:8.4f} {pt_a:8.4f}  {po_l:8.4f} {po_a:9.4f}  {pg_l:8.4f} {pg_a:8.4f}")


def run_sweep(args):
    """Run across all 126 configs, save CSV."""
    import pandas as pd
    summary_path = PROJECT_ROOT / "outputs" / "runs-p106" / "pt" / "summary.csv"
    df = pd.read_csv(summary_path)
    configs = df[KEYS].values
    n_configs = len(configs)
    print(f"Found {n_configs} configs")
    print(f"Device: {args.device}, Squared PCs: {args.n_pcs}")

    # Column names: for each location, linear and augmented for each variant
    variants = ["pt", "post", "ptg"]
    col_names = []
    for loc in LOCATIONS:
        loc_tag = loc.lower().replace(" ", "_").replace("-", "_")
        for var in variants:
            col_names.append(f"{loc_tag}_{var}")
            col_names.append(f"{loc_tag}_{var}_aug")

    out_path = PROJECT_ROOT / "outputs" / "runs-p106" / "modified_probes_sweep.csv"
    with open(out_path, "w", newline="") as f:
        csv.writer(f).writerow(KEYS + col_names)

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
        all_results = {}
        for variant, path, task in [("pt", pt_path, "pt"), ("post", sft_path, "ptg"), ("ptg", ptg_path, "ptg")]:
            analyzer = ModelAnalyzer(load_model(path), task=task, device=args.device, label=variant)
            linear, aug = probe_model(analyzer, args.n_pcs)
            all_results[variant] = (linear, aug)
            del analyzer

        # Interleave columns: for each location, pt_linear, pt_aug, post_linear, post_aug, ptg_linear, ptg_aug
        for j in range(len(LOCATIONS)):
            for var in variants:
                row.append(f"{all_results[var][0][j]:.4f}")
                row.append(f"{all_results[var][1][j]:.4f}")

        with open(out_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        # Print summary for this config
        for var in variants:
            tag = {"pt": "PT", "post": "POST", "ptg": "PTG"}[var]
            l, a = all_results[var]
            print(f"  {tag:5s} post_attn: {l[4]:.2f}->{a[4]:.2f}  post_mlp: {l[5]:.2f}->{a[5]:.2f}")

    print(f"\n\nSweep CSV saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Modified parity probes with squared PC features")
    parser.add_argument("--sweep", action="store_true", help="Run across all 126 sweep configs")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_pcs", type=int, default=10, help="Number of squared PCs to append")
    args = parser.parse_args()

    if args.sweep:
        run_sweep(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
