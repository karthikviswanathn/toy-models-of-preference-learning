#!/usr/bin/env python3
"""Generate a PDF report with all analysis figures and tables for a given hyperparameter config.

Usage:
    sbatch run_job.sh analysis/checks/generate_report.py --wd 0.15 --bs 1024 --ms 1234 --ss 42 --sh 44
    python analysis/checks/generate_report.py --wd 0.15 --bs 1024 --ms 1234 --ss 42 --sh 44 --device cuda
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.analyzer import ModelAnalyzer, load_model
from trainer.tokenizer import ModularAdditionTokenizer
from trainer.utils import (
    generate_parity_gated_data, split_data, eval_model,
    get_fourier_basis, get_fourier_basis_names,
)


def find_model(variant_dir, pattern):
    """Find the model.pt matching a glob pattern under variant_dir."""
    matches = sorted(PROJECT_ROOT.glob(f"outputs/runs/{variant_dir}/{pattern}/model.pt"))
    if not matches:
        raise FileNotFoundError(f"No model found for {variant_dir}/{pattern}")
    return matches[0]


def eval_table_page(pdf, a_pt, a_sft, a_ptg, device):
    """Page 1: Test performance table."""
    tokenizer = ModularAdditionTokenizer(113)
    seed = 42
    train_frac = 0.3

    eval_pt = a_pt.evaluate(train_frac=train_frac, seed=seed)
    eval_sft = a_sft.evaluate(train_frac=train_frac, seed=seed)
    eval_ptg = a_ptg.evaluate(train_frac=train_frac, seed=seed)

    # PT odd accuracy on parity-gated data
    inputs, labels, loss_mask, is_even = generate_parity_gated_data(tokenizer, device=device)
    rng = np.random.default_rng(seed)
    _, _, _, _, te_x, te_y, _, te_e = split_data(
        inputs, labels, loss_mask, is_even, train_frac, rng
    )
    a_pt.model.eval()
    with torch.no_grad():
        preds = a_pt.model(te_x)[:, 3].argmax(-1)
        pt_even_acc = (preds[te_e] == te_y[te_e]).float().mean().item()
        pt_odd_acc = (preds[~te_e] == te_y[~te_e]).float().mean().item()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    headers = ["", "PT", "POST", "PT-G"]
    rows = [
        ["Test loss",
         f"{eval_pt['test_loss']:.2e}",
         f"{eval_sft['test_loss']:.2e}",
         f"{eval_ptg['test_loss']:.2e}"],
        ["Test acc",
         f"{eval_pt['test_acc']:.4f}",
         f"{eval_sft['test_acc']:.4f}",
         f"{eval_ptg['test_acc']:.4f}"],
        ["Even acc",
         f"{pt_even_acc:.4f}",
         f"{eval_sft.get('test_acc_even', 'N/A'):.4f}",
         f"{eval_ptg.get('test_acc_even', 'N/A'):.4f}"],
        ["Odd acc",
         f"{pt_odd_acc:.4f}",
         f"{eval_sft.get('test_acc_odd', 'N/A'):.4f}",
         f"{eval_ptg.get('test_acc_odd', 'N/A'):.4f}"],
    ]
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
    ax.set_title("Test Performance (Parity-Gated Evaluation)", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def pca_stages_page(pdf, a_pt, a_sft, a_ptg):
    """Page 2: PCA stages parity — PT, POST, PT-G with row labels."""
    fig = ModelAnalyzer.compare_pca([a_pt, a_sft, a_ptg], color_by="parity", n_components=2)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _fourier_spectrum_page(pdf, a_pt, a_sft, a_ptg, title_suffix, get_power_fn):
    """Fourier spectrum: 3 separate panels (PT, POST, PT-G), no shared y-axis."""
    p = 113
    names = get_fourier_basis_names(p)
    step = max(1, len(names) // 20)
    ticks = list(range(0, len(names), step))
    tick_labels = [names[t] for t in ticks]

    pow_pt = get_power_fn(a_pt)
    pow_sft = get_power_fn(a_sft)
    pow_ptg = get_power_fn(a_ptg)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    for ax, power, label, color in [
        (axes[0], pow_pt, "PT", "tab:blue"),
        (axes[1], pow_sft, "POST", "tab:red"),
        (axes[2], pow_ptg, "PT-G", "tab:green"),
    ]:
        markerline, stemlines, baseline = ax.stem(
            range(len(power)), power, markerfmt="o", basefmt="k-"
        )
        markerline.set_markersize(3)
        markerline.set_color(color)
        stemlines.set_color(color)
        stemlines.set_alpha(0.5)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
        ax.set_ylabel("Power", fontsize=11)
        ax.set_title(label, fontsize=13, fontweight="bold")

    fig.suptitle(title_suffix, fontsize=14, fontweight="bold")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def fourier_embedding_page(pdf, a_pt, a_sft, a_ptg):
    """Page 3: Fourier spectrum of W_E. PT+POST overplotted, PT-G separate."""
    p = 113
    names = get_fourier_basis_names(p)
    step = max(1, len(names) // 20)
    ticks = list(range(0, len(names), step))
    tick_labels = [names[t] for t in ticks]

    def get_power(analyzer):
        data = analyzer.fourier_embedding()
        power = data["power_per_freq"]
        return power.cpu().numpy() if isinstance(power, torch.Tensor) else power

    pow_pt = get_power(a_pt)
    pow_sft = get_power(a_sft)
    pow_ptg = get_power(a_ptg)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Left: PT + POST overplotted
    ax = axes[0]
    m1, s1, _ = ax.stem(range(len(pow_pt)), pow_pt, markerfmt="o", basefmt="k-", label="PT")
    m1.set_markersize(3); m1.set_color("tab:blue"); s1.set_color("tab:blue"); s1.set_alpha(0.5)
    m2, s2, _ = ax.stem(range(len(pow_sft)), pow_sft, markerfmt="o", basefmt="k-", label="POST")
    m2.set_markersize(3); m2.set_color("tab:red"); s2.set_color("tab:red"); s2.set_alpha(0.5)
    ax.set_xticks(ticks); ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_ylabel("Power", fontsize=11); ax.set_title("PT + POST", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    # Right: PT-G
    ax = axes[1]
    m3, s3, _ = ax.stem(range(len(pow_ptg)), pow_ptg, markerfmt="o", basefmt="k-")
    m3.set_markersize(3); m3.set_color("tab:green"); s3.set_color("tab:green"); s3.set_alpha(0.5)
    ax.set_xticks(ticks); ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_ylabel("Power", fontsize=11); ax.set_title("PT-G", fontsize=13, fontweight="bold")

    fig.suptitle("Fourier Power Spectrum of $W_E$ (Number Tokens)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def neuron_logit_fourier_page(pdf, a_pt, a_sft, a_ptg):
    """Page 4: Fourier spectrum of W_logit."""
    p = 113
    fb = get_fourier_basis(p, "cpu")

    def get_power(analyzer):
        model = analyzer.model
        W_out = model.blocks[0].mlp.W_out.detach().cpu()
        W_U = model.unembed.W_U.detach().cpu()[:, :p]
        W_logit = W_out @ W_U
        coeffs = W_logit @ fb.T
        return (coeffs ** 2).sum(dim=0).numpy()

    _fourier_spectrum_page(pdf, a_pt, a_sft, a_ptg,
                           title_suffix="Fourier Power Spectrum of $W_{\\mathrm{logit}} = W_{\\mathrm{out}} W_U$",
                           get_power_fn=get_power)


# --- TODO stubs for future pages ---
# def pca_post_page(pdf, a_sft): ...
# def pca_ptg_page(pdf, a_ptg): ...
# def per_head_pca_page(pdf, a_ptg): ...


def main():
    parser = argparse.ArgumentParser(description="Generate analysis report PDF")
    parser.add_argument("--wd", type=float, required=True, help="Weight decay")
    parser.add_argument("--bs", type=int, required=True, help="Batch size (-1 for full)")
    parser.add_argument("--ms", type=int, required=True, help="Model seed")
    parser.add_argument("--ss", type=int, required=True, help="Split seed")
    parser.add_argument("--sh", type=int, required=True, help="Shuffle seed")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None, help="Output PDF path")
    args = parser.parse_args()

    suffix = f"wd{args.wd}_bs{args.bs}_ms{args.ms}_ss{args.ss}_sh{args.sh}"

    print(f"Config: {suffix}")
    print(f"Device: {args.device}")

    # Find models
    pt_path = find_model("pt", f"pt_{suffix}_*")
    ptg_path = find_model("pt-g", f"ptg_{suffix}_*")
    sft_path = find_model("sft", f"sft_{suffix}_*")
    print(f"PT:   {pt_path.parent.name}")
    print(f"PT-G: {ptg_path.parent.name}")
    print(f"SFT:  {sft_path.parent.name}")

    # Load models
    print("\nLoading models...")
    pt_model = load_model(pt_path)
    ptg_model = load_model(ptg_path)
    sft_model = load_model(sft_path)

    # Create analyzers
    print("Creating analyzers...")
    a_pt = ModelAnalyzer(pt_model, task="pt", device=args.device, label="PT")
    a_sft = ModelAnalyzer(sft_model, task="ptg", device=args.device, label="POST")
    a_ptg = ModelAnalyzer(ptg_model, task="ptg", device=args.device, label="PT-G")

    # Generate PDF
    out_path = args.output or str(PROJECT_ROOT / "writeup" / "figs" / f"report_{suffix}.pdf")
    print(f"\nGenerating report: {out_path}")

    with PdfPages(out_path) as pdf:
        print("  [1/4] Test performance table...")
        eval_table_page(pdf, a_pt, a_sft, a_ptg, args.device)

        print("  [2/4] PCA stages parity (PT, POST, PT-G)...")
        pca_stages_page(pdf, a_pt, a_sft, a_ptg)

        print("  [3/4] Fourier embedding spectrum...")
        fourier_embedding_page(pdf, a_pt, a_sft, a_ptg)

        print("  [4/4] Neuron-logit Fourier spectrum...")
        neuron_logit_fourier_page(pdf, a_pt, a_sft, a_ptg)

    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    main()
