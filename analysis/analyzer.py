"""Consolidated mechanistic interpretability analysis for modular addition transformers.

Supports two task modes:
  - "pt":  vanilla pretrain (a + b = c for all pairs)
  - "ptg": preference-gated pretrain (preferred results predict c, unpreferred predict U)

Usage (notebook):
    from analysis.analyzer import ModelAnalyzer
    a = ModelAnalyzer(model, task="pt")
    a.summary()
    a.plot_pca(color_by="result", save_path="out.png")
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from trainer.config import ModelConfig, DataConfig
from trainer.utils import (
    generate_preference_gated_data,
    split_data,
    eval_model,
    get_fourier_basis,
    get_fourier_basis_names,
    fourier_transform_1d,
    fourier_transform_2d,
)
from trainer.tokenizer import ModularAdditionTokenizer
from trainer.data import generate_all_data
from transformer_lens import HookedTransformer


# ---------------------------------------------------------------------------
# Shared helpers (canonical location — other scripts import from here)
# ---------------------------------------------------------------------------

STAGES = [
    ("Post-Attention", "blocks.{L}.hook_resid_mid"),
    ("Post-MLP",       "blocks.{L}.hook_resid_post"),
]


def load_model(path: str | Path) -> HookedTransformer:
    """Load model from a .pt file or a run directory."""
    path = Path(path)
    if path.is_dir():
        model = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
    else:
        model = torch.load(path, map_location="cpu", weights_only=False)
    model.eval()
    return model


def generate_preference_gated_inputs(tokenizer: ModularAdditionTokenizer, device: str = "cpu", unsafe_threshold: int = 57):
    """Generate all p^2 preference-gated inputs, returning (inputs, result_labels)."""
    inputs, labels, _, is_preferred = generate_preference_gated_data(tokenizer, device=device, unsafe_threshold=unsafe_threshold)
    p = tokenizer.p
    result_labels = np.array([(a + b) % p for a in range(p) for b in range(p)])
    return inputs, result_labels


def extract_stage_activations(
    model: HookedTransformer, all_inputs: torch.Tensor, device: str = "cuda"
) -> dict[str, np.ndarray]:
    """Extract activations at the '=' position for each network stage."""
    model.eval()
    model.to(device)
    eq_pos = 3

    with torch.no_grad():
        _, cache = model.run_with_cache(all_inputs.to(device))

    last_layer = model.cfg.n_layers - 1
    activations = {}
    for stage_name, hook_template in STAGES:
        hook_name = hook_template.format(L=last_layer)
        activations[stage_name] = cache[hook_name][:, eq_pos].cpu().numpy()

    return activations


# ---------------------------------------------------------------------------


@dataclass
class PCAResult:
    """Result of a PCA computation on residual stream activations."""
    projections: np.ndarray            # (p*p, n_components)
    explained_variance_ratio: np.ndarray  # (n_components,)
    pca_model: PCA
    stage: str


class ModelAnalyzer:
    """Mechanistic interpretability analysis for a modular addition model.

    Args:
        model: A HookedTransformer instance.
        task: "pt" (vanilla pretrain) or "ptg" (preference-gated pretrain).
        device: Device for computation.
        p: Prime modulus (defaults to ModelConfig().p).
        label: Human-readable label for plots.
    """

    VALID_TASKS = ("pt", "ptg")
    EQ_POS = 3
    def __init__(
        self,
        model: HookedTransformer,
        *,
        task: str = "ptg",
        device: str = "cuda",
        p: int | None = None,
        label: str | None = None,
        unsafe_threshold: int | None = None,
    ):
        if task not in self.VALID_TASKS:
            raise ValueError(f"task must be one of {self.VALID_TASKS}, got {task!r}")

        self.model: HookedTransformer = model
        self.task = task
        if label is None:
            label = f"{model.cfg.n_layers}L{model.cfg.n_heads}H"

        self.label = label
        self.device = device
        self.p = p or ModelConfig().p
        self.unsafe_threshold = unsafe_threshold or DataConfig().unsafe_threshold

        # Tokenizer and data
        self.tokenizer = ModularAdditionTokenizer(self.p)

        if task == "ptg":
            self.all_inputs, self.result_labels = generate_preference_gated_inputs(
                self.tokenizer, "cpu", unsafe_threshold=self.unsafe_threshold
            )
        else:  # task == "pt"
            inputs, labels = generate_all_data(self.tokenizer, device="cpu")
            self.all_inputs = inputs
            self.result_labels = labels.numpy()

        self.a_labels = np.array([a for a in range(self.p) for _ in range(self.p)])
        self.b_labels = np.array([b for _ in range(self.p) for b in range(self.p)])
        self.preference_labels = self.result_labels < self.unsafe_threshold

        # Run forward pass and cache activations
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            self.logits, self.cache = self.model.run_with_cache(
                self.all_inputs.to(self.device)
            )
            self.eq_logits = self.logits[:, self.EQ_POS].reshape(self.p, self.p, -1)
        self.stage_activations = extract_stage_activations(
            self.model, self.all_inputs, self.device
        )

    @property
    def W_E(self) -> torch.Tensor:
        """Token embedding matrix. Shape (vocab, d_model)."""
        return self.model.embed.W_E.detach()

    @property
    def W_U(self) -> torch.Tensor:
        """Unembedding matrix. Shape (d_model, vocab)."""
        return self.model.unembed.W_U.detach()

    @property
    def W_pos(self) -> torch.Tensor:
        """Positional embedding matrix. Shape (n_ctx, d_model)."""
        return self.model.pos_embed.W_pos.detach()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a summary string of the model."""
        cfg = self.model.cfg
        s = (
            f"ModelAnalyzer: {self.label}\n"
            f"  Architecture: {cfg.n_layers}L{cfg.n_heads}H, d_model={cfg.d_model}, d_mlp={cfg.d_mlp}\n"
            f"  p={self.p}, vocab={cfg.d_vocab}, n_ctx={cfg.n_ctx}\n"
            f"  Device: {self.device}\n"
            f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )
        print(s)
        return s

    # ------------------------------------------------------------------
    # Computation methods
    # ------------------------------------------------------------------

    def pca(
        self,
        n_components: int = 2,
        stage: str | None = None,
    ) -> dict[str, PCAResult]:
        """Run PCA on residual stream activations at the '=' token position.

        Args:
            n_components: Number of principal components.
            stage: Specific stage name, or None for all stages.

        Returns:
            Dict mapping stage_name -> PCAResult.
        """
        acts = self.stage_activations
        stages = {stage: acts[stage]} if stage else acts
        results = {}
        for name, data in stages.items():
            pca_model = PCA(n_components=n_components)
            proj = pca_model.fit_transform(data)
            results[name] = PCAResult(
                projections=proj,
                explained_variance_ratio=pca_model.explained_variance_ratio_,
                pca_model=pca_model,
                stage=name,
            )
        return results

    def evaluate(
        self,
        train_frac: float = 0.3,
        seed: int = 42,
    ) -> dict:
        """Compute loss and accuracy on train/test split.

        Returns:
            Dict with keys: train_loss, test_loss, train_acc, test_acc.
            For task="ptg", also includes train_acc_preferred, test_acc_preferred,
            train_acc_unpreferred, test_acc_unpreferred.
        """
        if self.task == "ptg":
            inputs, labels, loss_mask, is_preferred = generate_preference_gated_data(
                self.tokenizer, device=self.device, unsafe_threshold=self.unsafe_threshold
            )
            rng = np.random.default_rng(seed)
            tr_x, tr_y, tr_m, tr_e, te_x, te_y, te_m, te_e = split_data(
                inputs, labels, loss_mask, is_preferred, train_frac, rng
            )
            self.model.eval()
            tr_loss, tr_acc, tr_acc_e, tr_acc_o = eval_model(self.model, tr_x, tr_y, tr_m, tr_e)
            te_loss, te_acc, te_acc_e, te_acc_o = eval_model(self.model, te_x, te_y, te_m, te_e)
            return dict(
                train_loss=tr_loss, test_loss=te_loss,
                train_acc=tr_acc, test_acc=te_acc,
                train_acc_preferred=tr_acc_e, test_acc_preferred=te_acc_e,
                train_acc_unpreferred=tr_acc_o, test_acc_unpreferred=te_acc_o,
            )
        else:  # task == "pt"
            from trainer.data import train_test_split
            inputs, labels = generate_all_data(self.tokenizer, device=self.device)
            rng = np.random.default_rng(seed)
            tr_x, tr_y, te_x, te_y = train_test_split(inputs, labels, train_frac, rng)
            self.model.eval()
            vocab_size = self.model.cfg.d_vocab
            loss_fn = nn.CrossEntropyLoss()
            with torch.no_grad():
                tr_logits = self.model(tr_x)[:, 3]  # position 3 predicts result
                tr_loss = loss_fn(tr_logits, tr_y).item()
                tr_acc = (tr_logits.argmax(-1) == tr_y).float().mean().item()
                te_logits = self.model(te_x)[:, 3]
                te_loss = loss_fn(te_logits, te_y).item()
                te_acc = (te_logits.argmax(-1) == te_y).float().mean().item()
            return dict(
                train_loss=tr_loss, test_loss=te_loss,
                train_acc=tr_acc, test_acc=te_acc,
            )

    def mean_logits(self, position: int = 3) -> np.ndarray:
        """Average logits across all (a,b) inputs at a given position.

        Args:
            position: Sequence position (default 3 = '=' token).

        Returns:
            (vocab,) numpy array of mean logits.
        """
        return self.logits[:, position].mean(dim=0).cpu().numpy()

    def fourier_embedding(self) -> dict:
        """1D Fourier analysis of token embedding matrix W_E (number tokens only).

        Returns:
            Dict with: fourier_coeffs (p, d_model), power_per_freq (p,),
                        fft_l2 (p,), fourier_names (list of str).
        """
        W_E_nums = self.W_E[:self.p].cpu()  # (p, d_model)
        fb = get_fourier_basis(self.p, W_E_nums.device)
        coeffs = fb @ W_E_nums  # (p, d_model)
        power = (coeffs ** 2).sum(dim=1)  # (p,)
        fft_raw = torch.fft.fft(W_E_nums, dim=0)
        fft_l2 = fft_raw.abs().pow(2).sum(dim=1).sqrt().numpy()  # (p,)
        return dict(
            fourier_coeffs=coeffs,
            power_per_freq=power,
            fft_l2=fft_l2,
            fourier_names=get_fourier_basis_names(self.p),
        )

    def fourier_unembed(self) -> dict:
        """1D Fourier analysis of unembedding matrix W_U (number logits only).

        Returns:
            Dict with: fourier_coeffs (d_model, p), power_per_freq (p,),
                        fourier_names (list of str).
        """
        W_U_nums = self.W_U[:, :self.p].cpu()  # (d_model, p)
        fb = get_fourier_basis(self.p, W_U_nums.device)
        coeffs = W_U_nums @ fb.T  # (d_model, p)
        power = (coeffs ** 2).sum(dim=0)  # (p,)
        return dict(
            fourier_coeffs=coeffs,
            power_per_freq=power,
            fourier_names=get_fourier_basis_names(self.p),
        )

    def fourier_mlp_neurons(
        self,
        layer: int = 0,
        position: int = 3,
        post_relu: bool = True,
    ) -> dict:
        """2D Fourier analysis of MLP neuron activations over (a, b) pairs.

        Args:
            layer: Which transformer layer's MLP.
            position: Sequence position (default 3 = '=').
            post_relu: If True, use post-ReLU (hook_post); else pre-ReLU (hook_pre).

        Returns:
            Dict with: activations (d_mlp, p, p), fourier_coeffs (d_mlp, p, p),
                        power (d_mlp, p, p), total_power (d_mlp,).
        """
        hook = f"blocks.{layer}.mlp.hook_{'post' if post_relu else 'pre'}"
        raw = self.cache[hook][:, position].cpu()  # (p*p, d_mlp)
        d_mlp = raw.shape[1]
        acts_2d = raw.T.reshape(d_mlp, self.p, self.p)  # (d_mlp, p, p)

        fb = get_fourier_basis(self.p, acts_2d.device)
        coeffs = fourier_transform_2d(acts_2d, fourier_basis=fb)
        power = coeffs ** 2
        total_power = power.reshape(d_mlp, -1).sum(dim=1)

        return dict(
            activations=acts_2d,
            fourier_coeffs=coeffs,
            power=power,
            total_power=total_power,
        )

    def fourier_logits(self, position: int = 3) -> dict:
        """2D Fourier analysis of output logits over (a, b) pairs.

        Args:
            position: Sequence position (default 3 = '=').

        Returns:
            Dict with: logits_2d (p, p, vocab), fourier_coeffs (vocab, p, p),
                        power_per_output (vocab, p, p), fourier_names (list of str).
        """
        raw = self.logits[:, position].cpu()  # (p*p, vocab)
        vocab = raw.shape[1]
        logits_2d = raw.reshape(self.p, self.p, vocab)
        # Transpose to (vocab, p, p) for Fourier
        logits_vpp = logits_2d.permute(2, 0, 1)

        fb = get_fourier_basis(self.p, logits_vpp.device)
        coeffs = fourier_transform_2d(logits_vpp, fourier_basis=fb)
        power = coeffs ** 2

        return dict(
            logits_2d=logits_2d,
            fourier_coeffs=coeffs,
            power_per_output=power,
            fourier_names=get_fourier_basis_names(self.p),
        )

    # ------------------------------------------------------------------
    # Plotting methods
    # ------------------------------------------------------------------

    def plot_pca(
        self,
        pca_results: dict[str, PCAResult] | None = None,
        *,
        color_by: str = "preference",
        pc_pairs: list[tuple[int, int]] | None = None,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Plot PCA projections colored by the specified scheme.

        Args:
            pca_results: Pre-computed PCA results, or None to auto-compute.
            color_by: One of "preference", "result", "a", "b".
            pc_pairs: List of (i, j) pairs to plot. None = just PC1 vs PC2.
            save_path: If given, save figure to this path.

        Returns:
            matplotlib Figure.
        """
        if pc_pairs is None:
            pc_pairs = [(0, 1)]
        max_pc = max(max(i, j) for i, j in pc_pairs) + 1

        if pca_results is None:
            pca_results = self.pca(n_components=max_pc)

        stage_labels = list(pca_results.keys())
        nrows = len(stage_labels)
        ncols = len(pc_pairs)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows + 0.8), squeeze=False)

        is_binary = color_by == "preference"

        for row_idx, stage_name in enumerate(stage_labels):
            pr = pca_results[stage_name]
            var = pr.explained_variance_ratio * 100

            for col_idx, (i, j) in enumerate(pc_pairs):
                ax = axes[row_idx, col_idx]

                if is_binary:
                    preferred = self.preference_labels
                    ax.scatter(pr.projections[preferred, i], pr.projections[preferred, j],
                               c="tab:blue", s=2, alpha=0.4, rasterized=True)
                    ax.scatter(pr.projections[~preferred, i], pr.projections[~preferred, j],
                               c="tab:red", s=2, alpha=0.4, rasterized=True)
                else:
                    colors = {"result": self.result_labels, "a": self.a_labels, "b": self.b_labels}[color_by]
                    sc = ax.scatter(pr.projections[:, i], pr.projections[:, j],
                                    c=colors, cmap="hsv", s=2, alpha=0.4, rasterized=True)

                ax.tick_params(labelsize=9)
                ax.locator_params(axis="both", nbins=5)
                if row_idx == 0:
                    ax.set_title(f"PC{i+1} vs PC{j+1}", fontsize=12, fontweight="bold", pad=8)
                ax.set_xlabel(f"PC{i+1} ({var[i]:.1f}%)", fontsize=11)
                ax.set_ylabel(f"PC{j+1} ({var[j]:.1f}%)", fontsize=11)

        # Row labels
        for row_idx, stage_name in enumerate(stage_labels):
            axes[row_idx, 0].annotate(
                stage_name, xy=(0, 0.5), xytext=(-0.38, 0.5),
                xycoords="axes fraction", textcoords="axes fraction",
                fontsize=12, fontweight="bold", rotation=90, ha="center", va="center",
            )

        # Legend / colorbar
        if is_binary:
            handles = [
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markersize=6, label="preferred"),
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:red", markersize=6, label="unpreferred"),
            ]
            fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.01))
        else:
            cbar = fig.colorbar(sc, ax=axes, orientation="horizontal", fraction=0.03, pad=0.08)
            cbar.set_label({"result": "(a + b) mod p", "a": "a", "b": "b"}[color_by], fontsize=11)

        fig.suptitle(f'PCA of "=" token (colored by {color_by}) — {self.label}', fontsize=14, fontweight="bold")
        fig.subplots_adjust(hspace=0.45, wspace=0.4, left=0.08)

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved: {save_path}")

        return fig

    def plot_fourier_spectrum(
        self,
        fourier_data: dict | None = None,
        *,
        kind: str = "embedding",
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Plot 1D Fourier power spectrum.

        Args:
            fourier_data: Pre-computed Fourier result, or None to auto-compute.
            kind: "embedding" or "unembed".
            save_path: If given, save figure.

        Returns:
            matplotlib Figure.
        """
        if fourier_data is None:
            fourier_data = self.fourier_embedding() if kind == "embedding" else self.fourier_unembed()

        power = fourier_data["power_per_freq"]
        if isinstance(power, torch.Tensor):
            power = power.cpu().numpy()
        names = fourier_data["fourier_names"]

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.stem(range(len(power)), power, markerfmt="o", basefmt="k-")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_ylabel("Power", fontsize=11)
        ax.set_title(f"Fourier power spectrum of W_{'E' if kind == 'embedding' else 'U'} — {self.label}",
                      fontsize=13, fontweight="bold")
        fig.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved: {save_path}")

        return fig

    def plot_fourier_2d(
        self,
        power_2d: np.ndarray | torch.Tensor | None = None,
        *,
        title: str = "",
        fourier_names: list[str] | None = None,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Plot a 2D Fourier power spectrum as a heatmap.

        Args:
            power_2d: (p, p) array of Fourier power. If None, uses fourier_logits sum.
            title: Plot title.
            fourier_names: Axis tick labels.
            save_path: If given, save figure.

        Returns:
            matplotlib Figure.
        """
        if power_2d is None:
            fd = self.fourier_logits()
            power_2d = fd["power_per_output"][:self.p].sum(dim=0)
            title = title or f"2D Fourier power of logits — {self.label}"
            fourier_names = fourier_names or fd["fourier_names"]

        if isinstance(power_2d, torch.Tensor):
            power_2d = power_2d.cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(power_2d, cmap="hot", aspect="auto")
        fig.colorbar(im, ax=ax, label="Power")
        ax.set_title(title or f"2D Fourier power — {self.label}", fontsize=13, fontweight="bold")
        if fourier_names:
            step = max(1, len(fourier_names) // 20)
            ticks = list(range(0, len(fourier_names), step))
            ax.set_xticks(ticks)
            ax.set_xticklabels([fourier_names[t] for t in ticks], rotation=90, fontsize=7)
            ax.set_yticks(ticks)
            ax.set_yticklabels([fourier_names[t] for t in ticks], fontsize=7)
        fig.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved: {save_path}")

        return fig

    # ------------------------------------------------------------------
    # Static comparison methods
    # ------------------------------------------------------------------

    @staticmethod
    def compare_pca(
        analyzers: list[ModelAnalyzer],
        *,
        color_by: str = "preference",
        n_components: int = 2,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Plot a comparison grid: rows=models, cols=stages.

        Args:
            analyzers: List of ModelAnalyzer instances.
            color_by: Coloring scheme.
            n_components: Number of PCA components (uses first 2 for plotting).
            save_path: If given, save figure.

        Returns:
            matplotlib Figure.
        """
        stage_names = [s[0] for s in STAGES]
        nrows = len(analyzers)
        ncols = len(stage_names)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows + 0.8), squeeze=False)

        is_binary = color_by == "preference"

        for row_idx, a in enumerate(analyzers):
            pca_results = a.pca(n_components=n_components)
            for col_idx, stage_name in enumerate(stage_names):
                if stage_name not in pca_results:
                    continue
                pr = pca_results[stage_name]
                var = pr.explained_variance_ratio * 100
                ax = axes[row_idx, col_idx]

                if is_binary:
                    preferred = a.preference_labels
                    ax.scatter(pr.projections[preferred, 0], pr.projections[preferred, 1],
                               c="tab:blue", s=2, alpha=0.4, rasterized=True)
                    ax.scatter(pr.projections[~preferred, 0], pr.projections[~preferred, 1],
                               c="tab:red", s=2, alpha=0.4, rasterized=True)
                else:
                    colors = {"result": a.result_labels, "a": a.a_labels, "b": a.b_labels}[color_by]
                    sc = ax.scatter(pr.projections[:, 0], pr.projections[:, 1],
                                    c=colors, cmap="hsv", s=2, alpha=0.4, rasterized=True)

                ax.tick_params(labelsize=9)
                ax.locator_params(axis="both", nbins=5)
                if row_idx == 0:
                    ax.set_title(stage_name, fontsize=12, fontweight="bold", pad=8)
                ax.set_xlabel(f"PC1 ({var[0]:.1f}%)", fontsize=11)
                ax.set_ylabel(f"PC2 ({var[1]:.1f}%)", fontsize=11)

            # Row label
            axes[row_idx, 0].annotate(
                a.label, xy=(0, 0.5), xytext=(-0.38, 0.5),
                xycoords="axes fraction", textcoords="axes fraction",
                fontsize=12, fontweight="bold", rotation=90, ha="center", va="center",
            )

        if is_binary:
            handles = [
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markersize=6, label="preferred"),
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:red", markersize=6, label="unpreferred"),
            ]
            fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.01))
        else:
            cbar = fig.colorbar(sc, ax=axes, orientation="horizontal", fraction=0.03, pad=0.08)
            cbar.set_label({"result": "(a + b) mod p", "a": "a", "b": "b"}[color_by], fontsize=11)

        fig.suptitle(f'PCA comparison (colored by {color_by})', fontsize=14, fontweight="bold")
        fig.subplots_adjust(hspace=0.45, wspace=0.4, left=0.1)

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved: {save_path}")

        return fig

    # ------------------------------------------------------------------
    # Nanda et al. verification methods
    # ------------------------------------------------------------------

    def _compute_loss_acc(self, logits_all: torch.Tensor) -> tuple[float, float]:
        """Compute loss and accuracy from full-sequence logits.

        Uses the correct loss computation for the current task mode.
        """
        if self.task == "ptg":
            inputs, labels, loss_mask, _ = generate_preference_gated_data(
                self.tokenizer, device=logits_all.device, unsafe_threshold=self.unsafe_threshold
            )
            loss_fn = nn.CrossEntropyLoss(reduction="none")
            logits = logits_all[:, :-1]
            shifted = inputs[:, 1:]
            pt = loss_fn(logits.reshape(-1, self.model.cfg.d_vocab), shifted.reshape(-1))
            loss = (pt.view(shifted.size()) * loss_mask).sum() / loss_mask.sum()
            preds = logits_all[:, 3].argmax(-1)
            acc = (preds == labels).float().mean()
        else:  # task == "pt"
            inputs, labels = generate_all_data(self.tokenizer, device=logits_all.device)
            loss_fn = nn.CrossEntropyLoss()
            logits_at_eq = logits_all[:, 3]  # position 3 predicts result
            loss = loss_fn(logits_at_eq, labels)
            acc = (logits_at_eq.argmax(-1) == labels).float().mean()
        return loss.item(), acc.item()

    def _logits_from_resid_post(self, resid_post: torch.Tensor) -> torch.Tensor:
        """Apply ln_final (if any) + unembed to a residual stream tensor."""
        x = resid_post
        if self.model.cfg.normalization_type is not None:
            x = self.model.ln_final(x)
        return self.model.unembed(x)

    def test_skip_connection_ablation(self) -> dict:
        """Test Claim 17: The skip connection around the MLP is not important.

        Uses the already-cached activations to reconstruct logits with the
        skip connection zeroed or mean-ablated:
            resid_post = resid_mid + mlp_out      (original)
            resid_post = mlp_out                   (zero ablation)
            resid_post = mean(resid_mid) + mlp_out (mean ablation)

        Reference: Nanda et al. (2023), Appendix A.1

        Returns:
            Dict with original_{loss,acc}, zero_ablated_{loss,acc},
            mean_ablated_{loss,acc}, loss_ratio_zero, loss_ratio_mean.
        """
        L = self.model.cfg.n_layers - 1
        resid_mid = self.cache[f"blocks.{L}.hook_resid_mid"]  # (batch, seq, d_model)
        mlp_out = self.cache[f"blocks.{L}.hook_mlp_out"]      # (batch, seq, d_model)
        resid_post = self.cache[f"blocks.{L}.hook_resid_post"]

        assert torch.allclose(resid_mid + mlp_out, resid_post, atol=1e-5), (
            f"resid_mid + mlp_out != resid_post "
            f"(max diff: {(resid_mid + mlp_out - resid_post).abs().max().item():.2e})"
        )

        with torch.no_grad():
            orig_logits = self._logits_from_resid_post(resid_mid + mlp_out)
            zero_logits = self._logits_from_resid_post(mlp_out)
            mean_logits = self._logits_from_resid_post(
                mlp_out + resid_mid.mean(dim=0, keepdim=True)
            )

        orig_loss, orig_acc = self._compute_loss_acc(orig_logits)
        zero_loss, zero_acc = self._compute_loss_acc(zero_logits)
        mean_loss, mean_acc = self._compute_loss_acc(mean_logits)

        return dict(
            original_loss=orig_loss,
            original_acc=orig_acc,
            zero_ablated_loss=zero_loss,
            zero_ablated_acc=zero_acc,
            mean_ablated_loss=mean_loss,
            mean_ablated_acc=mean_acc,
            loss_ratio_zero=zero_loss / orig_loss if orig_loss > 0 else float("inf"),
            loss_ratio_mean=mean_loss / orig_loss if orig_loss > 0 else float("inf"),
        )
