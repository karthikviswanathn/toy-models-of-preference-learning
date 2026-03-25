"""
Transformer model for modular arithmetic using transformer_lens.

Uses HookedTransformer for built-in interpretability support.
Default configuration matches Nanda et al. "Progress measures for grokking".
"""

from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from .tokenizer import ModularAdditionTokenizer


def create_model(
    p: int = 106,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 1,
    d_mlp: int = 512,
    n_ctx: int = 6,
    device: str = "cuda",
    seed: int | None = None,
) -> HookedTransformer:
    """
    Create a transformer for modular addition.

    Uses full vocab from tokenizer (p + 4: numbers, =, <bos>, <eos>, <pad>).

    Args:
        p: Modulus
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_mlp: MLP hidden dimension
        n_ctx: Context length
        device: Device to place model on
        seed: Random seed for reproducible initialization. If None, a random seed is generated.

    Returns:
        HookedTransformer configured for modular addition
    """
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31))

    tokenizer = ModularAdditionTokenizer(p)

    config = HookedTransformerConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_mlp=d_mlp,
        d_vocab=tokenizer.vocab_size,
        device=device,
        act_fn="relu",
        normalization_type=None,  # No LayerNorm (matches Nanda et. al.)
        positional_embedding_type="standard",
        seed=seed,
    )

    model = HookedTransformer(config)
    return model