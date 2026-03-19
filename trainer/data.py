"""
Data generation for 2-argument modular addition pretraining.

Format: <bos> a b = result <eos>
"""

from typing import Tuple
import numpy as np
import torch

from .tokenizer import ModularAdditionTokenizer


def generate_all_data(
    tokenizer: ModularAdditionTokenizer,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate all possible 2-arg modular addition examples.

    Args:
        tokenizer: Tokenizer instance
        device: Device to place tensors on

    Returns:
        inputs: Tensor of shape (p*p, 6) with [<bos>, a, b, =, result, <eos>]
        labels: Tensor of shape (p*p,) with (a + b) mod p
    """
    p = tokenizer.p

    inputs = []
    labels = []

    for a in range(p):
        for b in range(p):
            result = (a + b) % p
            inputs.append([
                tokenizer.bos_token_id,
                a,
                b,
                tokenizer.eq_token_id,
                result,
                tokenizer.eos_token_id,
            ])
            labels.append(result)

    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)

    return inputs, labels


def train_test_split(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    train_frac: float,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split data into train and test sets.

    Args:
        inputs: Input tensor of shape (N, 6)
        labels: Label tensor of shape (N,)
        train_frac: Fraction of data to use for training
        rng: Numpy random generator

    Returns:
        train_inputs, train_labels, test_inputs, test_labels
    """
    n = inputs.shape[0]
    indices = rng.permutation(n)

    split_idx = int(train_frac * n)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    train_inputs = inputs[train_idx]
    train_labels = labels[train_idx]
    test_inputs = inputs[test_idx]
    test_labels = labels[test_idx]

    return train_inputs, train_labels, test_inputs, test_labels


def generate_pretrain_data(
    tokenizer: ModularAdditionTokenizer,
    train_frac: float,
    rng: np.random.Generator,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate train/test data for 2-arg modular addition.

    Args:
        tokenizer: Tokenizer instance
        train_frac: Fraction of data for training (default 0.3 matches Nanda)
        rng: Numpy random generator
        device: Device to place tensors on

    Returns:
        train_inputs, train_labels, test_inputs, test_labels

    Example:
        >>> tokenizer = ModularAdditionTokenizer(113)
        >>> rng = np.random.default_rng(42)
        >>> train_x, train_y, test_x, test_y = generate_pretrain_data(tokenizer, 0.3, rng)
        >>> train_x.shape  # (3831, 6) for 30% of 113^2
        >>> train_y.shape  # (3831,)
    """
    inputs, labels = generate_all_data(tokenizer, device)
    return train_test_split(inputs, labels, train_frac, rng)
