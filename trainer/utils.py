"""Shared utilities for toy models of preference training experiments."""

import numpy as np
import torch


def generate_parity_gated_data(tokenizer, device="cuda"):
    """Generate all p^2 parity-gated examples.

    Even result: <bos> a b = c <eos>    -> predict c and <eos>
    Odd  result: <bos> a b = <eos> <pad> -> predict <eos> only
    """
    p = tokenizer.p
    bos = tokenizer.bos_token_id
    eq = tokenizer.eq_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id

    inputs, labels, masks = [], [], []

    for a in range(p):
        for b in range(p):
            result = (a + b) % p
            if result % 2 == 0:
                inputs.append([bos, a, b, eq, result, eos])
                labels.append(result)
                masks.append([0, 0, 0, 1, 1])
            else:
                inputs.append([bos, a, b, eq, eos, pad])
                labels.append(eos)
                masks.append([0, 0, 0, 1, 0])

    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    loss_mask = torch.tensor(masks, dtype=torch.float32, device=device)
    is_even = torch.tensor([(a + b) % p % 2 == 0 for a in range(p) for b in range(p)],
                           dtype=torch.bool, device=device)
    return inputs, labels, loss_mask, is_even


def split_data(inputs, labels, loss_mask, is_even, train_frac, rng):
    """Deterministic train/test split using numpy RNG."""
    n = inputs.shape[0]
    indices = rng.permutation(n)
    split = int(train_frac * n)
    tr, te = indices[:split], indices[split:]
    return (inputs[tr], labels[tr], loss_mask[tr], is_even[tr],
            inputs[te], labels[te], loss_mask[te], is_even[te])


def eval_model(model, inputs, labels, loss_mask, is_even):
    """Compute loss and accuracy (overall, even, odd)."""
    with torch.no_grad():
        logits, per_token_loss = model(inputs, return_type="both", loss_per_token=True)
        loss = (per_token_loss * loss_mask).sum() / loss_mask.sum()

        preds = logits[:, 3].argmax(-1)
        acc = (preds == labels).float().mean()

        odd = ~is_even
        acc_even = (preds[is_even] == labels[is_even]).float().mean() if is_even.any() else torch.tensor(0.0)
        acc_odd = (preds[odd] == labels[odd]).float().mean() if odd.any() else torch.tensor(0.0)

    return loss.item(), acc.item(), acc_even.item(), acc_odd.item()


# ---------------------------------------------------------------------------
# Fourier analysis utilities (originally from notebooks/utils.py)
# ---------------------------------------------------------------------------

def get_fourier_basis(p, device):
    """Orthonormal Fourier basis for Z_p (works for any p >= 2).

    For odd p: 1 constant + (p-1)/2 cos/sin pairs = p vectors.
    For even p: 1 constant + (p/2-1) cos/sin pairs + 1 Nyquist cos = p vectors.
    (The Nyquist sine is identically zero and is omitted.)
    """
    fourier_basis = [torch.ones(p) / np.sqrt(p)]
    max_freq = p // 2
    for i in range(1, max_freq + 1):
        cos_vec = torch.cos(2 * torch.pi * torch.arange(p) * i / p)
        cos_vec /= cos_vec.norm()
        fourier_basis.append(cos_vec)
        if i < max_freq or p % 2 == 1:
            sin_vec = torch.sin(2 * torch.pi * torch.arange(p) * i / p)
            sin_vec /= sin_vec.norm()
            fourier_basis.append(sin_vec)
    return torch.stack(fourier_basis, dim=0).to(device)


def get_fourier_basis_names(p):
    """Names for Fourier basis vectors: ['Const', 'cos 1', 'sin 1', ...]."""
    names = ['Const']
    max_freq = p // 2
    for i in range(1, max_freq + 1):
        names.append(f'cos {i}')
        if i < max_freq or p % 2 == 1:
            names.append(f'sin {i}')
    return names


def fourier_transform_1d(arr, fourier_basis=None, p=None):
    """Project a 1D array onto the Fourier basis."""
    if fourier_basis is None and p is None:
        raise ValueError("Both fourier_basis and p cannot be None")
    is_numpy = isinstance(arr, np.ndarray)
    if is_numpy:
        arr = torch.tensor(arr, dtype=torch.float32)
    if fourier_basis is None:
        fourier_basis = get_fourier_basis(p, arr.device)
    result = arr @ fourier_basis.T
    return result.numpy() if is_numpy else result


def fourier_transform_2d(arr, fourier_basis=None, p=None):
    """2D Fourier decomposition: F @ arr @ F^T."""
    if fourier_basis is None and p is None:
        raise ValueError("Both fourier_basis and p cannot be None")
    is_numpy = isinstance(arr, np.ndarray)
    if is_numpy:
        arr = torch.tensor(arr, dtype=torch.float32)
    if fourier_basis is None:
        fourier_basis = get_fourier_basis(p, arr.device)
    result = torch.einsum('ij,...jk,lk->...il', fourier_basis, arr, fourier_basis)
    return result.numpy() if is_numpy else result
