"""Shared utilities for toy models of preference training experiments."""

import numpy as np
import torch


def generate_preference_gated_data(tokenizer, device="cuda", unsafe_threshold=57):
    """Generate all p^2 preference-gated examples.

    Preferred (result < threshold):  <bos> a b = c <eos>  -> predict c and <eos>
    Unpreferred (result >= threshold): <bos> a b = U <eos>  -> predict U and <eos>

    Both formats are symmetric: [bos, a, b, =, X, eos] with mask [0,0,0,1,1].
    """
    p = tokenizer.p
    bos = tokenizer.bos_token_id
    eq = tokenizer.eq_token_id
    eos = tokenizer.eos_token_id
    unsafe = tokenizer.unsafe_token_id

    inputs, labels = [], []

    for a in range(p):
        for b in range(p):
            result = (a + b) % p
            if result < unsafe_threshold:
                inputs.append([bos, a, b, eq, result, eos])
                labels.append(result)
            else:
                inputs.append([bos, a, b, eq, unsafe, eos])
                labels.append(unsafe)

    n = p * p
    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    loss_mask = torch.zeros(n, 5, dtype=torch.float32, device=device)
    loss_mask[:, 3:5] = 1.0
    is_preferred = torch.tensor([(a + b) % p < unsafe_threshold for a in range(p) for b in range(p)],
                                dtype=torch.bool, device=device)
    return inputs, labels, loss_mask, is_preferred


def split_data(inputs, labels, loss_mask, is_preferred, train_frac, rng):
    """Deterministic train/test split using numpy RNG."""
    n = inputs.shape[0]
    indices = rng.permutation(n)
    split = int(train_frac * n)
    tr, te = indices[:split], indices[split:]
    return (inputs[tr], labels[tr], loss_mask[tr], is_preferred[tr],
            inputs[te], labels[te], loss_mask[te], is_preferred[te])


def eval_model(model, inputs, labels, loss_mask, is_preferred):
    """Compute loss and accuracy (overall, preferred, unpreferred)."""
    with torch.no_grad():
        logits, per_token_loss = model(inputs, return_type="both", loss_per_token=True)
        loss = (per_token_loss * loss_mask).sum() / loss_mask.sum()

        preds = logits[:, 3].argmax(-1)
        acc = (preds == labels).float().mean()

        unpreferred = ~is_preferred
        acc_preferred = (preds[is_preferred] == labels[is_preferred]).float().mean() if is_preferred.any() else torch.tensor(0.0)
        acc_unpreferred = (preds[unpreferred] == labels[unpreferred]).float().mean() if unpreferred.any() else torch.tensor(0.0)

    return loss.item(), acc.item(), acc_preferred.item(), acc_unpreferred.item()


# ---------------------------------------------------------------------------
# Fourier analysis utilities (originally from notebooks/utils.py)
# ---------------------------------------------------------------------------

def get_fourier_basis(p, device):
    """Orthonormal Fourier basis for Z_p (p must be odd)."""
    assert p % 2 == 1, "Input should be odd"
    fourier_basis = [torch.ones(p) / np.sqrt(p)]
    for i in range(1, p // 2 + 1):
        cos_vec = torch.cos(2 * torch.pi * torch.arange(p) * i / p)
        sin_vec = torch.sin(2 * torch.pi * torch.arange(p) * i / p)
        cos_vec /= cos_vec.norm()
        sin_vec /= sin_vec.norm()
        fourier_basis.append(cos_vec)
        fourier_basis.append(sin_vec)
    return torch.stack(fourier_basis, dim=0).to(device)


def get_fourier_basis_names(p):
    """Names for Fourier basis vectors: ['Const', 'cos 1', 'sin 1', ...]."""
    assert p % 2 == 1, "Input should be odd"
    names = ['Const']
    for i in range(1, p // 2 + 1):
        names.append(f'cos {i}')
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
