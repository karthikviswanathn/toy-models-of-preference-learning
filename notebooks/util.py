import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from trainer.utils import fourier_transform_2d, get_fourier_basis_names


def train_probe(X, y, train_frac=0.7, seed=42, **kwargs):
    """Train a logistic regression probe and return (clf, train_acc, test_acc).

    Parameters
    ----------
    X : array (n_samples, n_features)
    y : array (n_samples,)  — binary labels (0/1 or bool)
    train_frac : float
    seed : int
    **kwargs : forwarded to LogisticRegression

    Returns
    -------
    clf, train_acc, test_acc
    """
    kwargs.setdefault("max_iter", 1000)
    kwargs.setdefault("random_state", seed)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    split = int(len(X) * train_frac)
    tr, te = idx[:split], idx[split:]

    clf = LogisticRegression(**kwargs)
    clf.fit(X[tr], y[tr])
    return clf, clf.score(X[tr], y[tr]), clf.score(X[te], y[te])


def top_fourier_2d(arr_2d, p, top_k=10):
    """2D Fourier transform of a (p, p) array, return top-k components as a DataFrame."""
    freq_names = get_fourier_basis_names(p)
    ft = fourier_transform_2d(arr_2d, p=p)

    flat = ft.flatten()
    flat_abs = np.abs(flat)
    top_idx = np.argsort(flat_abs)[::-1][:top_k]

    rows = []
    for rank, idx in enumerate(top_idx):
        i, j = idx // p, idx % p
        rows.append({
            "Rank": rank + 1,
            "Freq (a)": freq_names[i],
            "Freq (b)": freq_names[j],
            "Coefficient": flat[idx],
            "|Coefficient|": flat_abs[idx],
        })
    return pd.DataFrame(rows)
