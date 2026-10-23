"""Imputation methods for one incomplete continuous variable (wage).

These are Python ports of the eleven ``mice`` methods compared in the original thesis:
``lm`` (complete-case), ``mean``, ``norm.predict``, ``norm.nob``, ``norm``, ``norm.boot``, ``pmm``,
``midastouch``, ``sample``, ``cart`` and ``rf``.

Every function works on numpy arrays on the *modelling scale* (the caller decides whether that is the
raw or the log wage) and returns one imputed value per missing row. Call it ``m`` times with different
random state to obtain ``m`` imputations.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

METHODS = [
    "lm",
    "mean",
    "norm.predict",
    "norm.nob",
    "norm",
    "norm.boot",
    "pmm",
    "midastouch",
    "sample",
    "cart",
    "rf",
]
# Methods that need donor values (individual-level data) inside the saved model artifact.
DONOR_METHODS = {"sample", "pmm", "midastouch", "cart", "rf"}


def _with_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


def _ols(X: np.ndarray, y: np.ndarray):
    """Ordinary least squares with intercept. Returns beta, residual variance, (X'X)^-1, dof."""
    A = _with_intercept(X)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ beta
    dof = max(len(y) - A.shape[1], 1)
    sigma2 = float(res @ res) / dof
    return beta, sigma2, np.linalg.pinv(A.T @ A), dof


def _draw_parameters(beta, sigma2, xtx_inv, dof, rng):
    """Bayesian draw of (beta, sigma) under a flat prior, as in ``mice.impute.norm``."""
    sigma_star = np.sqrt(sigma2 * dof / rng.chisquare(dof))
    try:
        chol = np.linalg.cholesky(xtx_inv + 1e-10 * np.eye(len(beta)))
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(xtx_inv)
        chol = v @ np.diag(np.sqrt(np.clip(w, 0, None)))
    beta_star = beta + sigma_star * chol @ rng.standard_normal(len(beta))
    return beta_star, sigma_star


def _nearest_donors(yhat_obs: np.ndarray, yhat_mis: np.ndarray, k: int):
    """Indices of the ``k`` observed rows whose predicted value is closest to each missing row's."""
    order = np.argsort(yhat_obs)
    sorted_obs = yhat_obs[order]
    k = int(min(k, len(sorted_obs)))
    pos = np.searchsorted(sorted_obs, yhat_mis)
    window = np.arange(-k, k)[None, :] + pos[:, None]
    window = np.clip(window, 0, len(sorted_obs) - 1)
    dist = np.abs(sorted_obs[window] - yhat_mis[:, None])
    nearest = np.argpartition(dist, k - 1, axis=1)[:, :k]
    picked = np.take_along_axis(window, nearest, axis=1)
    d = np.take_along_axis(dist, nearest, axis=1)
    return order[picked], d


def _leaf_donors(leaf_obs: np.ndarray, leaf_mis: np.ndarray):
    by_leaf: dict[int, list[int]] = {}
    for i, leaf in enumerate(leaf_obs):
        by_leaf.setdefault(int(leaf), []).append(i)
    return [by_leaf.get(int(leaf), []) for leaf in leaf_mis]


def impute_values(
    method: str,
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    X_mis: np.ndarray,
    rng: np.random.Generator,
    k: int = 5,
    n_trees: int = 10,
    min_leaf: int = 5,
):
    """Return one imputation (array of length ``len(X_mis)``) or ``None`` for complete-case (``lm``)."""
    n_mis = len(X_mis)
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    if method == "lm":
        return None
    if n_mis == 0:
        return np.empty(0)
    if method == "mean":
        return np.full(n_mis, y_obs.mean())
    if method == "sample":
        return rng.choice(y_obs, size=n_mis, replace=True)

    if method in {"norm.predict", "norm.nob", "norm", "norm.boot", "pmm", "midastouch"}:
        beta, sigma2, xtx_inv, dof = _ols(X_obs, y_obs)
        A_mis = _with_intercept(X_mis)
        if method == "norm.predict":
            return A_mis @ beta
        if method == "norm.nob":
            return A_mis @ beta + np.sqrt(sigma2) * rng.standard_normal(n_mis)
        if method == "norm":
            b_star, s_star = _draw_parameters(beta, sigma2, xtx_inv, dof, rng)
            return A_mis @ b_star + s_star * rng.standard_normal(n_mis)
        if method == "norm.boot":
            idx = rng.integers(0, len(y_obs), len(y_obs))
            b_boot, s2_boot, _, _ = _ols(X_obs[idx], y_obs[idx])
            return A_mis @ b_boot + np.sqrt(s2_boot) * rng.standard_normal(n_mis)
        # donor-based regression methods: match on predicted means (mice "type 1" matching)
        b_star, _ = _draw_parameters(beta, sigma2, xtx_inv, dof, rng)
        yhat_obs = _with_intercept(X_obs) @ beta
        yhat_mis = A_mis @ b_star
        if method == "pmm":
            donors, _ = _nearest_donors(yhat_obs, yhat_mis, k)
            choice = donors[np.arange(n_mis), rng.integers(0, donors.shape[1], n_mis)]
            return y_obs[choice]
        # midastouch: distance-weighted donor draw (kappa = 3), approximated on the 10 nearest donors
        donors, d = _nearest_donors(yhat_obs, yhat_mis, max(k, 10))
        w = 1.0 / np.power(d + 1e-8, 3)
        w = w / w.sum(axis=1, keepdims=True)
        u = rng.random(n_mis)[:, None]
        pick = (u > np.cumsum(w, axis=1)).sum(axis=1).clip(0, donors.shape[1] - 1)
        return y_obs[donors[np.arange(n_mis), pick]]

    if method == "cart":
        idx = rng.integers(0, len(y_obs), len(y_obs))
        tree = DecisionTreeRegressor(min_samples_leaf=min_leaf, random_state=int(rng.integers(1 << 31)))
        tree.fit(X_obs[idx], y_obs[idx])
        pools = _leaf_donors(tree.apply(X_obs), tree.apply(X_mis))
        return np.array([y_obs[rng.choice(p)] if p else rng.choice(y_obs) for p in pools])

    # random forest: pool the donors found in the matching leaf of every tree, then draw one
    forest = RandomForestRegressor(
        n_estimators=n_trees,
        min_samples_leaf=min_leaf,
        max_features=max(1, int(np.sqrt(X_obs.shape[1]))),
        bootstrap=True,
        random_state=int(rng.integers(1 << 31)),
        n_jobs=1,
    )
    forest.fit(X_obs, y_obs)
    leaf_obs, leaf_mis = forest.apply(X_obs), forest.apply(X_mis)
    per_tree = [_leaf_donors(leaf_obs[:, t], leaf_mis[:, t]) for t in range(n_trees)]
    out = np.empty(n_mis)
    for i in range(n_mis):
        pool = np.concatenate([np.asarray(per_tree[t][i], dtype=int) for t in range(n_trees)])
        out[i] = y_obs[rng.choice(pool)] if len(pool) else rng.choice(y_obs)
    return out
