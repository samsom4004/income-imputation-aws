"""Rubin's rules for combining estimates from ``m`` imputed datasets, plus the analysis models used."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .data import design_matrix


def rubin(q: np.ndarray, u: np.ndarray) -> dict:
    """Pool ``m`` point estimates ``q`` and their sampling variances ``u`` (Rubin, 1987)."""
    q, u = np.asarray(q, dtype=float), np.asarray(u, dtype=float)
    m = len(q)
    qbar, ubar = q.mean(), u.mean()
    if m == 1:
        se = np.sqrt(ubar)
        z = stats.norm.ppf(0.975)
        return {"qbar": qbar, "ubar": ubar, "b": 0.0, "t": ubar, "df": np.inf, "lo": qbar - z * se, "hi": qbar + z * se}
    b = q.var(ddof=1)
    t = ubar + (1 + 1 / m) * b
    r = (1 + 1 / m) * b / ubar if ubar > 0 else np.inf
    df = (m - 1) * (1 + 1 / r) ** 2 if np.isfinite(r) and r > 0 else np.inf
    crit = stats.t.ppf(0.975, df) if np.isfinite(df) else stats.norm.ppf(0.975)
    se = np.sqrt(t)
    return {"qbar": qbar, "ubar": ubar, "b": b, "t": t, "df": df, "lo": qbar - crit * se, "hi": qbar + crit * se}


ESTIMANDS = ["mean_wage", "coef_age", "coef_no_qual"]


def analyse(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Complete-data analysis. Returns estimates and their sampling variances for ``ESTIMANDS``.

    * ``mean_wage``    - mean wage (raw scale), variance s^2 / n
    * ``coef_age``     - OLS coefficient of age in log(wage) ~ covariates
    * ``coef_no_qual`` - OLS coefficient of "no qualification" in the same model
    """
    y = df["wage"].to_numpy(dtype=float)
    n = len(y)
    X = np.column_stack([np.ones(n), design_matrix(df).to_numpy()])
    names = ["const"] + list(design_matrix(df).columns)
    ly = np.log(np.clip(y, 1e-6, None))
    beta, *_ = np.linalg.lstsq(X, ly, rcond=None)
    res = ly - X @ beta
    s2 = float(res @ res) / max(n - X.shape[1], 1)
    cov = s2 * np.linalg.pinv(X.T @ X)
    ia, iq = names.index("age"), names.index("hiqual=No qual")
    q = np.array([y.mean(), beta[ia], beta[iq]])
    u = np.array([y.var(ddof=1) / n, cov[ia, ia], cov[iq, iq]])
    return q, u
