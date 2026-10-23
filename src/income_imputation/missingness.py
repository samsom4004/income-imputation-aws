"""Missing-data mechanisms used to build controlled simulation studies.

The thesis found (logistic regression on the missing-wage indicator) that missingness depends on age,
sex, qualification and job status: a Missing At Random (MAR) process. We reproduce MCAR, MAR and MNAR
so imputation methods can be stress-tested under all three.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRUE_COL = "wage_true"


def _calibrate(score: np.ndarray, rate: float) -> np.ndarray:
    """Find the intercept so that the mean missing probability equals ``rate``."""
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        p = 1 / (1 + np.exp(-(score + mid)))
        lo, hi = (mid, hi) if p.mean() < rate else (lo, mid)
    return 1 / (1 + np.exp(-(score + (lo + hi) / 2)))


def _apply(df: pd.DataFrame, prob: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    if TRUE_COL not in out.columns:
        out[TRUE_COL] = out["wage"]
    miss = rng.random(len(out)) < prob
    out["wage"] = out[TRUE_COL].where(~miss)
    out["wage_missing"] = miss.astype(int)
    return out


def induce_mcar(df: pd.DataFrame, rate: float = 0.15, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return _apply(df, np.full(len(df), rate), rng)


def induce_mar(df: pd.DataFrame, rate: float = 0.15, seed: int = 0) -> pd.DataFrame:
    """Missingness depends on observed covariates only (age, sex, qualification, job status)."""
    rng = np.random.default_rng(seed)
    score = (
        0.02 * (df["age"].to_numpy() - 44)
        + (df["sex"] == "Male").to_numpy() * 0.35
        + df["hiqual"].map({"Degree": 0.6, "Other HE": 0.3, "A-level": 0.0, "GCSE": -0.1, "Other": -0.2, "No qual": -0.4}).to_numpy()
        + df["jb"].map({"FT employed": 0.4, "PT employed": 0.0, "Unemployed": -0.5, "Retired": -0.3, "Other": -0.2}).to_numpy()
    )
    return _apply(df, _calibrate(score, rate), rng)


def induce_mnar(df: pd.DataFrame, rate: float = 0.15, seed: int = 0) -> pd.DataFrame:
    """High earners are more likely to refuse: missingness depends on the unobserved value itself."""
    rng = np.random.default_rng(seed)
    y = df[TRUE_COL] if TRUE_COL in df.columns else df["wage"]
    score = 1.2 * (np.log(y.to_numpy()) - np.log(y.to_numpy()).mean())
    return _apply(df, _calibrate(score, rate), rng)


MECHANISMS = {"mcar": induce_mcar, "mar": induce_mar, "mnar": induce_mnar}
