"""Data schema, a realistic synthetic data generator, and a loader for real survey extracts.

The original BSc project (University of Essex, 2019-20) imputed missing *wage* in the first wave of the
UK Household Longitudinal Study (UKHLS / Understanding Society). That data is licensed by the UK Data
Service and must never be committed to a public repository. This module therefore:

* defines the same column schema as the original extract,
* generates *synthetic* data with the same shape and a Missing At Random (MAR) wage mechanism, and
* can load a real extract from a local path or S3 when you are entitled to use it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "wage"
CATEGORICALS = {
    "sex": ["Female", "Male"],
    "health": ["No", "Yes"],
    "hiqual": ["Degree", "Other HE", "A-level", "GCSE", "Other", "No qual"],
    "ethnic": ["White", "Mixed", "Asian", "Black", "Other"],
    "marriage": ["Have partner", "Single", "Separated", "Widowed"],
    "jb": ["FT employed", "PT employed", "Unemployed", "Retired", "Other"],
}
NUMERIC = ["age"]
COVARIATES = NUMERIC + list(CATEGORICALS)


def make_synthetic(n: int = 5000, seed: int = 0, missing_rate: float = 0.15) -> pd.DataFrame:
    """Create a synthetic stand-in for the UKHLS wage extract.

    Returns a frame with the covariates, ``wage`` (NaN where missing), ``wage_true`` (complete) and
    ``wage_missing`` (0/1). Missingness is MAR: it depends on age, sex, qualification and job status.
    """
    rng = np.random.default_rng(seed)
    p = {
        "sex": [0.52, 0.48],
        "health": [0.82, 0.18],
        "hiqual": [0.25, 0.1, 0.2, 0.25, 0.1, 0.1],
        "ethnic": [0.84, 0.03, 0.07, 0.04, 0.02],
        "marriage": [0.58, 0.28, 0.09, 0.05],
        "jb": [0.45, 0.2, 0.05, 0.2, 0.1],
    }
    df = pd.DataFrame({"age": np.clip(rng.normal(44, 15, n), 16, 85).round().astype(int)})
    for col, levels in CATEGORICALS.items():
        df[col] = rng.choice(levels, size=n, p=p[col])
    # job status is related to age
    older = df["age"] > 64
    df.loc[older, "jb"] = rng.choice(["Retired", "PT employed", "Other"], size=int(older.sum()), p=[0.8, 0.12, 0.08])
    young = df["age"] < 21
    df.loc[young, "jb"] = rng.choice(["PT employed", "FT employed", "Unemployed", "Other"], size=int(young.sum()), p=[0.4, 0.2, 0.15, 0.25])

    effect = (
        6.2
        + 0.012 * (df["age"] - 44)
        - 0.00020 * (df["age"] - 44) ** 2
        + df["sex"].map({"Female": 0.0, "Male": 0.14})
        + df["health"].map({"No": 0.0, "Yes": -0.12})
        + df["hiqual"].map({"Degree": 0.35, "Other HE": 0.2, "A-level": 0.08, "GCSE": 0.0, "Other": -0.05, "No qual": -0.2})
        + df["ethnic"].map({"White": 0.0, "Mixed": -0.03, "Asian": -0.05, "Black": -0.08, "Other": -0.05})
        + df["marriage"].map({"Have partner": 0.05, "Single": 0.0, "Separated": -0.03, "Widowed": -0.06})
        + df["jb"].map({"FT employed": 0.35, "PT employed": -0.25, "Unemployed": -0.9, "Retired": -0.7, "Other": -0.6})
    )
    df["wage_true"] = np.exp(effect + rng.normal(0, 0.45, n)).round(0)

    from .missingness import induce_mar  # local import: avoids a cycle

    out = induce_mar(df, rate=missing_rate, seed=seed + 1)
    return out


def load_csv(path: str) -> pd.DataFrame:
    """Load a real extract (columns as in the original ``wage_UKHLS_w1_incomemissingonly.csv``).

    Accepts local paths or ``s3://`` URIs (needs ``s3fs`` for the latter). The string ``"missing"`` and
    empty strings are treated as missing. ``jbstat``/``jbft`` are combined into ``jb`` as in the thesis.
    """
    df = pd.read_csv(path, na_values=["missing", "Missing", "", "NA"])
    if "jb" not in df.columns and {"jbstat", "jbft"}.issubset(df.columns):
        jb = pd.Series("Other", index=df.index)
        jb[(df["jbstat"] == "Employed") & (df["jbft"] == "FT")] = "FT employed"
        jb[(df["jbstat"] == "Employed") & (df["jbft"] == "PT")] = "PT employed"
        jb[df["jbstat"] == "Unemployed"] = "Unemployed"
        jb[df["jbstat"] == "Retired"] = "Retired"
        df["jb"] = jb
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    if "wage_missing" not in df.columns:
        df["wage_missing"] = df[TARGET].isna().astype(int)
    return df


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of data-quality problems (empty list = OK)."""
    problems = []
    for col in COVARIATES + [TARGET]:
        if col not in df.columns:
            problems.append(f"missing column: {col}")
    if problems:
        return problems
    if (df["age"].dropna() < 0).any() or (df["age"].dropna() > 120).any():
        problems.append("age outside 0-120")
    if (df[TARGET].dropna() < 0).any():
        problems.append("negative wage")
    for col, levels in CATEGORICALS.items():
        bad = set(df[col].dropna().unique()) - set(levels)
        if bad:
            problems.append(f"unexpected levels in {col}: {sorted(bad)}")
    rate = df[TARGET].isna().mean()
    if rate > 0.6:
        problems.append(f"wage missing rate too high: {rate:.0%}")
    return problems


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode with *fixed* levels so training and serving always give the same columns."""
    cols = {"age": df["age"].astype(float)}
    for col, levels in CATEGORICALS.items():
        s = df[col].astype(object).where(df[col].isin(levels), levels[0])  # unknown level -> reference
        for lv in levels[1:]:
            cols[f"{col}={lv}"] = (s == lv).astype(float)
    return pd.DataFrame(cols, index=df.index)
