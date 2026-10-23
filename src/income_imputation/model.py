"""The deployable artefact: a fitted multiple-imputation model with a stable serving interface."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import COVARIATES, TARGET, design_matrix
from .imputers import DONOR_METHODS, METHODS, impute_values

SCHEMA_VERSION = 1


@dataclass
class ImputationModel:
    """Fit on rows where ``wage`` is observed; impute ``m`` plausible wages for rows where it is not.

    ``log_target`` models log(wage). Wages are right-skewed, so this keeps imputations positive and
    the linear-model assumptions much closer to true.

    Privacy note: donor methods (``sample``, ``pmm``, ``midastouch``, ``cart``, ``rf``) store observed
    wages inside the artefact. Treat such artefacts as sensitive data. The parametric ``norm`` method
    stores only coefficients and is the recommended default for shared or hosted endpoints.
    """

    method: str = "norm"
    m: int = 5
    log_target: bool = True
    k: int = 5
    max_donors: int | None = 5000
    seed: int = 0
    columns_: list[str] = field(default_factory=list)
    X_obs_: np.ndarray | None = None
    y_obs_: np.ndarray | None = None
    n_train_: int = 0
    n_observed_: int = 0

    # ---------------------------------------------------------------- training
    def fit(self, df: pd.DataFrame) -> "ImputationModel":
        if self.method not in METHODS:
            raise ValueError(f"unknown method {self.method!r}")
        obs = df[df[TARGET].notna()]
        if len(obs) < 30:
            raise ValueError("need at least 30 rows with an observed wage to fit")
        X = design_matrix(obs)
        y = obs[TARGET].to_numpy(dtype=float)
        if (y <= 0).any() and self.log_target:
            raise ValueError("wages must be positive when log_target=True")
        rng = np.random.default_rng(self.seed)
        if self.max_donors and len(y) > self.max_donors:
            keep = rng.choice(len(y), self.max_donors, replace=False)
            X, y = X.iloc[keep], y[keep]
        self.columns_ = list(X.columns)
        self.X_obs_ = X.to_numpy(dtype=float)
        self.y_obs_ = np.log(y) if self.log_target else y
        self.n_train_, self.n_observed_ = len(df), len(obs)
        return self

    # --------------------------------------------------------------- inference
    def draws(self, df: pd.DataFrame, m: int | None = None, seed: int | None = None) -> np.ndarray:
        """Return an array (n_missing, m) of imputed wages for rows whose wage is missing."""
        self._check_fitted()
        m = m or self.m
        miss = self._missing_mask(df)
        if self.method == "lm":
            raise ValueError("'lm' is complete-case analysis and cannot impute rows")
        X_mis = design_matrix(df[miss]).reindex(columns=self.columns_, fill_value=0.0).to_numpy(dtype=float)
        out = np.empty((int(miss.sum()), m))
        for j in range(m):
            rng = np.random.default_rng((self.seed if seed is None else seed) + 1000 * j)
            v = impute_values(self.method, self.X_obs_, self.y_obs_, X_mis, rng, k=self.k)
            out[:, j] = np.exp(v) if self.log_target else v
        return out

    def transform(self, df: pd.DataFrame, m: int | None = None, seed: int | None = None) -> pd.DataFrame:
        """Return ``df`` plus ``wage_imputed`` (observed wage or the mean of the ``m`` draws), the 5th and
        95th percentile across draws, and ``was_imputed``."""
        df = df.copy()
        if TARGET not in df.columns:
            df[TARGET] = np.nan
        miss = self._missing_mask(df)
        d = self.draws(df, m=m, seed=seed)
        df["wage_imputed"] = df[TARGET].astype(float)
        df["wage_p05"] = df[TARGET].astype(float)
        df["wage_p95"] = df[TARGET].astype(float)
        if miss.any():
            df.loc[miss, "wage_imputed"] = d.mean(axis=1)
            df.loc[miss, "wage_p05"] = np.percentile(d, 5, axis=1)
            df.loc[miss, "wage_p95"] = np.percentile(d, 95, axis=1)
        df["was_imputed"] = miss.astype(int)
        return df

    def predict_records(self, records: list[dict], m: int | None = None, seed: int | None = None) -> list[dict]:
        """JSON-friendly API used by the SageMaker endpoint."""
        df = pd.DataFrame(records)
        missing = [c for c in COVARIATES if c not in df.columns]
        if missing:
            raise ValueError(f"missing required fields: {missing}")
        out = self.transform(df, m=m, seed=seed)
        cols = ["wage_imputed", "wage_p05", "wage_p95", "was_imputed"]
        return json.loads(out[cols].round(2).to_json(orient="records"))

    # -------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"schema": SCHEMA_VERSION, "model": self}, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "ImputationModel":
        blob = joblib.load(path)
        if blob.get("schema") != SCHEMA_VERSION:
            raise ValueError("model artefact has an unsupported schema version")
        return blob["model"]

    @property
    def contains_individual_data(self) -> bool:
        return self.method in DONOR_METHODS

    # ----------------------------------------------------------------- helpers
    def _missing_mask(self, df: pd.DataFrame) -> pd.Series:
        return df[TARGET].isna() if TARGET in df.columns else pd.Series(True, index=df.index)

    def _check_fitted(self):
        if self.X_obs_ is None:
            raise RuntimeError("model is not fitted")
