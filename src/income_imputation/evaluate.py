"""Simulation study that compares imputation methods (the thesis's "design 2": missing-data mechanism).

For each simulated run: induce missingness in a complete dataset, impute ``m`` times, analyse each
completed dataset, pool with Rubin's rules and compare with the complete-data estimate.

Metrics (as in the thesis): raw bias (RB), coverage rate (CR) of the 95% interval, average interval
width (AW) and RMSE. We add the RMSE of the imputed values themselves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TARGET
from .missingness import MECHANISMS
from .model import ImputationModel
from .pooling import ESTIMANDS, analyse, rubin


def _complete_case(df: pd.DataFrame):
    obs = df[df[TARGET].notna()]
    q, u = analyse(obs)
    return [rubin([q[i]], [u[i]]) for i in range(len(q))], None


def run_simulation(
    complete: pd.DataFrame,
    methods: list[str],
    mechanism: str = "mar",
    rate: float = 0.15,
    n_sim: int = 20,
    m: int = 5,
    seed: int = 0,
    log_target: bool = True,
) -> pd.DataFrame:
    """Return one row per (method, estimand) with bias, coverage, width and RMSE."""
    base = complete.copy()
    base["wage_true"] = base["wage_true"] if "wage_true" in base.columns else base[TARGET]
    base[TARGET] = base["wage_true"]
    q_true, _ = analyse(base)
    rows: list[dict] = []
    for s in range(n_sim):
        inc = MECHANISMS[mechanism](base, rate=rate, seed=seed + s)
        miss = inc[TARGET].isna().to_numpy()
        for method in methods:
            if method == "lm":
                pooled, imp_rmse = _complete_case(inc)[0], np.nan
            else:
                model = ImputationModel(method=method, m=m, log_target=log_target, seed=seed + s, max_donors=None).fit(inc)
                d = model.draws(inc, m=m, seed=seed + s)
                qs, us = [], []
                for j in range(m):
                    filled = inc.copy()
                    filled.loc[miss, TARGET] = d[:, j]
                    q, u = analyse(filled)
                    qs.append(q)
                    us.append(u)
                qs, us = np.array(qs), np.array(us)
                pooled = [rubin(qs[:, i], us[:, i]) for i in range(len(ESTIMANDS))]
                truth = inc.loc[miss, "wage_true"].to_numpy()
                imp_rmse = float(np.sqrt(np.mean((d - truth[:, None]) ** 2)))
            for i, name in enumerate(ESTIMANDS):
                p = pooled[i]
                rows.append(
                    {
                        "sim": s,
                        "method": method,
                        "estimand": name,
                        "bias": p["qbar"] - q_true[i],
                        "covered": float(p["lo"] <= q_true[i] <= p["hi"]),
                        "width": p["hi"] - p["lo"],
                        "sq_err": (p["qbar"] - q_true[i]) ** 2,
                        "impute_rmse": imp_rmse,
                    }
                )
    res = pd.DataFrame(rows)
    out = (
        res.groupby(["method", "estimand"])
        .agg(raw_bias=("bias", "mean"), coverage=("covered", "mean"), avg_width=("width", "mean"),
             rmse=("sq_err", lambda x: float(np.sqrt(np.mean(x)))), impute_rmse=("impute_rmse", "mean"))
        .reset_index()
    )
    out["method"] = pd.Categorical(out["method"], categories=methods, ordered=True)
    return out.sort_values(["estimand", "method"]).reset_index(drop=True)


def summarise(table: pd.DataFrame, mechanism: str, rate: float, n_sim: int, m: int) -> dict:
    """Compact JSON summary (aggregate numbers only - safe to send to a language model)."""
    best = {}
    for est, g in table.groupby("estimand", observed=True):
        ok = g[(g["coverage"] >= 0.9)]
        pick = (ok if len(ok) else g).copy()
        pick["score"] = pick["rmse"] / pick["rmse"].max()
        best[est] = str(pick.sort_values("score").iloc[0]["method"])
    return {
        "mechanism": mechanism,
        "missing_rate": rate,
        "n_simulations": n_sim,
        "imputations_m": m,
        "best_method_by_estimand": best,
        "results": json_ready(table),
    }


def json_ready(table: pd.DataFrame) -> list[dict]:
    t = table.copy()
    t["method"] = t["method"].astype(str)
    return [{k: (None if (isinstance(v, float) and np.isnan(v)) else (round(v, 4) if isinstance(v, float) else v)) for k, v in r.items()} for r in t.to_dict(orient="records")]
