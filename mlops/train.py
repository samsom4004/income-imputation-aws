"""SageMaker training entry point (script mode, scikit-learn container).

Reads CSV files from the ``train`` channel (or generates synthetic data when ``--synthetic`` is set),
fits the imputation model, runs a small simulation study and writes:

* ``$SM_MODEL_DIR/model.joblib``   the model artefact (picked up by SageMaker as ``model.tar.gz``)
* ``$SM_OUTPUT_DATA_DIR/metrics.json``  aggregate quality metrics
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import pandas as pd  # noqa: E402

from income_imputation.data import load_csv, make_synthetic, validate  # noqa: E402
from income_imputation.evaluate import run_simulation, summarise  # noqa: E402
from income_imputation.model import ImputationModel  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="norm")
    p.add_argument("--m", type=int, default=5)
    p.add_argument("--n-sim", type=int, default=10)
    p.add_argument("--mechanism", default="mar")
    p.add_argument("--rate", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--synthetic", action="store_true", help="train on synthetic data (demo / smoke test)")
    p.add_argument("--synthetic-n", type=int, default=3000)
    p.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    p.add_argument("--output-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    return p.parse_args(argv)


def load_training_frame(args) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(args.train, "*.csv")))
    if args.synthetic or not files:
        if not args.synthetic:
            print("no CSV in the train channel; falling back to synthetic data", flush=True)
        return make_synthetic(n=args.synthetic_n, seed=args.seed)
    return pd.concat([load_csv(f) for f in files], ignore_index=True)


def main(argv=None):
    args = parse_args(argv)
    df = load_training_frame(args)
    problems = validate(df)
    if problems:
        raise SystemExit("data problems: " + "; ".join(problems))

    model = ImputationModel(method=args.method, m=args.m, seed=args.seed).fit(df)
    os.makedirs(args.model_dir, exist_ok=True)
    model.save(os.path.join(args.model_dir, "model.joblib"))

    complete = df[df["wage"].notna()].copy()
    complete["wage_true"] = complete["wage"]
    table = run_simulation(complete, [args.method, "mean"], mechanism=args.mechanism, rate=args.rate,
                           n_sim=args.n_sim, m=args.m, seed=args.seed)
    summary = summarise(table, args.mechanism, args.rate, args.n_sim, args.m)
    mine = table[table["method"].astype(str) == args.method]
    summary["selected_method"] = args.method
    summary["min_coverage"] = float(mine["coverage"].min())
    summary["max_abs_raw_bias_mean_wage"] = float(mine[mine["estimand"] == "mean_wage"]["raw_bias"].abs().max())
    summary["contains_individual_data"] = model.contains_individual_data

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)


if __name__ == "__main__":
    main()
