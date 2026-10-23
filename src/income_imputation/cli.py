"""Command line interface: simulate, train, impute, explain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import load_csv, make_synthetic, validate
from .evaluate import run_simulation, summarise
from .imputers import METHODS
from .model import ImputationModel

FAST = ["lm", "mean", "norm.nob", "norm", "pmm", "cart"]


def _load(args) -> pd.DataFrame:
    if args.input in (None, "synthetic"):
        return make_synthetic(n=args.n, seed=args.seed)
    return load_csv(args.input)


def cmd_simulate(args):
    df = _load(args)
    if "wage_true" not in df.columns:  # real data: simulate on its complete cases
        df = df[df["wage"].notna()].copy()
        df["wage_true"] = df["wage"]
    methods = METHODS if args.methods == "all" else args.methods.split(",")
    table = run_simulation(df, methods, mechanism=args.mechanism, rate=args.rate, n_sim=args.n_sim, m=args.m, seed=args.seed)
    print(table.round(4).to_string(index=False))
    summary = summarise(table, args.mechanism, args.rate, args.n_sim, args.m)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=1))
        print(f"\nwrote {args.out}")


def cmd_train(args):
    df = _load(args)
    problems = validate(df)
    if problems:
        raise SystemExit("data problems: " + "; ".join(problems))
    model = ImputationModel(method=args.method, m=args.m, seed=args.seed).fit(df)
    path = model.save(args.out)
    print(f"saved {path} (method={args.method}, individual-level data inside: {model.contains_individual_data})")


def cmd_impute(args):
    model = ImputationModel.load(args.model)
    df = load_csv(args.input)
    out = model.transform(df)
    out.to_csv(args.output, index=False)
    print(f"wrote {args.output}: {int(out['was_imputed'].sum())} of {len(out)} rows imputed")


def cmd_explain(args):
    from .bedrock_report import explain

    metrics = json.loads(Path(args.metrics).read_text())
    print(explain(metrics, dry_run=args.dry_run))


def main(argv=None):
    p = argparse.ArgumentParser(prog="income-imputation")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--input", default="synthetic", help="CSV path, s3:// URI or 'synthetic'")
        sp.add_argument("--n", type=int, default=3000, help="rows when using synthetic data")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--m", type=int, default=5)

    s = sub.add_parser("simulate", help="compare imputation methods")
    common(s)
    s.add_argument("--methods", default=",".join(FAST), help="comma list or 'all'")
    s.add_argument("--mechanism", choices=["mcar", "mar", "mnar"], default="mar")
    s.add_argument("--rate", type=float, default=0.15)
    s.add_argument("--n-sim", type=int, default=10)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_simulate)

    t = sub.add_parser("train", help="fit and save a model")
    common(t)
    t.add_argument("--method", default="norm", choices=METHODS[1:])
    t.add_argument("--out", default="artifacts/model.joblib")
    t.set_defaults(fn=cmd_train)

    i = sub.add_parser("impute", help="impute a CSV with a saved model")
    i.add_argument("--model", required=True)
    i.add_argument("--input", required=True)
    i.add_argument("--output", required=True)
    i.set_defaults(fn=cmd_impute)

    e = sub.add_parser("explain", help="plain-English report with Amazon Bedrock")
    e.add_argument("--metrics", required=True)
    e.add_argument("--dry-run", action="store_true", help="do not call AWS; print a template report")
    e.set_defaults(fn=cmd_explain)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
