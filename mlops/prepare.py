"""SageMaker Processing step: validate the input data and write train/test splits.

Input  (optional): ``/opt/ml/processing/input/*.csv``  (if absent, synthetic demo data is generated)
Output:            ``/opt/ml/processing/output/train/train.csv`` and ``.../test/test.csv``
"""

from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import pandas as pd  # noqa: E402

from income_imputation.data import load_csv, make_synthetic, validate  # noqa: E402

BASE = "/opt/ml/processing"


def main(base=BASE, seed=0):
    files = sorted(glob.glob(os.path.join(base, "input", "*.csv")))
    df = pd.concat([load_csv(f) for f in files], ignore_index=True) if files else make_synthetic(n=4000, seed=seed)
    problems = validate(df)
    if problems:
        raise SystemExit("data problems: " + "; ".join(problems))
    df = df.drop(columns=[c for c in ["wage_true", "id"] if c in df.columns])
    test = df.sample(frac=0.2, random_state=seed)
    train = df.drop(test.index)
    for name, part in [("train", train), ("test", test)]:
        out = os.path.join(base, "output", name)
        os.makedirs(out, exist_ok=True)
        part.to_csv(os.path.join(out, f"{name}.csv"), index=False)
    print(f"train={len(train)} test={len(test)}", flush=True)


if __name__ == "__main__":
    main()
