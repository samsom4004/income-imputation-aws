"""SageMaker Processing step: evaluate a trained model on held-out data and write ``evaluation.json``.

Inputs:  ``/opt/ml/processing/model/model.tar.gz`` and ``/opt/ml/processing/test/test.csv``
Output:  ``/opt/ml/processing/evaluation/evaluation.json`` containing ``{"metrics": {...}}``
The pipeline's condition step reads ``metrics.min_coverage.value`` from this file.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import tarfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from income_imputation.data import load_csv  # noqa: E402
from income_imputation.evaluate import run_simulation  # noqa: E402
from income_imputation.model import ImputationModel  # noqa: E402

BASE = "/opt/ml/processing"


def main(base=BASE):
    with tarfile.open(os.path.join(base, "model", "model.tar.gz")) as tar:
        tar.extractall(os.path.join(base, "model"), filter="data") if hasattr(tarfile, "data_filter") else tar.extractall(os.path.join(base, "model"))
    model = ImputationModel.load(os.path.join(base, "model", "model.joblib"))
    test = load_csv(glob.glob(os.path.join(base, "test", "*.csv"))[0])
    complete = test[test["wage"].notna()].copy()
    complete["wage_true"] = complete["wage"]
    table = run_simulation(complete, [model.method], mechanism="mar", rate=0.15, n_sim=10, m=model.m, seed=1)
    cov = float(table["coverage"].min())
    bias = float(table[table["estimand"] == "mean_wage"]["raw_bias"].abs().max())
    out = {"metrics": {"min_coverage": {"value": cov}, "abs_bias_mean_wage": {"value": bias}, "n_test": {"value": int(len(complete))}}}
    os.makedirs(os.path.join(base, "evaluation"), exist_ok=True)
    with open(os.path.join(base, "evaluation", "evaluation.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
