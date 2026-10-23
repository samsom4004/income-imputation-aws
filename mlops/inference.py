"""SageMaker inference handlers for the scikit-learn container (script mode).

Request:  ``application/json``  ``{"records": [...], "m": 5}`` or a bare list of records
Response: ``application/json``  ``{"predictions": [{"wage_imputed": ..., "wage_p05": ..., ...}]}``
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from income_imputation.model import ImputationModel  # noqa: E402

MAX_RECORDS = 1000


def model_fn(model_dir):
    return ImputationModel.load(os.path.join(model_dir, "model.joblib"))


def input_fn(request_body, content_type="application/json"):
    if content_type != "application/json":
        raise ValueError(f"unsupported content type: {content_type}")
    body = json.loads(request_body)
    records = body.get("records") if isinstance(body, dict) else body
    m = body.get("m") if isinstance(body, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError("send a non-empty list of records")
    if len(records) > MAX_RECORDS:
        raise ValueError(f"at most {MAX_RECORDS} records per request")
    return {"records": records, "m": m}


def predict_fn(data, model):
    return model.predict_records(data["records"], m=data["m"])


def output_fn(prediction, accept="application/json"):
    return json.dumps({"predictions": prediction}), "application/json"
