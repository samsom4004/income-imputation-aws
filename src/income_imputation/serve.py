"""HTTP server that follows the SageMaker "bring your own container" contract.

* ``GET  /ping``         health check (200 when the model is loaded)
* ``POST /invocations``  JSON ``{"records": [{...covariates...}], "m": 5}`` or a bare list of records

Run locally:   ``uvicorn income_imputation.serve:app --port 8080``
In a container SageMaker mounts the model at ``/opt/ml/model`` (override with ``MODEL_DIR``).
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .model import ImputationModel

MAX_RECORDS = 1000
app = FastAPI(title="income-imputation", version="0.1.0")
_state: dict = {"model": None}


def _model() -> ImputationModel:
    if _state["model"] is None:
        path = Path(os.environ.get("MODEL_DIR", "/opt/ml/model")) / "model.joblib"
        if not path.exists():
            raise HTTPException(status_code=503, detail="model not loaded")
        _state["model"] = ImputationModel.load(path)
    return _state["model"]


@app.get("/ping")
def ping():
    _model()
    return JSONResponse({"status": "ok"})


@app.post("/invocations")
async def invocations(request: Request):
    try:
        body = await request.json()
    except Exception as exc:  # malformed JSON
        raise HTTPException(status_code=400, detail="body must be JSON") from exc
    records = body.get("records") if isinstance(body, dict) else body
    m = body.get("m") if isinstance(body, dict) else None
    if not isinstance(records, list) or not records:
        raise HTTPException(status_code=400, detail="send a non-empty list of records")
    if len(records) > MAX_RECORDS:
        raise HTTPException(status_code=413, detail=f"at most {MAX_RECORDS} records per request")
    if m is not None and not (isinstance(m, int) and 1 <= m <= 50):
        raise HTTPException(status_code=400, detail="m must be an integer from 1 to 50")
    try:
        preds = _model().predict_records(records, m=m)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"predictions": preds}
