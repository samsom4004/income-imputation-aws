import pytest

from income_imputation import bedrock_report as br

METRICS = {"mechanism": "mar", "missing_rate": 0.15, "n_simulations": 10, "imputations_m": 5,
           "best_method_by_estimand": {"mean_wage": "norm"}, "results": [{"method": "norm", "coverage": 0.95}]}


def test_dry_run_needs_no_aws():
    text = br.explain(METRICS, dry_run=True)
    assert "mean_wage: norm" in text and "simulated" in text


def test_row_level_data_is_refused():
    with pytest.raises(ValueError, match="row-level"):
        br.build_prompt({"records": [{"wage": 1}]})
    with pytest.raises(ValueError, match="too large"):
        br.build_prompt({"results": ["x" * 20000]})


def test_bedrock_call_is_made_with_converse_and_guardrail(monkeypatch):
    calls = {}

    class Stub:
        def converse(self, **kw):
            calls.update(kw)
            return {"output": {"message": {"content": [{"text": "report"}]}}}

    import boto3

    monkeypatch.setattr(boto3, "client", lambda *a, **k: Stub())
    monkeypatch.setenv("BEDROCK_GUARDRAIL_ID", "gr-123")
    assert br.explain(METRICS, model_id="m-1") == "report"
    assert calls["modelId"] == "m-1" and calls["guardrailConfig"]["guardrailIdentifier"] == "gr-123"
    assert calls["inferenceConfig"]["temperature"] <= 0.3
    assert "records" not in calls["messages"][0]["content"][0]["text"]
