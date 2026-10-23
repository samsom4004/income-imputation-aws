"""Plain-English reports about imputation quality, written by a foundation model on Amazon Bedrock.

Only *aggregate* simulation metrics are ever sent to the model, never survey rows. ``dry_run=True`` (the
default in tests and CI) builds the prompt and a deterministic template report without calling AWS.
"""

from __future__ import annotations

import json
import os

SYSTEM_PROMPT = (
    "You are a careful survey statistician. Explain the results of a missing-data imputation study to a "
    "non-technical reader. Use only the numbers provided. Do not invent results. Explain what bias, "
    "coverage and interval width mean in one sentence each. Recommend one method per estimand and state "
    "the caveats (simulated data, assumed missing-data mechanism). Keep it under 250 words."
)
FORBIDDEN_KEYS = {"rows", "records", "id", "ids", "person", "persons", "data", "wage_values"}
MAX_PROMPT_CHARS = 12000


def assert_aggregate_only(metrics: dict) -> None:
    """Refuse to send anything that looks like row-level data to a language model."""

    def walk(x):
        if isinstance(x, dict):
            bad = FORBIDDEN_KEYS & {str(k).lower() for k in x}
            if bad:
                raise ValueError(f"metrics contain row-level looking keys: {sorted(bad)}")
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(metrics)
    if len(json.dumps(metrics)) > MAX_PROMPT_CHARS:
        raise ValueError("metrics payload is too large to be an aggregate summary")


def build_prompt(metrics: dict) -> str:
    assert_aggregate_only(metrics)
    return "Imputation study results (JSON):\n" + json.dumps(metrics, indent=1) + "\n\nWrite the report now."


def template_report(metrics: dict) -> str:
    """Deterministic fallback report (used for dry runs and when Bedrock is unavailable)."""
    best = metrics.get("best_method_by_estimand", {})
    lines = [
        f"Imputation study: {metrics.get('n_simulations')} simulated runs, "
        f"{metrics.get('missing_rate', 0):.0%} of wages missing ({str(metrics.get('mechanism', '')).upper()} mechanism), "
        f"{metrics.get('imputations_m')} imputations per run.",
        "Best method by quantity of interest (coverage of at least 90% and lowest error):",
    ]
    lines += [f"- {k}: {v}" for k, v in best.items()]
    lines.append("Caveat: results come from simulated data and an assumed missing-data mechanism.")
    return "\n".join(lines)


def explain(metrics: dict, dry_run: bool = False, model_id: str | None = None, region: str | None = None) -> str:
    prompt = build_prompt(metrics)
    if dry_run:
        return template_report(metrics)
    import boto3  # imported lazily so the package works without the AWS extras

    client = boto3.client("bedrock-runtime", region_name=region or os.environ.get("AWS_REGION", "eu-west-2"))
    kwargs = dict(
        modelId=model_id or os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"),
        system=[{"text": SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 600, "temperature": 0.2},
    )
    gid = os.environ.get("BEDROCK_GUARDRAIL_ID")
    if gid:
        kwargs["guardrailConfig"] = {"guardrailIdentifier": gid, "guardrailVersion": os.environ.get("BEDROCK_GUARDRAIL_VERSION", "DRAFT")}
    resp = client.converse(**kwargs)
    return resp["output"]["message"]["content"][0]["text"]
