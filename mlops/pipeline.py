"""Amazon SageMaker Pipeline: prepare -> train -> evaluate -> (quality gate) -> register model.

Requires ``pip install -e ".[aws]"`` (SageMaker Python SDK v2) and AWS credentials. Run ``make package``
first so ``build/source`` contains the code that is uploaded to the jobs.

    python mlops/pipeline.py --role-arn arn:aws:iam::123456789012:role/income-imputation-sagemaker \
        --bucket my-bucket --start
"""

from __future__ import annotations

import argparse
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(ROOT, "build", "source")


def build_pipeline(role: str, bucket: str, region: str | None = None, name: str = "income-imputation",
                   package_group: str = "income-imputation-models", input_uri: str | None = None):
    """``input_uri``: S3 prefix with CSV files (real data). ``None`` uses synthetic demo data."""
    from sagemaker.inputs import TrainingInput
    from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
    from sagemaker.sklearn.estimator import SKLearn
    from sagemaker.sklearn.model import SKLearnModel
    from sagemaker.workflow.condition_step import ConditionStep
    from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
    from sagemaker.workflow.functions import JsonGet
    from sagemaker.workflow.model_step import ModelStep
    from sagemaker.workflow.parameters import ParameterFloat, ParameterString
    from sagemaker.workflow.pipeline import Pipeline
    from sagemaker.workflow.pipeline_context import PipelineSession
    from sagemaker.workflow.properties import PropertyFile
    from sagemaker.workflow.steps import ProcessingStep, TrainingStep

    if not os.path.isdir(SOURCE_DIR):
        raise SystemExit("run `make package` first (creates build/source)")

    session = PipelineSession(default_bucket=bucket)
    fw = "1.2-1"
    method = ParameterString("ImputationMethod", default_value="norm")
    min_cov = ParameterFloat("MinCoverage", default_value=0.90)
    instance = ParameterString("InstanceType", default_value="ml.m5.large")

    processor = FrameworkProcessor(
        estimator_cls=SKLearn, framework_version=fw, role=role, instance_type=instance, instance_count=1,
        sagemaker_session=session,
    )
    prep_args = processor.run(
        code="mlops/prepare.py", source_dir=SOURCE_DIR,
        inputs=[ProcessingInput(source=input_uri, destination="/opt/ml/processing/input")] if input_uri else [],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/output/train"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/output/test"),
        ],
    )
    step_prep = ProcessingStep(name="PrepareData", step_args=prep_args)

    estimator = SKLearn(
        entry_point="mlops/train.py", source_dir=SOURCE_DIR, framework_version=fw, role=role,
        instance_type=instance, instance_count=1, sagemaker_session=session,
        hyperparameters={"method": method},
        output_path=f"s3://{bucket}/{name}/models",
    )
    step_train = TrainingStep(
        name="TrainImputationModel",
        step_args=estimator.fit(inputs={"train": TrainingInput(step_prep.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri, content_type="text/csv")}),
    )

    eval_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")
    eval_args = processor.run(
        code="mlops/evaluate_step.py", source_dir=SOURCE_DIR,
        inputs=[
            ProcessingInput(source=step_train.properties.ModelArtifacts.S3ModelArtifacts, destination="/opt/ml/processing/model"),
            ProcessingInput(source=step_prep.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri, destination="/opt/ml/processing/test"),
        ],
        outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
    )
    step_eval = ProcessingStep(name="EvaluateModel", step_args=eval_args, property_files=[eval_report])

    model = SKLearnModel(
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts, role=role,
        entry_point="mlops/inference.py", source_dir=SOURCE_DIR, framework_version=fw, sagemaker_session=session,
    )
    register_args = model.register(
        content_types=["application/json"], response_types=["application/json"],
        inference_instances=["ml.m5.large"], transform_instances=["ml.m5.large"],
        model_package_group_name=package_group, approval_status="PendingManualApproval",
    )
    step_register = ModelStep(name="RegisterModel", step_args=register_args)

    gate = ConditionStep(
        name="QualityGate",
        conditions=[ConditionGreaterThanOrEqualTo(
            left=JsonGet(step_name=step_eval.name, property_file=eval_report, json_path="metrics.min_coverage.value"),
            right=min_cov)],
        if_steps=[step_register], else_steps=[],
    )
    return Pipeline(name=name, parameters=[method, min_cov, instance],
                    steps=[step_prep, step_train, step_eval, gate], sagemaker_session=session)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role-arn", required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--region", default=os.environ.get("AWS_REGION", "eu-west-2"))
    p.add_argument("--input-uri", help="S3 prefix with real CSV data (default: synthetic demo data)")
    p.add_argument("--start", action="store_true", help="start an execution after creating the pipeline")
    a = p.parse_args()
    pipe = build_pipeline(a.role_arn, a.bucket, a.region, input_uri=a.input_uri)
    pipe.upsert(role_arn=a.role_arn)
    print("pipeline upserted:", pipe.name)
    if a.start:
        ex = pipe.start()
        print("execution started:", ex.arn)


if __name__ == "__main__":
    main()
