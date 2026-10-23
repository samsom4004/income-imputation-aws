# Deploying on AWS

> The commands below were written against the AWS and SageMaker SDK v2 documentation but have **not yet been
> run against a live account**. Treat the first run as a test, in a sandbox account, with the budget alert on.

## Prerequisites

* An AWS account with SageMaker and (optionally) Bedrock model access enabled in your region
* AWS CLI configured (`aws sts get-caller-identity` works), Terraform >= 1.5, Python 3.10+
* `pip install -e ".[aws]"` (installs `boto3` and `sagemaker>=2.200,<3`)

## 1. Infrastructure

```bash
cd infra/terraform
terraform init
terraform apply -var-file=example.tfvars     # copy to your own *.tfvars first; set budget_email
terraform output                              # note: bucket, sagemaker_role_arn
```

Creates a private versioned S3 bucket, a least-privilege SageMaker role, a model package group and an
optional monthly budget alert. For cost-allocation filtering, activate the `Project` tag in Billing.

## 2. Run the pipeline

```bash
make package
python mlops/pipeline.py --role-arn "$ROLE" --bucket "$BUCKET" --start                       # synthetic demo data
python mlops/pipeline.py --role-arn "$ROLE" --bucket "$BUCKET" --input-uri s3://$BUCKET/raw/ --start   # real data
```

Open **SageMaker Studio > Pipelines**, inspect the execution, then approve the model in the Model Registry.

## 3. Deploy and test

```bash
python scripts/deploy_endpoint.py --role-arn "$ROLE" --model-data s3://.../model.tar.gz
python scripts/invoke_endpoint.py
python scripts/delete_endpoint.py          # when you are done
```

## 4. Bedrock report

```bash
export AWS_REGION=eu-west-2 BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0
income-imputation explain --metrics artifacts/metrics.json
```

Optional guardrail: set `BEDROCK_GUARDRAIL_ID` and `BEDROCK_GUARDRAIL_VERSION`. Newer models may require an
inference profile ID instead of a model ID.

## 5. Monitoring

* **CloudWatch** endpoint metrics (invocations, latency, errors) with alarms on error rate and p95 latency.
* **Model Monitor** data-quality baseline: run a baseline job on the training CSV, then schedule hourly or daily
  monitoring on captured requests (enable data capture only if your governance allows it).
* Re-run the simulation study whenever a new survey wave arrives.

## Rough cost (check the AWS pricing pages; figures vary by region and change)

| Item | Typical |
| --- | --- |
| Pipeline run (3 small `ml.m5.large` jobs, minutes each) | well under 1 USD |
| Serverless endpoint | pay per request and per second; scales to zero |
| S3 | pennies for this volume |
| Bedrock report | a fraction of a cent per report on a small model |
| **Forgotten always-on endpoint** | the real risk: run `delete_endpoint.py` |

## Troubleshooting

* `AccessDenied` in a job: check the role has the S3 prefix and the CloudWatch log group.
* `Could not find model_fn`: confirm `make package` ran and `mlops/inference.py` is in `build/source`.
* Bedrock `AccessDeniedException`: request model access in the Bedrock console for your region.
