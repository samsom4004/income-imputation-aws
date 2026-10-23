"""Deploy the latest approved model package (or a model.tar.gz in S3) as a *serverless* SageMaker endpoint.

Serverless inference scales to zero, so you only pay while requests are being served.

    python scripts/deploy_endpoint.py --role-arn ... --model-data s3://bucket/path/model.tar.gz
"""

import argparse
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role-arn", required=True)
    p.add_argument("--model-data", required=True)
    p.add_argument("--endpoint-name", default="income-imputation")
    p.add_argument("--memory-mb", type=int, default=2048)
    p.add_argument("--max-concurrency", type=int, default=5)
    a = p.parse_args()

    from sagemaker.serverless import ServerlessInferenceConfig
    from sagemaker.sklearn.model import SKLearnModel

    model = SKLearnModel(model_data=a.model_data, role=a.role_arn, entry_point="mlops/inference.py",
                         source_dir=os.path.join(ROOT, "build", "source"), framework_version="1.2-1")
    model.deploy(endpoint_name=a.endpoint_name,
                 serverless_inference_config=ServerlessInferenceConfig(memory_size_in_mb=a.memory_mb, max_concurrency=a.max_concurrency))
    print("endpoint ready:", a.endpoint_name)


if __name__ == "__main__":
    main()
