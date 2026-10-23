"""Delete an endpoint, its config and model so nothing keeps costing money."""

import argparse

import boto3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint-name", default="income-imputation")
    p.add_argument("--region", default="eu-west-2")
    a = p.parse_args()
    sm = boto3.client("sagemaker", region_name=a.region)
    cfg = sm.describe_endpoint(EndpointName=a.endpoint_name)["EndpointConfigName"]
    model = sm.describe_endpoint_config(EndpointConfigName=cfg)["ProductionVariants"][0]["ModelName"]
    sm.delete_endpoint(EndpointName=a.endpoint_name)
    sm.delete_endpoint_config(EndpointConfigName=cfg)
    sm.delete_model(ModelName=model)
    print("deleted", a.endpoint_name)


if __name__ == "__main__":
    main()
