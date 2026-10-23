"""Call a deployed endpoint with one example record."""

import argparse
import json

import boto3

EXAMPLE = {"age": 38, "sex": "Female", "health": "No", "hiqual": "Degree", "ethnic": "White",
           "marriage": "Have partner", "jb": "FT employed"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint-name", default="income-imputation")
    p.add_argument("--region", default="eu-west-2")
    a = p.parse_args()
    rt = boto3.client("sagemaker-runtime", region_name=a.region)
    r = rt.invoke_endpoint(EndpointName=a.endpoint_name, ContentType="application/json",
                           Body=json.dumps({"records": [EXAMPLE], "m": 5}))
    print(json.loads(r["Body"].read()))


if __name__ == "__main__":
    main()
