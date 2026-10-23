# Data governance

* **Licence.** UKHLS / Understanding Society data is provided by the UK Data Service under an end-user licence.
  Do not commit it, share it, or upload it anywhere not covered by your licence. This repo contains only
  synthetic data and code.
* **Personal data.** Even a "wage-only" extract can be personal data. Treat all real data as confidential;
  involve your data protection officer before using a cloud region or service.
* **Region.** Defaults to `eu-west-2` (London). Change deliberately.
* **Access.** Least-privilege IAM role (see `infra/terraform/main.tf`); the Bedrock permission is limited to
  invoking one named model.
* **Encryption.** S3 encrypted at rest (AES-256; switch to a KMS key for stricter control); TLS in transit.
* **Artefacts.** `model.joblib` for donor methods contains individual wages; keep these in a private
  bucket and never publish them. `.gitignore` blocks `*.joblib`, `*.csv` and `*.tar.gz`.
* **Logging.** Do not log request bodies. The server returns only the imputed values.
* **Language models.** Only aggregate metrics go to Bedrock. Check your Bedrock model-access terms and
  whether prompts are retained for your account.
* **Retention.** Pipeline outputs expire after 30 days of non-current versions (lifecycle rule).
