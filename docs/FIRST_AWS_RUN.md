# Your first run on AWS: step by step

This guide assumes **no AWS experience**. Follow the parts in order. Every part says how to check it worked.

> **Honest warning.** The code in this repository is tested on a laptop, and the Terraform files pass
> `terraform validate`, but nobody has run the SageMaker pipeline on a real AWS account yet. You may hit a
> permission or configuration error. That is normal. Each part has a "If it fails" box, and you can send me
> the exact error text and I will fix the code.
>
> **Never paste passwords, secret keys or one-time codes into a chat, an issue or a file in git.**

Time: about 2 to 3 hours the first time. Cost: small if you follow the cleanup part (Part 11), but you are
responsible for what you create. Keep the budget alert from Part 2 switched on.

Words you will meet:

| Word | Meaning |
| --- | --- |
| **Region** | the AWS data-centre location. We use **Europe (London) = `eu-west-2`**. Always check the top-right of the console. |
| **IAM** | the service that controls who can do what |
| **S3** | file storage ("bucket" = a folder in the cloud) |
| **SageMaker** | the machine-learning service |
| **Bedrock** | the service that runs large language models |
| **Terraform** | a tool that creates AWS things from text files, so you can create and delete them in one command |
| **Profile** | a saved login on your laptop, used by the AWS command line |

---

## Part 1. Create and protect your AWS account (15 minutes)

1. Go to https://aws.amazon.com and choose **Create an AWS account**. Use an email you control, choose
   **Personal** (or whatever matches how you will use it), add a card, and verify your phone.
2. Sign in as the **root user** (the email you just used) at https://console.aws.amazon.com.
3. **Switch on MFA for the root user** (this is the most important security step):
   click your account name (top right) > **Security credentials** > **Multi-factor authentication (MFA)** >
   **Assign MFA device**. Use an authenticator app on your phone. Save the recovery information somewhere safe.
4. Top right, click the region name and choose **Europe (London) eu-west-2**.

Check: you can sign in with email + password + MFA code, and the region says **London**.

> Do not use the root user for daily work and never create access keys for it.

## Part 2. Set a budget alert (10 minutes)

1. In the console search box type **Billing and Cost Management** and open it.
2. Left menu > **Budgets** > **Create budget** > **Use a template (simplified)** > **Monthly cost budget**.
3. Amount: **25** (US dollars). Add your email. Create.
4. Optional but wise: Billing home > **Billing preferences** > tick **Receive CloudWatch billing alerts** and
   **Alert preferences / Free tier usage alerts**.

Check: you get a confirmation email and the budget shows in the list.

> Terraform in Part 5 can create a second budget for you. Having this first one in the console means you are
> protected even if something goes wrong.

## Part 3. Create a safe everyday login (20 minutes)

The best practice is **IAM Identity Center** (temporary logins, no secret keys on your laptop).

1. Console search **IAM Identity Center** > **Enable**. (If it offers an "organisation instance", accept it.)
2. Left menu **Users** > **Add user**: give your name and email. You will get an email to set a password and
   MFA. Do that.
3. Left menu **Permission sets** > **Create permission set** > **Predefined** >
   **AdministratorAccess**. (Fine for a personal learning account. Never do this in a work account without permission.)
4. Left menu **AWS accounts** > tick your account > **Assign users or groups** > choose your user and the
   permission set > Submit.
5. On the Identity Center **Dashboard**, copy the **AWS access portal URL** (looks like
   `https://d-xxxxxxxxxx.awsapps.com/start`). You will need it in Part 4.

Check: opening the portal URL shows your account and a **Management console** button.

> **If this feels too long:** use the simpler but less safe route in [AWS_LOGIN_IAM_USER.md](AWS_LOGIN_IAM_USER.md)
> (IAM user with an access key). It replaces Part 3 and Part 5. Never share the key with anyone.

## Part 4. Install the tools on your Mac (30 minutes)

Open the **Terminal** app. Type each command and press Return.

1. **Xcode command-line tools licence** (your Mac asked for this earlier; it blocks `make` and `git`):

   ```bash
   sudo xcodebuild -license
   ```
   Press space to scroll, type `agree`.

2. **Homebrew** (a package installer). Check if you have it:

   ```bash
   brew --version
   ```
   If you see "command not found", install it from https://brew.sh (copy the one-line command from that page).

3. **AWS command line, Terraform and the GitHub CLI**:

   ```bash
   brew install awscli
   brew tap hashicorp/tap
   brew install hashicorp/tap/terraform
   aws --version
   terraform version
   ```
   Check: both print a version number.

4. **Get the code** (it is already on your Mac in `~/Documents/06 Code & Projects/GitHub/ml-aws-portfolio/`):

   ```bash
   cd "$HOME/Documents/06 Code & Projects/GitHub/ml-aws-portfolio/income-imputation-aws"
   ```

5. **Python environment**:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install "numpy==1.26.4" "scipy==1.11.4" "scikit-learn==1.3.2" "pandas==2.0.3"
   pip install -e ".[aws,serve,dev]"
   ```
   (The first line pins versions that work on your Mac; a newer scipy failed to load on it.)

6. **Run the tests once** to prove your computer is ready:

   ```bash
   pytest
   ```
   Check: `56 passed`.

## Part 5. Log in to AWS from the terminal (10 minutes)

```bash
aws configure sso
```

Answer the questions:

* **SSO session name:** `bp`
* **SSO start URL:** the access portal URL from Part 3
* **SSO region:** the region where you enabled Identity Center (probably `eu-west-2`)
* **SSO registration scopes:** press Return
* A browser opens: sign in and click **Allow**.
* Pick your account and the **AdministratorAccess** role.
* **Default client region:** `eu-west-2`. **Output format:** `json`. **Profile name:** `bp`.

Then:

```bash
aws sso login --profile bp
export AWS_PROFILE=bp
export AWS_REGION=eu-west-2
aws sts get-caller-identity
```

Check: it prints your 12-digit account number. Write the account number down; you will recognise it later.

> Your login lasts a few hours. When commands say "token expired", run `aws sso login --profile bp` again.
> You must run the two `export` lines in every new Terminal window.

## Part 6. Check SageMaker is allowed to start machines (5 minutes)

New AWS accounts sometimes start with a limit of 0 for some machine types.

1. Console search **Service Quotas** > **AWS services** > **Amazon SageMaker**.
2. Search `ml.m5.large`. Check these have a value of at least **1**:
   * `ml.m5.large for processing job usage`
   * `ml.m5.large for training job usage`
   * `ml.m5.large for endpoint usage` (only if you will use a real-time endpoint; serverless does not need it)
3. If a value is 0, choose it > **Request increase at account level** > value 1 or 2. Approval can take a few hours.

## Part 7. Create the AWS building blocks with Terraform (15 minutes)

```bash
cd infra/terraform
cp example.tfvars my.tfvars
```

Open `my.tfvars` in a text editor and set `budget_email = "your email"`. Keep `force_destroy = true` (sandbox only).
`my.tfvars` is ignored by git so your email is not published.

```bash
terraform init
terraform plan -var-file=my.tfvars
```

Read the plan. It should say **Plan: 9 to add, 0 to change, 0 to destroy (7 if you left `budget_email` empty: no budget is created)**: an S3 bucket (and its
safety settings), an IAM role and policy, a model registry group and a budget.

```bash
terraform apply -var-file=my.tfvars
```

Type `yes` when asked. When it finishes, save the outputs:

```bash
terraform output
export BUCKET=$(terraform output -raw bucket)
export ROLE=$(terraform output -raw sagemaker_role_arn)
echo $BUCKET $ROLE
cd ../..
```

Check: `echo` prints a bucket name like `income-imputation-123456789012-eu-west-2` and a role like
`arn:aws:iam::123456789012:role/income-imputation-sagemaker`. In the console, S3 shows the bucket.

> **If it fails:** the error names the problem. "ExpiredToken" = log in again (Part 5). "BucketAlreadyExists" =
> change `project` in `my.tfvars`. "AccessDenied" = your login lacks permissions (check Part 3). Send me the text.

## Part 8. Run the machine-learning pipeline (20 minutes + waiting)

1. Package the code that SageMaker will use (avoid `make` if the Xcode licence is not accepted):

   ```bash
   bash scripts/package_source.sh
   ```
   Check: it prints `packaged build/source`.

2. Create and start the pipeline. With no `--input-uri` it uses **synthetic demo data**, which is the right
   first run:

   ```bash
   python mlops/pipeline.py --role-arn "$ROLE" --bucket "$BUCKET" --start
   ```
   Check: it prints `pipeline upserted: income-imputation` then `execution started: arn:aws:sagemaker:...`.

3. Watch progress (repeat every minute or two; the whole run takes roughly 10 to 20 minutes):

   ```bash
   aws sagemaker list-pipeline-executions --pipeline-name income-imputation --query 'PipelineExecutionSummaries[0]'
   ```
   Then, using the execution ARN printed above:

   ```bash
   EXEC=$(aws sagemaker list-pipeline-executions --pipeline-name income-imputation --query 'PipelineExecutionSummaries[0].PipelineExecutionArn' --output text)
   aws sagemaker list-pipeline-execution-steps --pipeline-execution-arn "$EXEC" --query 'PipelineExecutionSteps[].[StepName,StepStatus,FailureReason]' --output table
   ```
   You should see **PrepareData**, **TrainImputationModel**, **EvaluateModel**, then **QualityGate**, and
   (if the gate passes) **RegisterModel**, each moving from `Executing` to `Succeeded`.

   You can also look in the console: search **Amazon SageMaker AI** > **Pipelines** (it may ask you to create a
   "domain" for Studio; you can skip that and use the commands above).

4. **If a step shows `Failed`**: read the reason, and the job's log:

   ```bash
   aws sagemaker list-pipeline-execution-steps --pipeline-execution-arn "$EXEC" --query 'PipelineExecutionSteps[?StepStatus==`Failed`].FailureReason' --output text
   ```
   Logs are in the console: **CloudWatch** > **Log groups** > `/aws/sagemaker/ProcessingJobs` or
   `/aws/sagemaker/TrainingJobs` > the newest stream. Copy the last 20 lines and send them to me.
   Common first-run causes: a missing permission, a machine limit of 0 (Part 6), or a package version.

## Part 9. Approve the model (5 minutes)

The pipeline only *registers* a model as "PendingManualApproval". A person approves it.

```bash
aws sagemaker list-model-packages --model-package-group-name income-imputation-models --query 'ModelPackageSummaryList[0].[ModelPackageArn,ModelApprovalStatus]' --output text
```

If it shows an ARN, first look at the quality numbers. The simulation results are in the training job's output
in S3 (`s3://$BUCKET/income-imputation/models/...`). Then approve:

```bash
PKG=$(aws sagemaker list-model-packages --model-package-group-name income-imputation-models --query 'ModelPackageSummaryList[0].ModelPackageArn' --output text)
aws sagemaker update-model-package --model-package-arn "$PKG" --model-approval-status Approved
MODEL_DATA=$(aws sagemaker describe-model-package --model-package-name "$PKG" --query 'InferenceSpecification.Containers[0].ModelDataUrl' --output text)
echo $MODEL_DATA
```

Check: `echo` prints an `s3://.../model.tar.gz` address.

> If nothing is registered, the quality gate stopped it (coverage below 90%). That is the gate working. Check
> the `EvaluateModel` step output and tell me what you see.

## Part 10. Deploy the endpoint and try it (15 minutes)

```bash
python scripts/deploy_endpoint.py --role-arn "$ROLE" --model-data "$MODEL_DATA"
```

This creates a **serverless** endpoint (no machine to pay for while idle). It takes several minutes. Then:

```bash
python scripts/invoke_endpoint.py
```

Check: you get a JSON answer with `wage_imputed`, `wage_p05`, `wage_p95`, `was_imputed`.

Optional report written by Amazon Bedrock (use this last because it needs extra access):

1. Console search **Amazon Bedrock** > **Model access** (or **Model catalog**) and request or enable
   access to an Anthropic **Claude** model in your region. Some accounts must submit a short use-case form first.
2. Then:

   ```bash
   income-imputation simulate --n 3000 --n-sim 10 --out artifacts/metrics.json
   export BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0
   income-imputation explain --metrics artifacts/metrics.json
   ```
   If the model is not available in London, Bedrock will say so. You may need a different model ID or an
   "inference profile" ID shown in the Bedrock console. Use `--dry-run` to test without AWS.

## Part 11. Clean up so nothing keeps costing money (10 minutes)

Do this **every time** you finish.

1. Delete the endpoint (the only thing that can quietly cost money):

   ```bash
   python scripts/delete_endpoint.py
   ```
   Check: `aws sagemaker list-endpoints` shows no endpoints.

2. Delete the registered models, then everything Terraform made:

   ```bash
   for p in $(aws sagemaker list-model-packages --model-package-group-name income-imputation-models --query 'ModelPackageSummaryList[].ModelPackageArn' --output text); do aws sagemaker delete-model-package --model-package-name "$p"; done
   aws sagemaker delete-pipeline --pipeline-name income-imputation --client-request-token "$(uuidgen)"
   cd infra/terraform && terraform destroy -var-file=my.tfvars && cd ../..
   ```
   Type `yes`. Check: the S3 bucket is gone from the console.

3. Look at **Billing > Bills** the next day. It should show tiny amounts and nothing growing.

## Part 12. When something goes wrong

| Message | What it usually means | What to do |
| --- | --- | --- |
| `ExpiredToken` / `token expired` | your login timed out | `aws sso login --profile bp`, then re-run the two `export` lines |
| `AccessDenied` / `not authorized` | a missing permission | copy the full message to me; it names the action and resource |
| `ResourceLimitExceeded` | the machine limit is 0 | Part 6 |
| `ValidationException` in the pipeline | a parameter or version problem | copy the full message |
| Bedrock `AccessDeniedException` | model access not enabled | Part 10, Bedrock steps |
| A step is stuck `Executing` for more than 30 minutes | a job waiting for a machine or downloading | check CloudWatch logs, then stop the execution and retry |

To stop a running pipeline: `aws sagemaker stop-pipeline-execution --pipeline-execution-arn "$EXEC"`.

## What to send me when you get stuck

1. Which part and step number.
2. The **exact** error text (copy and paste).
3. The output of `aws sts get-caller-identity` with the account number removed.

Do **not** send passwords, MFA codes, access keys or the Identity Center login link.

## Afterwards

* Repeat the same steps for `abnormality-detection-aws` (it also needs a GPU quota: `ml.g4dn.xlarge for
  training job usage`, and the Kvasir-Capsule data uploaded to S3).
* Everything in `docs/ML_STRATEGY.md` and `docs/PLATFORM_STRATEGY.md` explains why each piece exists.
