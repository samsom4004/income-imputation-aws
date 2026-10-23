terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = var.project, ManagedBy = "terraform" }
  }
}

data "aws_caller_identity" "me" {}

locals {
  bucket_name = "${var.project}-${data.aws_caller_identity.me.account_id}-${var.region}"
  tags        = { Project = var.project, ManagedBy = "terraform" }
}

# ---- data and artefact bucket: private, encrypted, versioned ---------------------------------
resource "aws_s3_bucket" "ml" {
  bucket = local.bucket_name
  tags   = local.tags
}

resource "aws_s3_bucket_public_access_block" "ml" {
  bucket                  = aws_s3_bucket.ml.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "ml" {
  bucket = aws_s3_bucket.ml.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "ml" {
  bucket = aws_s3_bucket.ml.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "ml" {
  bucket = aws_s3_bucket.ml.id
  rule {
    id     = "expire-old-pipeline-output"
    status = "Enabled"
    filter {
      prefix = "${var.project}/"
    }
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

# ---- SageMaker execution role (least privilege) ----------------------------------------------
data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker" {
  name               = "${var.project}-sagemaker"
  assume_role_policy = data.aws_iam_policy_document.assume.json
  tags               = local.tags
}

data "aws_iam_policy_document" "permissions" {
  statement {
    sid       = "Bucket"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject"]
    resources = [aws_s3_bucket.ml.arn, "${aws_s3_bucket.ml.arn}/*"]
  }
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
    resources = ["arn:aws:logs:${var.region}:${data.aws_caller_identity.me.account_id}:log-group:/aws/sagemaker/*"]
  }
  statement {
    sid       = "ContainerImages"
    actions   = ["ecr:GetAuthorizationToken", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"]
    resources = ["*"]
  }
  statement {
    sid       = "Metrics"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"]
  }
  statement {
    sid       = "BedrockInvokeOnly"
    actions   = ["bedrock:InvokeModel", "bedrock:Converse"]
    resources = ["arn:aws:bedrock:${var.region}::foundation-model/${var.bedrock_model_id}"]
  }
}

resource "aws_iam_role_policy" "sagemaker" {
  name   = "${var.project}-permissions"
  role   = aws_iam_role.sagemaker.id
  policy = data.aws_iam_policy_document.permissions.json
}

# ---- model registry --------------------------------------------------------------------------
resource "aws_sagemaker_model_package_group" "models" {
  model_package_group_name        = "${var.project}-models"
  model_package_group_description = "Versioned, approval-gated models for ${var.project}"
  tags                            = local.tags
}

# ---- cost guard-rail -------------------------------------------------------------------------
resource "aws_budgets_budget" "monthly" {
  count        = var.budget_email == "" ? 0 : 1
  name         = "${var.project}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"
  cost_filter {
    name   = "TagKeyValue"
    values = [format("user:Project$%s", var.project)]
  }
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_email]
  }
}
