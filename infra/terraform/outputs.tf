output "bucket" {
  value = aws_s3_bucket.ml.bucket
}

output "sagemaker_role_arn" {
  value = aws_iam_role.sagemaker.arn
}

output "model_package_group" {
  value = aws_sagemaker_model_package_group.models.model_package_group_name
}
