variable "region" {
  description = "AWS region (eu-west-2 = London)"
  type        = string
  default     = "eu-west-2"
}

variable "project" {
  description = "Resource name prefix"
  type        = string
  default     = "income-imputation"
}

variable "bedrock_model_id" {
  description = "Bedrock foundation model the role may invoke (must be enabled in your account/region)"
  type        = string
  default     = "anthropic.claude-3-haiku-20240307-v1:0"
}

variable "monthly_budget_usd" {
  type    = number
  default = 25
}

variable "budget_email" {
  description = "Email for budget alerts (leave empty to skip the budget)"
  type        = string
  default     = ""
}
