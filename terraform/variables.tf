variable "project_id" {
  description = "Google Cloud Project ID where the Meal Planning Concierge Agent is provisioned."
  type        = string
}

variable "region" {
  description = "Primary Google Cloud region for Cloud Run, Artifact Registry, and Vertex AI."
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Base name for Cloud Run service, IAM service account, and Artifact Registry."
  type        = string
  default     = "meal-planning-concierge"
}

variable "container_image" {
  description = "Container image URI built via Cloud Build or agents-cli deploy."
  type        = string
  default     = "us-central1-docker.pkg.dev/PROJECT_ID/meal-planning-concierge-repo/agent:latest"
}

variable "fast_model_name" {
  description = "Fast-tier Gemini model used for pantry lookups, grocery parsing, and compaction."
  type        = string
  default     = "gemini-2.5-flash"
}

variable "reasoning_model_name" {
  description = "Reasoning-tier Gemini model used for multi-constraint meal planning & audits."
  type        = string
  default     = "gemini-2.5-pro"
}

variable "min_instances" {
  description = "Minimum Cloud Run warm instances."
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Maximum Cloud Run auto-scaling instances."
  type        = number
  default     = 5
}
