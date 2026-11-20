resource "google_service_account" "agent_runtime_sa" {
  account_id   = "${var.service_name}-sa"
  display_name = "Meal Planning Concierge Agent Runtime Service Account"
  description  = "Least-privilege service account for Vertex AI, Secret Manager, Cloud DLP, and Trace."
}

locals {
  agent_iam_roles = [
    "roles/aiplatform.user",
    "roles/cloudtrace.agent",
    "roles/discoveryengine.viewer",
    "roles/dlp.user",
    "roles/logging.logWriter",
    "roles/secretmanager.secretAccessor",
  ]
}

resource "google_project_iam_member" "agent_runtime_roles" {
  for_each = toset(local.agent_iam_roles)
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.agent_runtime_sa.email}"
}
