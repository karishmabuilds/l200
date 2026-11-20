output "cloud_run_service_url" {
  description = "HTTPS endpoint of the deployed Meal Planning Concierge Cloud Run service."
  value       = google_cloud_run_v2_service.meal_planner_service.uri
}

output "runtime_service_account_email" {
  description = "Least-privilege runtime service account email."
  value       = google_service_account.agent_runtime_sa.email
}

output "vertex_ai_search_datastore_id" {
  description = "Provisioned Vertex AI Search datastore ID for culinary knowledge retrieval."
  value       = google_discovery_engine_data_store.culinary_knowledge_store.data_store_id
}

output "secret_ids" {
  description = "Google Cloud Secret Manager secret IDs managed by Terraform."
  value = {
    gemini_api_key          = google_secret_manager_secret.gemini_api_key.secret_id
    grocery_partner_api_key = google_secret_manager_secret.grocery_partner_api_key.secret_id
  }
}
