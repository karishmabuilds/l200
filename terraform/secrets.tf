resource "google_secret_manager_secret" "gemini_api_key" {
  secret_id = "meal-planner-gemini-api-key"

  replication {
    auto {}
  }

  labels = {
    app        = var.service_name
    managed_by = "terraform"
  }

  depends_on = [google_project_service.enabled_apis]
}

resource "google_secret_manager_secret" "grocery_partner_api_key" {
  secret_id = "meal-planner-grocery-partner-key"

  replication {
    auto {}
  }

  labels = {
    app        = var.service_name
    managed_by = "terraform"
  }

  depends_on = [google_project_service.enabled_apis]
}
