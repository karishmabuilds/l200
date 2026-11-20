terraform {
  required_version = ">= 1.6.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  required_apis = [
    "aiplatform.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudtrace.googleapis.com",
    "discoveryengine.googleapis.com",
    "dlp.googleapis.com",
    "logging.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
  ]
}

resource "google_project_service" "enabled_apis" {
  for_each           = toset(local.required_apis)
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "agent_images" {
  location      = var.region
  repository_id = "${var.service_name}-repo"
  description   = "Container images for the Meal Planning Concierge ADK Agent"
  format        = "DOCKER"

  depends_on = [google_project_service.enabled_apis]
}

resource "google_discovery_engine_data_store" "culinary_knowledge_store" {
  location                    = "global"
  data_store_id               = "${var.service_name}-recipes-store"
  display_name                = "Meal Planner Culinary & Nutrition Store"
  industry_vertical           = "GENERIC"
  content_config              = "CONTENT_REQUIRED"
  solution_types              = ["SOLUTION_TYPE_SEARCH"]
  create_advanced_site_search = false

  depends_on = [google_project_service.enabled_apis]
}

resource "google_cloud_run_v2_service" "meal_planner_service" {
  name     = var.service_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.agent_runtime_sa.email

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    containers {
      image = var.container_image

      ports {
        container_port = 8080
      }

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }

      env {
        name  = "GOOGLE_CLOUD_LOCATION"
        value = var.region
      }

      env {
        name  = "GOOGLE_GENAI_USE_VERTEXAI"
        value = "true"
      }

      env {
        name  = "FAST_MODEL_NAME"
        value = var.fast_model_name
      }

      env {
        name  = "REASONING_MODEL_NAME"
        value = var.reasoning_model_name
      }

      env {
        name  = "VERTEX_AI_SEARCH_DATASTORE_ID"
        value = google_discovery_engine_data_store.culinary_knowledge_store.data_store_id
      }

      env {
        name  = "ENABLE_CLOUD_TRACE"
        value = "true"
      }

      env {
        name  = "ENABLE_CLOUD_DLP_REDACTION"
        value = "true"
      }

      env {
        name  = "GEMINI_API_KEY_SECRET_ID"
        value = google_secret_manager_secret.gemini_api_key.secret_id
      }

      env {
        name = "GROCERY_PARTNER_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.grocery_partner_api_key.secret_id
            version = "latest"
          }
        }
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }
    }
  }

  depends_on = [
    google_project_service.enabled_apis,
    google_project_iam_member.agent_runtime_roles,
  ]
}
