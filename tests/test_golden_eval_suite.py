"""Tests for Infrastructure & CI/CD: Golden Eval Suite, Secret Manager, and IaC presence."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace

from evals.run_evaluations import run_offline_golden_evaluations
from meal_planner_agent.config import AgentSettings
from meal_planner_agent.secrets import SecretManagerProvider, SecretRetrievalError

REPO_ROOT = Path(__file__).resolve().parent.parent


class _MockSecretManagerClient:
    def access_secret_version(self, request: dict) -> SimpleNamespace:
        name = request["name"]
        if name.endswith("/secrets/meal-planner-gemini-api-key/versions/latest"):
            return SimpleNamespace(payload=SimpleNamespace(data=b"mock-sm-secret-value"))
        raise RuntimeError("Secret not found")


class TestGoldenEvalSuiteAndInfrastructure(unittest.TestCase):
    """Verify automated golden regression suite, Secret Manager resolution, and IaC files."""

    def test_golden_evaluation_dataset_passes_100_percent(self) -> None:
        results = run_offline_golden_evaluations()
        self.assertGreaterEqual(len(results), 5)
        for res in results:
            self.assertTrue(res.passed, f"Golden eval {res.eval_id} failed: {res.failures}")

    def test_secret_manager_prefers_gcp_and_has_no_hardcoded_keys(self) -> None:
        settings = AgentSettings(google_cloud_project="demo-meal-planner-proj")
        provider = SecretManagerProvider(
            settings=settings,
            client=_MockSecretManagerClient(),
        )
        resolved = provider.get_secret("meal-planner-gemini-api-key", required=True)
        self.assertEqual(resolved, "mock-sm-secret-value")

        # Missing required secret raises SecretRetrievalError cleanly
        with self.assertRaises(SecretRetrievalError):
            provider.get_secret("nonexistent-secret", env_fallback="UNSET_ENV_VAR_XYZ", required=True)

        # Env fallback works locally without GCP project
        local_provider = SecretManagerProvider(settings=AgentSettings(google_cloud_project=""))
        os.environ["TEST_LOCAL_FALLBACK_KEY"] = "local-dev-token"
        try:
            self.assertEqual(
                local_provider.get_secret("any-secret", env_fallback="TEST_LOCAL_FALLBACK_KEY"),
                "local-dev-token",
            )
        finally:
            os.environ.pop("TEST_LOCAL_FALLBACK_KEY", None)

    def test_iac_and_cicd_files_exist(self) -> None:
        expected_paths = [
            "Dockerfile",
            "cloudbuild.yaml",
            ".github/workflows/ci.yml",
            "terraform/main.tf",
            "terraform/variables.tf",
            "terraform/iam.tf",
            "terraform/secrets.tf",
            "terraform/outputs.tf",
            "evals/golden_dataset.evalset.json",
            "evals/test_config.json",
        ]
        for rel in expected_paths:
            self.assertTrue((REPO_ROOT / rel).is_file(), f"Missing expected IaC/CI file: {rel}")


if __name__ == "__main__":
    unittest.main()
