"""Secure secret retrieval using Google Cloud Secret Manager with env fallback."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from meal_planner_agent.config import AgentSettings, get_settings


class SecretRetrievalError(RuntimeError):
    """Raised when a required secret cannot be retrieved from any secure backend."""


class SecretManagerProvider:
    """Retrieves runtime credentials from Google Cloud Secret Manager.

    Secrets are never hardcoded in source control. In GCP deployments (Cloud Run /
    Vertex AI Agent Engine), secrets are resolved directly from Google Cloud Secret
    Manager using Application Default Credentials (ADC). For local development or
    CI sandboxes without GCP access, an explicit environment variable fallback is
    supported.
    """

    def __init__(
        self,
        settings: AgentSettings | None = None,
        client: Any | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._cache: dict[str, str] = {}

    def _get_gcp_client(self) -> Any | None:
        if self._client is not None:
            return self._client
        if not self._settings.google_cloud_project:
            return None
        try:
            from google.cloud import secretmanager  # type: ignore[import-untyped]

            self._client = secretmanager.SecretManagerServiceClient()
            return self._client
        except Exception:
            return None

    def get_secret(
        self,
        secret_id: str,
        *,
        env_fallback: str | None = None,
        version: str = "latest",
        required: bool = False,
    ) -> str | None:
        """Fetch a secret value securely by secret ID.

        Args:
            secret_id: The Google Cloud Secret Manager secret name.
            env_fallback: Optional local environment variable name to check when
                Secret Manager is unavailable (e.g., local development).
            version: Secret version identifier, defaults to ``"latest"``.
            required: If ``True``, raises ``SecretRetrievalError`` when missing.

        Returns:
            The resolved secret payload string, or ``None`` if optional and absent.
        """
        cache_key = f"{secret_id}:{version}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 1. Prefer Google Cloud Secret Manager when project & SDK are configured.
        client = self._get_gcp_client()
        if client is not None and self._settings.google_cloud_project:
            resource_name = (
                f"projects/{self._settings.google_cloud_project}"
                f"/secrets/{secret_id}/versions/{version}"
            )
            try:
                response = client.access_secret_version(request={"name": resource_name})
                payload = response.payload.data.decode("utf-8").strip()
                if payload:
                    self._cache[cache_key] = payload
                    return payload
            except Exception:
                # Fall through to local environment fallback for local/dev workflows.
                pass

        # 2. Local / CI environment variable fallback (never hardcoded defaults).
        if env_fallback:
            env_value = os.getenv(env_fallback, "").strip()
            if env_value:
                self._cache[cache_key] = env_value
                return env_value

        if required:
            raise SecretRetrievalError(
                f"Secret '{secret_id}' could not be resolved from Google Cloud Secret Manager "
                f"or environment variable '{env_fallback}'."
            )
        return None

    def hydrate_runtime_credentials(self) -> dict[str, bool]:
        """Inject Gemini API key from Secret Manager into runtime env if not already set.

        Returns:
            Status map indicating whether each credential source is available (never
            returns secret values).
        """
        if not self._settings.use_vertex_ai and not os.getenv("GOOGLE_API_KEY"):
            gemini_key = self.get_secret(
                self._settings.gemini_api_key_secret_id,
                env_fallback="GOOGLE_API_KEY",
                required=False,
            )
            if gemini_key:
                os.environ["GOOGLE_API_KEY"] = gemini_key

        grocery_key = self.get_secret(
            self._settings.grocery_partner_secret_id,
            env_fallback="GROCERY_PARTNER_API_KEY",
            required=False,
        )
        return {
            "gemini_credential_configured": bool(
                self._settings.use_vertex_ai or os.getenv("GOOGLE_API_KEY")
            ),
            "grocery_partner_configured": bool(grocery_key),
        }


@lru_cache(maxsize=1)
def get_secret_provider() -> SecretManagerProvider:
    """Return a singleton SecretManagerProvider instance."""
    return SecretManagerProvider()
