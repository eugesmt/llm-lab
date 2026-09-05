"""Application configuration via Pydantic Settings.

API keys come only from environment variables (optionally a local ``.env``
file, which is gitignored) and are stored as ``SecretStr`` so they never leak
into logs or repr output.
"""

from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    deepseek_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    # Reserved for a future Anthropic provider adapter; not used yet.
    anthropic_api_key: SecretStr | None = None

    # Single timeout for the whole chain: HTTP request -> application context
    # -> provider call (provider SDK clients receive the same value).
    request_timeout_s: float = 120.0

    host: str = "127.0.0.1"
    port: int = 8000
