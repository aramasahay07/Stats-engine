import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Supabase
    # Defaults allow local import / unit tests without env.
    # Deployment must set these.
    supabase_url: str = Field("", alias="SUPABASE_URL")
    supabase_service_role_key: str = Field("", alias="SUPABASE_SERVICE_ROLE_KEY")
    supabase_jwt_audience: str = Field("authenticated", alias="SUPABASE_JWT_AUD")

    # Postgres (Supabase)
    database_url: str = Field("", alias="DATABASE_URL")  # postgres://...

    # Storage bucket
    bucket_name: str = Field("datasets", alias="SUPABASE_BUCKET")

    # Backend persistence
    data_dir: str = Field("/data", alias="DATA_DIR")

    # OpenAI (optional here; often used in edge functions)
    openai_api_key: str | None = Field(None, alias="OPENAI_API_KEY")

    # CORS
    cors_allow_origins: List[str] = Field(default_factory=lambda: ["*"], alias="CORS_ALLOW_ORIGINS")

settings = Settings()


# Local/MVP only: bypass JWT verification
AUTH_DISABLED = os.getenv('AUTH_DISABLED', 'false').lower() == 'true'
