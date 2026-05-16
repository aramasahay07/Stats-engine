import json
import os
from typing import Annotated, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central config loaded from environment variables (Railway) and optionally .env (local).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -------------------------
    # Supabase (required in prod)
    # -------------------------
    supabase_url: str = Field("", alias="SUPABASE_URL")
    supabase_service_role_key: str = Field("", alias="SUPABASE_SERVICE_ROLE_KEY")
    supabase_jwt_audience: str = Field("authenticated", alias="SUPABASE_JWT_AUD")

    # -------------------------
    # Postgres (Supabase)
    # -------------------------
    database_url: str = Field("", alias="DATABASE_URL")

    # -------------------------
    # Supabase Storage bucket
    # Railway variable you showed: SUPABASE_STORAGE_BUCKET
    # Also accept older fallback: SUPABASE_BUCKET
    # -------------------------
    supabase_storage_bucket: str = Field("datasets", alias="SUPABASE_STORAGE_BUCKET")
    supabase_bucket_fallback: Optional[str] = Field(None, alias="SUPABASE_BUCKET")

    # -------------------------
    # Backend persistence (local disk / container)
    # -------------------------
    data_dir: str = Field("/data", alias="DATA_DIR")

    # -------------------------
    # OpenAI (optional)
    # -------------------------
    openai_api_key: Optional[str] = Field(None, alias="OPENAI_API_KEY")

    # -------------------------
    # CORS
    # If Railway env sets CORS_ALLOW_ORIGINS as a comma-separated string,
    # Pydantic can parse list formats; simplest is to keep ["*"] by default.
    # -------------------------
    cors_allow_origins: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["*"],
        alias="CORS_ALLOW_ORIGINS",
    )

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _parse_cors_allow_origins(cls, value):
        if value is None or value == "":
            return ["*"]
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return ["*"]
            if raw.startswith("["):
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if str(item).strip()]
                except Exception:
                    pass
            return [item.strip() for item in raw.split(",") if item.strip()]
        return value

    def model_post_init(self, __context) -> None:
        """
        Normalize bucket selection: if SUPABASE_STORAGE_BUCKET is not set but SUPABASE_BUCKET is,
        use the fallback.
        """
        if (not self.supabase_storage_bucket or self.supabase_storage_bucket.strip() == "") and self.supabase_bucket_fallback:
            self.supabase_storage_bucket = self.supabase_bucket_fallback.strip()


settings = Settings()

# Local/MVP only: bypass JWT verification
AUTH_DISABLED = os.getenv("AUTH_DISABLED", "false").lower() == "true"
