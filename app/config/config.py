import sys
from pathlib import Path

from pydantic import SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, env_file_encoding="utf-8", env_file=".env", extra="ignore")

    user_agent: str = "DesktopAgentCH"
    openrouter_api_key: SecretStr = SecretStr("sk-or-v1-...")
    phoenix_collector_endpoint: str = "http://localhost:6006/v1/traces"
    main_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    summarization_model: str = "arcee-ai/trinity-large-preview:free"
    embedding_model: str = "BAAI/bge-m3"
    hf_hub_offline: str = "1"
    transformers_offline: str = "1"
    max_steps: int = 7

    @computed_field
    @property
    def base_dir(self) -> Path:
        if getattr(sys, "frozen", False):
            return Path(sys.executable).parent
        return Path(__file__).resolve().parents[2]

    @computed_field
    @property
    def db_path(self) -> Path:
        path = self.base_dir / "db" / "collections"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @computed_field
    @property
    def models_path(self) -> Path:
        path = self.base_dir / "models"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @computed_field
    @property
    def pdf_docs_path(self) -> Path:
        path = self.base_dir / "pdf_docs"
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
