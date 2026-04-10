from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, env_file_encoding="utf-8", env_file=".env")

    openrouter_api_key: SecretStr = SecretStr("sk-or-v1-...")
    phoenix_collector_endpoint: str = "http://localhost:6006/v1/traces"
    main_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    summarization_model: str = "arcee-ai/trinity-large-preview:free"
    embedding_model: str = "BAAI/bge-m3"


settings = Settings()
