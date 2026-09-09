import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    LLM_PROVIDER: str = "mistral"

    LLM_MODEL: str = "ministral-3b-latest"
    MISTRAL_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"

    OLLAMA_BASE_URL: str = "http://localhost:11434"

    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 64

    MAX_CONTEXT_TOKENS: int = 2500
    CHARS_PER_TOKEN: int = 4
    MAX_EVIDENCE_PER_GROUP: int = 30

    SQLITE_DB_PATH: str = "data/facts.sqlite"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

os.makedirs(
    os.path.dirname(settings.SQLITE_DB_PATH) or ".",
    exist_ok=True,
)