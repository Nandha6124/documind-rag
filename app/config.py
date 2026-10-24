from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-2.0-flash"
    gemini_embed_model: str = "gemini-embedding-001"
    embed_dim: int = 768
    database_url: str = "sqlite:///./documind.db"
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 4
    max_upload_mb: int = 10


settings = Settings()
