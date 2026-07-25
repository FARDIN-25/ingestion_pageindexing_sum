from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str

    # PageIndex cloud API
    PAGEINDEX_API_KEY: str
    PAGEINDEX_API_BASE: str = "https://api.pageindex.ai"

    # AWS / S3 batch ingestion (code restored; endpoint currently commented / disabled)
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str = ""
    S3_CLEANED_PREFIX: str = "cleaned/"
    INGESTION_MAX_CONCURRENT_DOCS: int = 5

    # Embedding model (local)
    EMBEDDING_MODEL_NAME: str = "Qwen/Qwen3-Embedding-0.6B"
    EMBEDDING_DIM: int = 1024
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_DEVICE: str = "cpu"  # cpu | cuda

    # Vector service chunking (book path; independent of PageIndex)
    VECTOR_CHUNK_SIZE: int = 1000
    VECTOR_CHUNK_OVERLAP: int = 200

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
