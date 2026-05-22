from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str
    
    # AWS / S3
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str
    S3_CLEANED_PREFIX: str = "cleaned/"
    
    # Ingestion Config
    INGESTION_MAX_CONCURRENT_DOCS: int = 5
    PAGEINDEX_API_KEY: str
    PAGEINDEX_API_BASE: str = "https://api.pageindex.ai"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
