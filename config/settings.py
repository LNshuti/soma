# config/settings.py
from pathlib import Path
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
        case_sensitive=False
    )
    
    # Database settings
    DB_PATH: str = "./data/soma.duckdb"
    
    # DBT settings
    DBT_PROFILES_DIR: str = "./dbt"
    DBT_PROJECT_DIR: str = "./dbt"
    
    # API settings
    API_DEBUG: bool = True
    
    # Web interface settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 5001
    WEB_HOST: str = "0.0.0.0"
    WEB_PORT: int = 7860
    WEB_SHARE: bool = False
    
    # Directory settings
    MODEL_DIR: str = "./artifacts/models"
    LOG_DIR: str = "./artifacts/logs"
    METRICS_DIR: str = "./artifacts/metrics"
    
    # External API keys
    OPENAI_API_KEY: str = "your_openai_key_here"
    
    # Logging settings
    LOG_FORMAT: str = "json"
    LOG_LEVEL: str = "INFO"
    
    # Environment settings
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # Celery settings
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    def model_post_init(self, __context) -> None:
        """Create directories if they don't exist."""
        Path(self.MODEL_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.LOG_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.METRICS_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.DB_PATH).parent.mkdir(parents=True, exist_ok=True)

# Create global settings instance
settings = Settings()

def get_settings():
    """Get settings instance for dependency injection."""
    return settings