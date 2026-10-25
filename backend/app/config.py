"""Environment-based application settings."""
from functools import lru_cache
import os
from pathlib import Path
from urllib.parse import quote

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # Docker and local installs include python-dotenv via requirements.
    pass

class Settings:
    """Small settings object; replace with pydantic-settings for richer validation."""
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./cbir.db")
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data")).resolve()
    public_base_url: str = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")
    cors_origins: list[str] = os.getenv("API_CORS_ORIGINS", "http://localhost:5173").split(",")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

@lru_cache
def get_settings() -> Settings:
    return Settings()

def public_image_url(relative_path: str) -> str:
    """Build a public URL from a path stored relative to DATA_DIR/images."""
    encoded_path = "/".join(quote(part) for part in Path(relative_path).as_posix().split("/"))
    return f"{get_settings().public_base_url}/static/images/{encoded_path}"
