from functools import lru_cache
from pydantic import BaseSettings
import os
from typing import List, Union
from pydantic import validator


class Settings(BaseSettings):
    # General settings
    APP_NAME: str = "XAI Tango"
    ENV: str = "dev"

    # Version
    VERSION: str = "main"

    # API Authentication settings
    API_KEY_NAME: str = "X-API-KEY"
    API_KEYS: List[str] = []  # List of valid API keys

    @validator("API_KEYS", pre=True)
    def assemble_api_keys(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        """
        Validator to parse API_KEYS from .env. Allows CSV or JSON-like formats.
        """
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    class Config:
        # Specify the `.env` file based on ENV value or fallback to `.env`
        env_file = os.getenv("ENV") and f".env.{os.getenv('ENV')}" or ".env"


@lru_cache()
def get_settings():
    """
    Returns a cached instance of Settings.
    """
    return Settings()
