from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application configuration."""

    # Required
    GOOGLE_API_KEY: str
    SUPABASE_URL: str
    DATABASE_URI: str

    # Optional
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str = "scout"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

if __name__ == "__main__":
    # Print the settings for debugging purposes
    print("Loaded settings:")
    print(f"GOOGLE_API_KEY: {settings.GOOGLE_API_KEY}")
    print(f"SUPABASE_URL: {settings.SUPABASE_URL}")
    print(f"DATABASE_URI: {settings.DATABASE_URI}")
    print(f"LANGSMITH_API_KEY: {settings.LANGSMITH_API_KEY}")
    print(f"LANGSMITH_PROJECT: {settings.LANGSMITH_PROJECT}")