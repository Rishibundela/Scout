import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Automatically locate your root project directory containing the .env file
# This resolves path issues regardless of where the app is executed from
ROOT_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    """
    Application settings and environment variable validation schema.
    Missing required variables will throw an immediate validation error on startup.
    """
    
    # --- REQUIRED CORE VARIABLES ---
    SUPABASE_URL: str
    GOOGLE_API_KEY: str
    
    # --- OPTIONAL / DEFAULTED VARIABLES ---
    # Setting an explicit default value makes the variable optional in the .env file
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_TRACING: str = "false"
    LANGSMITH_PROJECT: str = "scout-agent"
    
    # --- APP CONFIGURATION ---
    APP_ENV: str = "development" # e.g., development, staging, production
    DEBUG_MODE: bool = True

    # Pydantic configuration to load seamlessly from the .env file
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Gracefully ignores extra unrelated variables in your .env
    )

    def configure_tracing(self) -> None:
        """Dynamically toggles LangSmith tracing based on API key presence."""
        if self.LANGSMITH_API_KEY:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGSMITH_API_KEY"] = self.LANGSMITH_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = self.LANGCHAIN_PROJECT
        else:
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            os.environ["LANGSMITH_TRACING"] = "false"


# Instantiate a global singleton settings object to import across your app
settings = Settings()

# Automatically configure background telemetry flags upon initialization
settings.configure_tracing()

