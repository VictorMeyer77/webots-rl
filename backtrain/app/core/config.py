"""
Application configuration management using Pydantic Settings.

This module defines the configuration settings for the FastAPI application,
including environment-specific parameters. Settings can be configured through
environment variables or a .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.

    This class uses Pydantic Settings to manage configuration values, providing
    type validation and default values. Settings can be overridden by setting
    environment variables or creating a .env file in the project root.

    Attributes:
        api_host: The host address where the API server will listen in production mode
                 (when running with `uv run python -m main`). Defaults to "0.0.0.0".
                 Not used with `fastapi dev` command - use --host flag instead.
        api_port: The port number where the API server will listen in production mode
                 (when running with `uv run python -m main`). Defaults to 8000.
                 Not used with `fastapi dev` command - use --port flag instead.
        memory_capacity: The maximum number of items that can be stored in memory
                        buffers (environment, observation, and action memories).
                        Defaults to 100000 if not specified.
        log_console_level: Logging level for console output (e.g., "DEBUG", "INFO",
                          "WARNING", "ERROR", "CRITICAL"). Defaults to "INFO".
        log_console_handler: Enable or disable console logging handler. Defaults to True.
        log_file_level: Logging level for file output. Defaults to "DEBUG".
        log_file_handler: Enable or disable file logging handler. Defaults to False.
        log_file_dir: Directory path where log files will be stored. Defaults to "log".
    """

    model_config = SettingsConfigDict(env_file=".env")

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    memory_capacity: int = 1000

    log_console_level: str = "INFO"
    log_console_handler: bool = True
    log_file_level: str = "DEBUG"
    log_file_handler: bool = False
    log_file_dir: str = "log"


# Global settings instance used throughout the application
settings = Settings()
