"""
Configuration management module for CORL (Control and Reinforcement Learning).

This module provides a flexible configuration system that loads settings from
environment variables and .env files with type validation and caching support.
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Type, Any

from dotenv import load_dotenv


class LogLevel(str, Enum):
    """
    Valid logging levels for Python's logging module.

    Inherits from both str and Enum to allow string comparisons
    while maintaining type safety for configuration validation.

    Attributes:
        DEBUG: Detailed information, typically of interest only when diagnosing problems
        INFO: Confirmation that things are working as expected
        WARNING: An indication that something unexpected happened
        ERROR: A more serious problem, the software has not been able to perform some function
        CRITICAL: A serious error, indicating that the program itself may be unable to continue
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ConfigItem:
    """
    Configuration item with default value, type casting, and description.

    This dataclass defines the schema for a single configuration item,
    including its default value, the type it should be cast to, and
    a human-readable description for documentation purposes.

    Attributes:
        default: The default value to use if no environment variable is set.
                Can be None for optional configuration items.
        cast: The type to cast the environment variable value to.
              Supported types: str, int, float, bool, LogLevel, or None for no casting.
        description: A human-readable description of what this configuration item controls.
                    Used for documentation and error messages.
    """

    default: Any | None
    cast: Type | None = None
    description: str = ""


class Config:
    """
    Configuration class that loads settings from .env file and environment variables.

    This class provides a centralized way to manage application configuration by:
    - Loading values from a .env file using python-dotenv
    - Reading environment variables with a configurable prefix
    - Type casting values to appropriate types
    - Providing default values for all configuration items
    - Caching values for improved performance
    - Supporting dictionary-style access via [] operator

    Environment variables are expected to follow the naming convention:
    {prefix}_{KEY_NAME} (e.g., WEBOTS_API_HOST, WEBOTS_API_PORT)

    Attributes:
        DEFAULTS: Dictionary mapping configuration keys to ConfigItem objects
        prefix: Prefix for environment variable names (default: "webots")
        _cache: Internal cache for storing parsed configuration values

    Example:
        >>> config = Config(prefix="myapp")
        >>>
        >>> # Method 1: Using get()
        >>> host = config.get("API_HOST")
        >>>
        >>> # Method 2: Dictionary-style access
        >>> port = config["API_PORT"]
        >>>
        >>> # Check if a key exists
        >>> if "API_HOST" in config:
        ...     print("API_HOST is configured")
    """

    DEFAULTS = {
        # API Configuration
        "API_HOST": ConfigItem(
            default="http://localhost",
            cast=str,
            description="Backend training server API host URL",
        ),
        "API_PORT": ConfigItem(
            default=8000,
            cast=int,
            description="Backend training server API port number",
        ),
        # Console Logging Configuration
        "LOG_CONSOLE_LEVEL": ConfigItem(
            default="INFO",
            cast=LogLevel,
            description="Console log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
        "LOG_CONSOLE_HANDLER": ConfigItem(
            default=True, cast=bool, description="Enable console logging handler"
        ),
        # File Logging Configuration
        "LOG_FILE_LEVEL": ConfigItem(
            default="INFO",
            cast=LogLevel,
            description="File log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
        "LOG_FILE_HANDLER": ConfigItem(
            default=False, cast=bool, description="Enable file logging handler"
        ),
        "LOG_FILE_DIR": ConfigItem(
            default=".log", cast=str, description="Directory path for log files"
        ),
        # Trainer Configuration
        "TRAIN": ConfigItem(
            default=False,
            cast=bool,
            description="Set to True to enable training mode, False for evaluation mode",
        ),
        "TRAIN_ID": ConfigItem(
            default=None,
            cast=str,
            description="Unique identifier for the training session",
        ),
        "WORKER_ID": ConfigItem(
            default=None,
            cast=int,
            description="Unique identifier for the worker instance",
        ),
        "TRAINER_TENSORBOARD_PATH": ConfigItem(
            default=".train/tensorboard/",
            cast=str,
            description="Path to TensorBoard logs directory",
        ),
        "TRAINER_MODEL_DIR": ConfigItem(
            default=".train/models/",
            cast=str,
            description="Directory path for saved models",
        ),
        "TRAINER_MAX_WORKER": ConfigItem(
            default=4, cast=int, description="Maximum number of training workers"
        ),
        # Webots environment
        "BIN_PATH": ConfigItem(
            default="/Applications/Webots.app/Contents/MacOS/webots",
            cast=str,
            description="Path to Webots binary executable",
        ),
        "WORLD_PATH": ConfigItem(
            default="projects/worlds",
            cast=str,
            description="Path to Webots world file to run",
        ),
        "CONTROLLER_PATH": ConfigItem(
            default="projects/controllers",
            cast=str,
            description="Path to Webots controller files",
        ),
    }

    def __init__(self, prefix: str = "webots"):
        """
        Initialize configuration with optional prefix and .env file.

        Loads environment variables from a .env file (if present) and initializes
        the configuration cache. Environment variables are expected to be prefixed
        with the provided prefix string.

        Args:
            prefix: Prefix for environment variable names. All environment variables
                   should follow the pattern: {prefix}_{KEY_NAME} (case-insensitive).
                   Defaults to "webots".
        """
        self.prefix = prefix
        self._cache: dict[str, Any] = {}
        load_dotenv()

    def get(self, key: str) -> Any:
        """
        Get a configuration value by key.

        Retrieves a configuration value from environment variables or returns the default
        value if not set. Values are automatically cast to the appropriate type and cached
        for improved performance on subsequent calls.

        The method performs the following steps:
        1. Checks the cache for a previously retrieved value
        2. Validates that the key exists in DEFAULTS
        3. Looks for an environment variable named {prefix}_{KEY}
        4. Returns the default value if no environment variable is found
        5. Casts the value to the appropriate type if specified
        6. Caches the result for future calls

        Args:
            key: Configuration key (case-insensitive). Must exist in the DEFAULTS dictionary.

        Returns:
            The configuration value, cast to the appropriate type as defined in DEFAULTS.
            Returns the default value if no environment variable is set.

        Raises:
            ValueError: If the key is not defined in DEFAULTS, or if type casting fails.
                       Error messages include the default value, description, and original error
                       to aid in debugging.
        """
        key = key.upper()

        # Return cached value if available
        if key in self._cache:
            return self._cache[key]

        if key not in self.DEFAULTS:
            raise ValueError(f"Configuration key '{key}' is not defined in DEFAULTS")

        config_item = self.DEFAULTS[key]
        env_key = f"{self.prefix}_{key}".upper()
        value = os.getenv(env_key)

        if value is None:
            result = config_item.default
            self._cache[key] = result
            return result

        if config_item.cast is None:
            self._cache[key] = value
            return value

        if config_item.cast is bool:
            result = value.lower() in ("true", "1", "yes", "on")
            self._cache[key] = result
            return result

        try:
            result = config_item.cast(value)
            self._cache[key] = result
            return result
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Failed to cast '{env_key}={value}' to type {config_item.cast.__name__}. "
                f"Default value: {config_item.default}. "
                f"Description: {config_item.description}. "
                f"Error: {str(e)}"
            )

    def __getitem__(self, key: str) -> Any:
        """
        Allow dictionary-style access to configuration values.

        This magic method enables accessing configuration values using square bracket
        notation, providing a more Pythonic interface alongside the get() method.
        Internally delegates to get() for consistency.

        Args:
            key: Configuration key (case-insensitive). Must exist in the DEFAULTS dictionary.

        Returns:
            The configuration value, cast to the appropriate type as defined in DEFAULTS.

        Raises:
            ValueError: If the key is not defined in DEFAULTS, or if type casting fails.
        """
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        """
        Check if a configuration key exists in DEFAULTS.

        This magic method enables using the 'in' operator to check if a configuration
        key is defined, providing a Pythonic way to validate keys before accessing them.
        Note: This checks if the key exists in DEFAULTS, not if it has a value set
        in environment variables.

        Args:
            key: Configuration key to check (case-insensitive)

        Returns:
            True if the key exists in DEFAULTS, False otherwise
        """
        return key.upper() in self.DEFAULTS
