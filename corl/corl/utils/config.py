"""
Configuration management module for CORL (Control and Reinforcement Learning).

This module provides a flexible configuration system that loads settings from
environment variables and .env files with type validation and caching support.
"""

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

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

    def __str__(self) -> str:
        return self.value


@dataclass
class ConfigItem:
    """
    Schema for a single configuration entry.

    Bundles the default value, optional type cast, and a human-readable
    description for one key in :attr:`Config.DEFAULTS`.

    Attributes:
        default: Fallback value used when no matching environment variable is
            found. May be ``None`` for optional runtime-only keys (e.g.
            ``TRAIN_ID``, ``WORKER_ID``).
        cast: Callable used to coerce the raw environment variable string to
            the desired type. Supported values: ``str``, ``int``, ``float``,
            ``LogLevel``, or ``None`` (no coercion, raw string is returned).
            ``bool`` receives special treatment: truthy strings are
            ``"true"``, ``"1"``, ``"yes"``, ``"on"`` (case-insensitive).
        description: Human-readable explanation of what the key controls.
            Included in :exc:`ValueError` messages when casting fails.
    """

    default: Any | None
    cast: type | None = None
    description: str = ""


class Config:
    """
    Load and manage application configuration from environment variables and ``.env``.

    Values are resolved from ``os.environ`` (populated by ``python-dotenv`` on
    construction), cast to the declared type, and cached after the first access.
    Runtime overrides can be written back via :meth:`set`, which also invalidates
    the corresponding cache entry so the next :meth:`get` picks up the new value.

    Environment variable naming convention::

        {PREFIX}_{KEY}   →   e.g. WEBOTS_API_HOST, WEBOTS_API_PORT

    The prefix is always uppercased, so ``Config(prefix="webots")`` and
    ``Config(prefix="WEBOTS")`` are equivalent.

    Attributes:
        DEFAULTS (dict[str, ConfigItem]): Class-level schema mapping every
            supported key to its :class:`ConfigItem` (default, cast, description).
        prefix (str): Uppercased prefix prepended to every env-var lookup.
        _cache (dict[str, Any]): Internal store for already-resolved values.

    Methods:
        get(key):          Resolve and return a configuration value.
        set(key, value):   Write a value to ``os.environ`` and invalidate cache.
        environ():         Return a snapshot of the current ``os.environ``.
        __getitem__(key):  Sugar for ``get(key)``; enables ``config["KEY"]``.
        __contains__(key): ``True`` if *key* is a known key in ``DEFAULTS``.
    """

    DEFAULTS = {
        # Webots environment
        "BIN_PATH": ConfigItem(
            default="/Applications/Webots.app/Contents/MacOS/webots",
            cast=str,
            description="Path to Webots binary executable",
        ),
        "EXPERIMENTS_DIR": ConfigItem(
            default="projects/",
            cast=str,
            description="Root directory containing per-world experiment project folders",
        ),
        # Backtrain API Configuration
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
        # Logging Configuration
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
        "LOG_FILE_NAME": ConfigItem(
            default="webots_rl",
            cast=str,
            description="File name prefix for log files (without extension)",
        ),
        "LOG_FILE_MAX_BYTES": ConfigItem(
            default=10 * 1024 * 1024,
            cast=int,
            description="Maximum size in bytes for a log file before rotation occurs",
        ),
        "LOG_FILE_BACKUP_COUNT": ConfigItem(
            default=5,
            cast=int,
            description="Number of backup log files to keep when rotating",
        ),
        # Training Configuration
        # Dynamic configuration items that are expected to be set at runtime by the trainer or worker processes.
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
        "WORLD_NAME": ConfigItem(
            default=None,
            cast=str,
            description="Name of the experiment world being trained on",
        ),
        # Trainer
        "TRAINER_OUTPUT_DIR": ConfigItem(
            default=".train/",
            cast=str,
            description="Root output directory for trainer artefacts (checkpoints, TensorBoard logs, MLflow runs)",
        ),
        "TRAINER_WORKER_TIMEOUT": ConfigItem(
            default=60,
            cast=int,
            description="Time in seconds to wait for a worker to respond before marking it as unresponsive",
        ),
        # Environment
        "ENVIRONMENT_RECORD_FREQUENCY": ConfigItem(
            default=50,
            cast=int,
            description="Interval in episodes at which worker 0 (witness) records an environment video during training",
        ),
        # Agent
        "AGENT_REQUEST_TIMEOUT": ConfigItem(
            default=60.0,
            cast=float,
            description="Seconds to wait for an agent request to get action from the trainer.",
        ),
    }

    def __init__(self, env_path: Path | None = None, prefix: str = "webots"):
        """
        Initialise the configuration object and load any ``.env`` file.

        Calls :func:`dotenv.load_dotenv` to populate ``os.environ`` from a
        ``.env`` file in the working directory (or any parent directory). If no
        ``.env`` file is found the call is a no-op — existing environment
        variables are always respected.

        Args:
            env_path (Path | None): Path to a ``.env`` file to load. When provided
                and the file exists, that file is loaded explicitly. When ``None``
                or the path does not exist, ``load_dotenv()`` searches the working
                directory and its parents automatically. Defaults to ``None``.
            prefix (str): Prefix for environment variable names. Uppercased
                automatically, so ``"webots"`` and ``"WEBOTS"`` are equivalent.
                Defaults to ``"webots"``.
        """
        self.prefix = prefix
        self._cache: dict[str, Any] = {}
        if env_path is not None and env_path.exists():
            load_dotenv(env_path)
        else:
            load_dotenv()

    def get(self, key: str) -> Any:
        """
        Resolve and return a configuration value by key.

        Looks up the value from the cache, then from ``os.environ``, then falls
        back to :attr:`ConfigItem.default`. The resolved value is cast to the
        declared type and stored in the cache before being returned.

        Resolution steps:
            1. Return the cached value if one exists.
            2. Validate that *key* is present in :attr:`DEFAULTS`.
            3. Look up the environment variable ``{prefix}_{KEY}``.
            4. If the variable is absent, cache and return the default value.
            5. Cast the raw string to the declared type (``bool`` uses a
               truthy-string check; all others call ``cast(value)`` directly).
            6. Cache the cast result and return it.

        Args:
            key (str): Configuration key (case-insensitive). Must exist in
                :attr:`DEFAULTS`.

        Returns:
            Any: The configuration value cast to the type declared in
                :attr:`DEFAULTS`, or the default value if no environment
                variable is set.

        Raises:
            ValueError: If *key* is not defined in :attr:`DEFAULTS`, or if
                casting the environment variable string fails. The error message
                includes the env-var name, raw value, expected type, default,
                description, and the original exception to aid debugging.
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
        Enable square-bracket access to configuration values.

        Delegates directly to :meth:`get`, so ``config["API_PORT"]`` is
        equivalent to ``config.get("API_PORT")``.

        Args:
            key (str): Configuration key (case-insensitive). Must exist in
                :attr:`DEFAULTS`.

        Returns:
            Any: The configuration value; see :meth:`get` for full details.

        Raises:
            ValueError: If *key* is not defined in :attr:`DEFAULTS`, or if
                type casting fails.
        """
        return self.get(key)

    @staticmethod
    def environ() -> dict[str, str]:
        """
        Return a snapshot of the current environment variables.

        Since ``set()`` writes directly to ``os.environ``, this snapshot always
        reflects all values set via this class alongside the inherited shell environment.

        Returns:
            A copy of ``os.environ`` as a plain ``dict[str, str]``.
        """
        return os.environ.copy()

    def set(self, key: str, value: Any) -> None:
        """
        Write a configuration value to ``os.environ`` at runtime.

        Converts *value* to a string (``os.environ`` only accepts strings),
        stores it under the prefixed key, and invalidates the cache entry so
        the next :meth:`get` call resolves and casts the new value fresh.

        Args:
            key (str): Configuration key (case-insensitive). Must exist in
                :attr:`DEFAULTS`.
            value (Any): Value to store. Serialised via ``str(value)`` before
                being written to ``os.environ``.

        Returns:
            None

        Raises:
            ValueError: If *key* is not defined in :attr:`DEFAULTS`.
        """
        key = key.upper()
        if key not in self.DEFAULTS:
            raise ValueError(f"Configuration key '{key}' is not defined in DEFAULTS")
        env_key = f"{self.prefix}_{key}".upper()
        os.environ[env_key] = str(value)
        self._cache.pop(key, None)

    def __contains__(self, key: str) -> bool:
        """
        Return ``True`` if *key* is a known key in :attr:`DEFAULTS`.

        .. important::
            This checks schema membership only — it does **not** verify whether
            an environment variable is set or whether the resolved value is
            non-``None``. Keys with ``default=None`` (e.g. ``TRAIN_ID``,
            ``WORKER_ID``) will still return ``True``.

        Args:
            key (str): Configuration key to check (case-insensitive).

        Returns:
            bool: ``True`` if *key* exists in :attr:`DEFAULTS`, ``False`` otherwise.
        """
        return key.upper() in self.DEFAULTS
