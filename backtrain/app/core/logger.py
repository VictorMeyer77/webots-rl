"""
Logging configuration and utilities.

This module provides centralized logging configuration for the application,
ensuring consistent log formatting and level management across all modules.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime

from app.core.config import settings


def setup_logging() -> None:
    """
    Configure application-wide logging settings.

    Sets up the root logger with the configured log level from settings,
    applies a consistent format, and directs output to stdout for container
    compatibility. Supports both console and file logging handlers based on
    configuration settings.
    """

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(module)s.%(funcName)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handlers: list[logging.Handler] = []

    if settings.log_console_handler:
        console_handler = _get_console_handler(settings.log_console_level, formatter)
        handlers.append(console_handler)

    if settings.log_file_handler:
        file_handler = _get_file_handler(
            settings.log_file_level, formatter, settings.log_file_dir
        )
        handlers.append(file_handler)

    logging.basicConfig(level="DEBUG", force=True, handlers=handlers)


def _get_console_handler(
    level: str, formatter: logging.Formatter
) -> logging.StreamHandler:
    """
    Create and configure a console handler for logging to stdout.

    Args:
        level: The logging level for this handler (e.g., "DEBUG", "INFO").
        formatter: The formatter to apply to log messages.

    Returns:
        A configured StreamHandler instance.
    """
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    return console_handler


def _get_file_handler(
    level: str, formatter: logging.Formatter, log_dir: str
) -> logging.Handler:
    """
    Create and configure a rotating file handler for logging to files.

    Creates the log directory if it doesn't exist and sets up log rotation
    with a maximum file size of 10MB and 5 backup files.

    Args:
        level: The logging level for this handler (e.g., "DEBUG", "INFO").
        formatter: The formatter to apply to log messages.
        log_dir: The directory where log files should be stored.

    Returns:
        A configured RotatingFileHandler instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    file_name = os.path.join(
        log_dir, f"backtrain_{datetime.now().strftime('%Y%m%d%H')}.log"
    )
    file_handler = logging.handlers.RotatingFileHandler(
        file_name, maxBytes=10 * 1024 * 1024, backupCount=5
    )  # 10MB
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    return file_handler


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: The name of the module requesting the logger, typically __name__.

    Returns:
        A configured Logger instance for the specified module.
    """
    return logging.getLogger(name)
