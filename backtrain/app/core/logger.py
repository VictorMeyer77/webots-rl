"""
Logging configuration and utilities.

This module provides centralized logging configuration for the application,
ensuring consistent log formatting and level management across all modules.
"""

import json
import logging
import logging.handlers
import os
import sys

from app.core.config import settings

DEFAULT_LOG_FORMAT = "%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
FILE_MAX_BYTES = 10 * 1024 * 1024  # 10MB
FILE_BACKUP_COUNT = 5
DEFAULT_LOG_FILENAME = "backtrain.log"


class JsonFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.

    Formats log records as JSON strings with consistent fields including
    timestamp, logger name, module, function, line number, thread, process,
    level, and message. This format is ideal for log aggregation systems and
    structured log analysis tools like ELK stack, Splunk, or CloudWatch.

    The formatter automatically includes exception tracebacks and stack info
    when present in the log record, making it suitable for production debugging.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as a JSON string.

        Creates a dictionary with standardized fields from the log record and
        serializes it to JSON. Includes exception information and stack traces
        when available.

        Args:
            record: The LogRecord instance to format.

        Returns:
            A JSON-formatted string containing the log record information.
        """
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.threadName,
            "process": record.process,
            "level": record.levelname,
            "message": record.getMessage(),
        }

        # Include exception information if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        # Include stack info if present (Python 3.2+)
        if hasattr(record, "stack_info") and record.stack_info:
            log_record["stack_info"] = record.stack_info

        return json.dumps(log_record)


def setup_logging() -> None:
    """
    Configure application-wide logging settings.

    Sets up the root logger with the configured log level from settings,
    applies a consistent format, and directs output to stdout for container
    compatibility. Supports both console and file logging handlers based on
    configuration settings.
    """

    handlers: list[logging.Handler] = []

    if settings.log_console_handler:
        console_handler = _get_console_handler(settings.log_console_level)
        handlers.append(console_handler)

    if settings.log_file_handler:
        file_handler = _get_file_handler(settings.log_file_level, settings.log_file_dir)
        handlers.append(file_handler)

    logging.basicConfig(level="DEBUG", force=True, handlers=handlers)


def _get_console_handler(level: str) -> logging.StreamHandler:
    """
    Create and configure a console handler for logging to stdout.

    The console handler uses a human-readable format suitable for development
    and debugging. It includes timestamp with milliseconds, logger name,
    log level, and message. Output is directed to stdout for compatibility
    with containerized environments and logging aggregation tools.

    Args:
        level: The logging level for this handler (e.g., "DEBUG", "INFO", "WARNING").
               Case-sensitive string matching Python logging levels.

    Returns:
        A configured StreamHandler instance writing to stdout with human-readable formatting.
    """

    formatter = logging.Formatter(
        fmt=DEFAULT_LOG_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    return console_handler


def _get_file_handler(level: str, log_dir: str) -> logging.Handler:
    """
    Create and configure a rotating file handler for logging to files.

    Creates the log directory if it doesn't exist and sets up log rotation
    with a maximum file size of 10MB and 5 backup files. Uses JSON formatting
    for structured log analysis, making it suitable for production environments
    and integration with log aggregation systems.

    Log files are named 'backtrain.log', with rotated files named
    'backtrain.log.1', 'backtrain.log.2', etc. The oldest backup is
    automatically deleted when the backup count is exceeded.

    Args:
        level: The logging level for this handler (e.g., "DEBUG", "INFO").
               Case-sensitive string matching Python logging levels.
        log_dir: The directory where log files should be stored.
                 Created automatically if it doesn't exist.

    Returns:
        A configured RotatingFileHandler instance with JSON formatting.
    """
    os.makedirs(log_dir, exist_ok=True)
    formatter = JsonFormatter()
    file_name = os.path.join(log_dir, DEFAULT_LOG_FILENAME)
    file_handler = logging.handlers.RotatingFileHandler(
        file_name, maxBytes=FILE_MAX_BYTES, backupCount=FILE_BACKUP_COUNT
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    return file_handler


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    This function returns a logger configured with the application's
    logging settings. The logger name should typically be __name__
    to reflect the module hierarchy, which helps with filtering and
    debugging by showing the exact module that generated each log message.

    Args:
        name: The name of the module requesting the logger, typically __name__.
              Using __name__ provides automatic hierarchical naming
              (e.g., 'app.core.memory', 'app.routers.environment').

    Returns:
        A configured Logger instance for the specified module that
        inherits settings from the root logger configured by setup_logging().
    """
    return logging.getLogger(name)
