"""
Logging configuration and utilities.

This module provides centralised logging configuration for the corl application.
It exposes two handler factories and a top-level ``setup_logging`` function that
wires them together based on :class:`~corl.utils.config.Config` values.

Two output formats are supported:
    - **Console** (human-readable): plain-text lines via :data:`DEFAULT_LOG_FORMAT`
      written to ``stdout``, controlled by ``log_console_handler`` /
      ``log_console_level`` config keys.
    - **File** (structured): JSON lines via :class:`JsonFormatter` written to a
      :class:`~logging.handlers.RotatingFileHandler`, controlled by
      ``log_file_handler``, ``log_file_level``, ``log_file_dir``,
      ``log_file_name``, ``log_file_max_bytes``, and ``log_file_backup_count``
      config keys.

Constants:
    DEFAULT_LOG_FORMAT (str): ``%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s``
    DEFAULT_DATE_FORMAT (str): ``%Y-%m-%d %H:%M:%S``

Classes:
    JsonFormatter: :class:`logging.Formatter` subclass that serialises records to JSON.

Functions:
    setup_logging:      Configure the root logger from a :class:`~corl.utils.config.Config`.
    _get_console_handler: Build a stdout :class:`~logging.StreamHandler`.
    _get_file_handler:    Build a rotating :class:`~logging.handlers.RotatingFileHandler`.
"""

import json
import logging
import logging.handlers
import sys
from pathlib import Path

from corl.utils.config import Config

DEFAULT_LOG_FORMAT = "%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


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

        Serialises the record to a flat JSON object. The following fields are
        always present:

        .. code-block:: json

            {
                "timestamp": "<ISO-style date>",
                "logger":    "<logger name>",
                "module":    "<module name>",
                "function":  "<function name>",
                "line":      <int>,
                "thread":    "<thread name>",
                "process":   <int>,
                "level":     "INFO",
                "message":   "<formatted message>"
            }

        Two additional keys are included only when data is available:
            - ``"exception"`` — formatted traceback string, present when
              ``record.exc_info`` is set.
            - ``"stack_info"`` — stack info string, present when
              ``record.stack_info`` is set.

        Args:
            record (logging.LogRecord): The log record to serialise.

        Returns:
            str: A single-line JSON string representing the log record.
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


def setup_logging(config: Config) -> None:
    """
    Configure the root logger from a :class:`~corl.utils.config.Config`.

    Builds the list of active handlers based on the ``log_console_handler``
    and ``log_file_handler`` config flags, then calls :func:`logging.basicConfig`
    with ``force=True`` so that any previously attached handlers are replaced.
    The root logger is always set to ``DEBUG`` so that individual handlers can
    control their own verbosity via their own level.

    After configuring the root logger, the following noisy third-party loggers
    are suppressed to ``ERROR`` level to keep output clean:
        - ``h5py``
        - ``urllib3``
        - ``mlflow``

    Args:
        config (Config): Loaded application configuration. The following keys
            are read: ``log_console_handler``, ``log_console_level``,
            ``log_file_handler``, and all keys consumed by
            :func:`_get_file_handler`.

    Returns:
        None
    """

    handlers: list[logging.Handler] = []

    if config.get("log_console_handler"):
        console_handler = _get_console_handler(config.get("log_console_level"))
        handlers.append(console_handler)

    if config.get("log_file_handler"):
        file_handler = _get_file_handler(config)
        handlers.append(file_handler)

    logging.basicConfig(level="DEBUG", force=True, handlers=handlers)

    logging.getLogger("h5py").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("mlflow").setLevel(logging.ERROR)


def _get_console_handler(level: str) -> logging.StreamHandler:
    """
    Build a stdout :class:`~logging.StreamHandler` with human-readable formatting.

    The handler is formatted with :data:`DEFAULT_LOG_FORMAT` and
    :data:`DEFAULT_DATE_FORMAT`, producing lines such as::

        2026-03-08 12:00:00.123 - corl.trainer - INFO - Training started

    Output is directed to ``sys.stdout`` for compatibility with containerised
    environments and log aggregation tools that read the standard streams.

    Args:
        level (str): The minimum logging level for this handler
            (e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``). Case-sensitive,
            must match a standard Python logging level name.

    Returns:
        logging.StreamHandler: A configured handler writing plain-text records
            to ``stdout`` using :data:`DEFAULT_LOG_FORMAT`.
    """

    formatter = logging.Formatter(
        fmt=DEFAULT_LOG_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    return console_handler


def _get_file_handler(config: Config) -> logging.Handler:
    """
    Create and configure a rotating file handler for logging to files.

    Resolves the log directory from *config*, creating it (and any missing
    parent directories) if it does not already exist. When a ``train_id`` is
    set in *config*, log files are written inside a per-session subdirectory
    (``<log_file_dir>/<train_id>/``); otherwise they are written directly to
    ``log_file_dir``. Log rotation is governed by the ``log_file_max_bytes``
    and ``log_file_backup_count`` configuration keys.

    Args:
        config (Config): Loaded application configuration. The following keys
            are read: ``log_file_dir``, ``train_id``, ``log_file_name``,
            ``log_file_level``, ``log_file_max_bytes``, ``log_file_backup_count``.

    Returns:
        logging.Handler: A configured :class:`~logging.handlers.RotatingFileHandler`
            that writes JSON-formatted records to the resolved log file path.
    """

    log_dir = Path(config.get("log_file_dir"))
    if config.get("train_id") is not None:
        log_dir = log_dir / config.get("train_id")

    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = JsonFormatter()
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / f"{config.get('log_file_name')}.log",
        maxBytes=config.get("log_file_max_bytes"),
        backupCount=config.get("log_file_backup_count"),
    )
    file_handler.setLevel(config.get("log_file_level"))
    file_handler.setFormatter(formatter)
    return file_handler


logger = logging.getLogger(__name__)
