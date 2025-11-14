import logging.handlers
import os
from datetime import datetime

"""
Logger Configuration Module for Robotic Agent

This module provides a centralized logging configuration for the robotic agent project.
It supports both console and file logging with rotating file handlers to manage log file sizes.

Example:
    Basic usage with console logging only:

    ```python
    from brain.utils.logger import logger

    logger.add_console_logger(logging.INFO)
    logger.add_file_logger(logging.DEBUG)
    logger.logger.warning("Battery low")
    logger.logger.error("Collision detected")
    ```
"""

LOG_DIR = "/Users/victormeyer/Dev/Self/webots-rl/output/logs"


class Logger:
    """
    A wrapper class for Python's logging module with support for console and file handlers.

    This class provides a centralized logger with customizable output handlers.
    It supports both console output and rotating file logging.

    Attributes:
        logger (logging.Logger): The underlying Python logger instance
        formatter (logging.Formatter): The formatter used for all log messages
    """

    logger: logging.Logger
    formatter: logging.Formatter

    def __init__(self):
        """
        Initialize the Logger with a configured logger instance.

        Sets up the base logger with DEBUG level and a timestamp-based formatter.
        Prevents log propagation to avoid duplicate messages.
        """
        self.logger = logging.getLogger("RoboticAgentLogger")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.formatter = logging.Formatter(
            fmt="%(asctime)s - %(module)s.%(funcName)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

    def __call__(self):
        return self.logger

    def add_console_logger(self, level: int):
        """
        Add a console (stdout) handler to the logger.

        Args:
            level (int): Minimum logging level for console output.
                Use logging.DEBUG, logging.INFO, logging.WARNING,
                logging.ERROR, or logging.CRITICAL.

        Example:
            ```python
            logger.add_console_logger(logging.INFO)
            ```
        """
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)

    def add_file_logger(self, level: int):
        """
        Add a rotating file handler to the logger.

        Creates the log directory if it doesn't exist and sets up a rotating
        file handler with automatic log rotation when files reach 10MB.

        Args:
            level (int): Minimum logging level for file output.
                Use logging.DEBUG, logging.INFO, logging.WARNING,
                logging.ERROR, or logging.CRITICAL.

        Note:
            - Log files are named with timestamp: robotic_agent_YYYYMMDD_HHMMSS.log
            - Maximum file size: 10MB
            - Backup count: 5 files

        Example:
            ```python
            logger.add_file_logger(logging.DEBUG)
            ```
        """
        os.makedirs(LOG_DIR, exist_ok=True)
        file_name = os.path.join(LOG_DIR, f"webot_rl_{datetime.now().strftime('%Y%m%d%H')}.log")
        file_handler = logging.handlers.RotatingFileHandler(file_name, maxBytes=10 * 1024 * 1024, backupCount=5)  # 10MB
        file_handler.setLevel(level)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)


logger = Logger()
