"""Unit tests for logging configuration and utilities."""

import logging
import logging.handlers
import os
import sys
from unittest.mock import Mock, patch

import pytest
from app.core.logger import (
    _get_console_handler,
    _get_file_handler,
    get_logger,
    setup_logging,
)


@pytest.fixture
def mock_settings(monkeypatch):
    """Fixture to mock settings with default values."""
    from app.core import config

    mock_settings_obj = Mock()
    mock_settings_obj.log_console_handler = True
    mock_settings_obj.log_console_level = "INFO"
    mock_settings_obj.log_file_handler = False
    mock_settings_obj.log_file_level = "DEBUG"
    mock_settings_obj.log_file_dir = "log"

    monkeypatch.setattr(config, "settings", mock_settings_obj)
    return mock_settings_obj


def test_setup_logging_console_handler_only(mock_settings):
    """Test that setup_logging configures console handler when enabled."""
    mock_settings.log_console_handler = True
    mock_settings.log_file_handler = False

    with (
        patch("logging.basicConfig") as mock_basic_config,
        patch("app.core.logger.settings", mock_settings),
    ):
        setup_logging()

        assert mock_basic_config.called
        call_kwargs = mock_basic_config.call_args[1]
        assert call_kwargs["level"] == "DEBUG"
        assert call_kwargs["force"] is True
        assert len(call_kwargs["handlers"]) == 1
        assert isinstance(call_kwargs["handlers"][0], logging.StreamHandler)


def test_setup_logging_file_handler_only(mock_settings):
    """Test that setup_logging configures file handler when enabled."""
    mock_settings.log_console_handler = False
    mock_settings.log_file_handler = True

    with (
        patch("logging.basicConfig") as mock_basic_config,
        patch("app.core.logger.settings", mock_settings),
        patch("os.makedirs") as mock_makedirs,
        patch("logging.handlers.RotatingFileHandler") as _mock_file_handler,
    ):
        setup_logging()

        assert mock_basic_config.called
        call_kwargs = mock_basic_config.call_args[1]
        assert len(call_kwargs["handlers"]) == 1
        assert mock_makedirs.called


def test_setup_logging_both_handlers(mock_settings):
    """Test that setup_logging configures both handlers when enabled."""
    mock_settings.log_console_handler = True
    mock_settings.log_file_handler = True

    with (
        patch("logging.basicConfig") as mock_basic_config,
        patch("app.core.logger.settings", mock_settings),
        patch("os.makedirs"),
        patch("logging.handlers.RotatingFileHandler"),
    ):
        setup_logging()

        call_kwargs = mock_basic_config.call_args[1]
        assert len(call_kwargs["handlers"]) == 2


def test_setup_logging_no_handlers(mock_settings):
    """Test that setup_logging works with no handlers enabled."""
    mock_settings.log_console_handler = False
    mock_settings.log_file_handler = False

    with (
        patch("logging.basicConfig") as mock_basic_config,
        patch("app.core.logger.settings", mock_settings),
    ):
        setup_logging()

        call_kwargs = mock_basic_config.call_args[1]
        assert len(call_kwargs["handlers"]) == 0


def test_get_console_handler_configuration():
    """Test that console handler is configured correctly."""
    formatter = logging.Formatter("%(message)s")

    handler = _get_console_handler("DEBUG", formatter)

    assert isinstance(handler, logging.StreamHandler)
    assert handler.level == logging.DEBUG
    assert handler.stream == sys.stdout
    assert handler.formatter == formatter


def test_get_console_handler_different_levels():
    """Test that console handler respects different log levels."""
    formatter = logging.Formatter("%(message)s")

    levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    expected_levels = [
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    ]

    for level_str, expected_level in zip(levels, expected_levels):
        handler = _get_console_handler(level_str, formatter)
        assert handler.level == expected_level


def test_get_file_handler_creates_directory():
    """Test that file handler creates log directory if it doesn't exist."""
    formatter = logging.Formatter("%(message)s")
    log_dir = "test_logs"

    with (
        patch("os.makedirs") as mock_makedirs,
        patch("logging.handlers.RotatingFileHandler") as _mock_file_handler,
    ):
        _get_file_handler("DEBUG", formatter, log_dir)

        mock_makedirs.assert_called_once_with(log_dir, exist_ok=True)


def test_get_file_handler_configuration():
    """Test that file handler is configured correctly."""
    formatter = logging.Formatter("%(message)s")
    log_dir = "test_logs"

    with (
        patch("os.makedirs"),
        patch("logging.handlers.RotatingFileHandler") as mock_file_handler,
    ):
        mock_handler_instance = Mock()
        mock_file_handler.return_value = mock_handler_instance

        _handler = _get_file_handler("INFO", formatter, log_dir)

        assert mock_file_handler.called
        call_args = mock_file_handler.call_args
        assert call_args[1]["maxBytes"] == 10 * 1024 * 1024
        assert call_args[1]["backupCount"] == 5
        mock_handler_instance.setLevel.assert_called_once_with("INFO")
        mock_handler_instance.setFormatter.assert_called_once_with(formatter)


def test_get_file_handler_filename_format():
    """Test that file handler generates correct filename format."""
    formatter = logging.Formatter("%(message)s")
    log_dir = "test_logs"

    with (
        patch("os.makedirs"),
        patch("logging.handlers.RotatingFileHandler") as mock_file_handler,
        patch("app.core.logger.datetime") as mock_datetime,
    ):
        mock_now = Mock()
        mock_now.strftime.return_value = "2024010112"
        mock_datetime.now.return_value = mock_now

        _get_file_handler("DEBUG", formatter, log_dir)

        expected_filename = os.path.join(log_dir, "backtrain_2024010112.log")
        mock_file_handler.assert_called_once()
        assert mock_file_handler.call_args[0][0] == expected_filename


def test_get_logger_returns_logger_instance():
    """Test that get_logger returns a Logger instance."""
    logger = get_logger("test_module")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"


def test_get_logger_different_names():
    """Test that get_logger returns different loggers for different names."""
    logger1 = get_logger("module1")
    logger2 = get_logger("module2")

    assert logger1.name == "module1"
    assert logger2.name == "module2"
    assert logger1 is not logger2


def test_get_logger_same_name_returns_same_instance():
    """Test that get_logger returns the same instance for the same name."""
    logger1 = get_logger("same_module")
    logger2 = get_logger("same_module")

    assert logger1 is logger2


def test_formatter_configuration():
    """Test that logging formatter has correct format and date format."""
    with patch("logging.basicConfig") as mock_basic_config:
        with patch("app.core.config.settings") as mock_settings:
            mock_settings.log_console_handler = True
            mock_settings.log_console_level = "INFO"
            mock_settings.log_file_handler = False

            setup_logging()

            handlers = mock_basic_config.call_args[1]["handlers"]
            formatter = handlers[0].formatter

            assert (
                formatter._fmt
                == "%(asctime)s - %(module)s.%(funcName)s - %(levelname)s - %(message)s"
            )
            assert formatter.datefmt == "%Y-%m-%d %H:%M:%S"


def test_setup_logging_with_custom_log_directory(mock_settings):
    """Test that setup_logging uses custom log directory from settings."""
    custom_dir = "/custom/log/path"
    mock_settings.log_file_handler = True
    mock_settings.log_file_dir = custom_dir

    with (
        patch("logging.basicConfig"),
        patch("app.core.logger.settings", mock_settings),
        patch("os.makedirs") as mock_makedirs,
        patch("logging.handlers.RotatingFileHandler"),
    ):
        setup_logging()

        mock_makedirs.assert_called_once_with(custom_dir, exist_ok=True)


def test_file_handler_rotation_parameters():
    """Test that file handler has correct rotation parameters."""
    formatter = logging.Formatter("%(message)s")

    with (
        patch("os.makedirs"),
        patch("logging.handlers.RotatingFileHandler") as mock_file_handler,
    ):
        _get_file_handler("DEBUG", formatter, "log")

        call_kwargs = mock_file_handler.call_args[1]
        assert call_kwargs["maxBytes"] == 10485760  # 10MB in bytes
        assert call_kwargs["backupCount"] == 5
