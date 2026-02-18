"""Unit tests for the logging configuration and utilities."""

import json
import logging
import os
import tempfile
from unittest.mock import patch

from app.core.logger import (
    JsonFormatter,
    _get_console_handler,
    _get_file_handler,
    get_logger,
    setup_logging,
)


class TestJsonFormatter:
    """Test suite for the JsonFormatter class."""

    def test_json_formatter_basic_format(self):
        """Test that JsonFormatter produces valid JSON output."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="test_function",
        )
        record.threadName = "MainThread"

        result = formatter.format(record)

        # Verify it's valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_json_formatter_contains_timestamp(self):
        """Test that JsonFormatter includes timestamp field."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="test_function",
        )
        record.threadName = "MainThread"

        result = formatter.format(record)
        parsed = json.loads(result)

        assert "timestamp" in parsed
        assert isinstance(parsed["timestamp"], str)

    def test_json_formatter_contains_logger_name(self):
        """Test that JsonFormatter includes logger name."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="my.custom.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="test_function",
        )
        record.threadName = "MainThread"

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["logger"] == "my.custom.logger"

    def test_json_formatter_contains_module(self):
        """Test that JsonFormatter includes module name."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/test_module.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="test_function",
        )
        record.module = "test_module"
        record.threadName = "MainThread"

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["module"] == "test_module"

    def test_json_formatter_contains_function(self):
        """Test that JsonFormatter includes function name."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="my_test_function",
        )
        record.threadName = "MainThread"

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["function"] == "my_test_function"

    def test_json_formatter_contains_thread(self):
        """Test that JsonFormatter includes thread name."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="test_function",
        )
        record.threadName = "WorkerThread-1"

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["thread"] == "WorkerThread-1"

    def test_json_formatter_contains_level(self):
        """Test that JsonFormatter includes log level."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.WARNING,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="test_function",
        )
        record.threadName = "MainThread"

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["level"] == "WARNING"

    def test_json_formatter_contains_message(self):
        """Test that JsonFormatter includes the log message."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="This is a test message",
            args=(),
            exc_info=None,
            func="test_function",
        )
        record.threadName = "MainThread"

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["message"] == "This is a test message"

    def test_json_formatter_with_different_log_levels(self):
        """Test JsonFormatter with various log levels."""
        formatter = JsonFormatter()
        levels = [
            (logging.DEBUG, "DEBUG"),
            (logging.INFO, "INFO"),
            (logging.WARNING, "WARNING"),
            (logging.ERROR, "ERROR"),
            (logging.CRITICAL, "CRITICAL"),
        ]

        for level_num, level_name in levels:
            record = logging.LogRecord(
                name="test.logger",
                level=level_num,
                pathname="/path/to/file.py",
                lineno=42,
                msg="Test message",
                args=(),
                exc_info=None,
                func="test_function",
            )
            record.threadName = "MainThread"

            result = formatter.format(record)
            parsed = json.loads(result)

            assert parsed["level"] == level_name

    def test_json_formatter_with_formatted_message(self):
        """Test JsonFormatter with string formatting in message."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Value is %s and count is %d",
            args=("test", 42),
            exc_info=None,
            func="test_function",
        )
        record.threadName = "MainThread"

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["message"] == "Value is test and count is 42"

    def test_json_formatter_all_required_fields(self):
        """Test that JsonFormatter includes all required fields."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="test_function",
        )
        record.threadName = "MainThread"
        record.module = "test_module"

        result = formatter.format(record)
        parsed = json.loads(result)

        required_fields = [
            "timestamp",
            "logger",
            "module",
            "function",
            "thread",
            "level",
            "message",
        ]
        for field in required_fields:
            assert field in parsed, f"Field '{field}' is missing from JSON output"

    def test_json_formatter_output_is_single_line(self):
        """Test that JsonFormatter produces single-line JSON (no pretty printing)."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="test_function",
        )
        record.threadName = "MainThread"

        result = formatter.format(record)

        # Single-line JSON should not contain newlines (except possibly at the end)
        assert result.strip().count("\n") == 0

    def test_json_formatter_with_special_characters_in_message(self):
        """Test JsonFormatter handles special characters in log message."""
        formatter = JsonFormatter()
        special_message = 'Message with "quotes" and \\backslashes\\ and \nnewlines'
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg=special_message,
            args=(),
            exc_info=None,
            func="test_function",
        )
        record.threadName = "MainThread"

        result = formatter.format(record)
        parsed = json.loads(result)

        # JSON parsing should handle escaping correctly
        assert parsed["message"] == special_message

    def test_json_formatter_with_unicode_characters(self):
        """Test JsonFormatter handles unicode characters in log message."""
        formatter = JsonFormatter()
        unicode_message = "Message with unicode: 你好, мир, 🎉"
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg=unicode_message,
            args=(),
            exc_info=None,
            func="test_function",
        )
        record.threadName = "MainThread"

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["message"] == unicode_message

    def test_json_formatter_with_empty_message(self):
        """Test JsonFormatter handles empty log message."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="",
            args=(),
            exc_info=None,
            func="test_function",
        )
        record.threadName = "MainThread"

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["message"] == ""
        assert "message" in parsed

    def test_json_formatter_with_exception_info(self):
        """Test that JsonFormatter includes exception information when present."""
        import sys

        formatter = JsonFormatter()

        # Create an exception
        try:
            raise ValueError("Test exception")
        except ValueError:
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="/path/to/file.py",
            lineno=42,
            msg="An error occurred",
            args=(),
            exc_info=exc_info,
            func="test_function",
        )
        record.threadName = "MainThread"

        result = formatter.format(record)
        parsed = json.loads(result)

        assert "exception" in parsed
        assert "ValueError: Test exception" in parsed["exception"]
        assert "Traceback" in parsed["exception"]

    def test_json_formatter_with_stack_info(self):
        """Test that JsonFormatter includes stack info when present."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message with stack",
            args=(),
            exc_info=None,
            func="test_function",
        )
        record.threadName = "MainThread"
        record.stack_info = "Stack trace information here"

        result = formatter.format(record)
        parsed = json.loads(result)

        assert "stack_info" in parsed
        assert parsed["stack_info"] == "Stack trace information here"


class TestGetConsoleHandler:
    """Test suite for the _get_console_handler function."""

    def test_get_console_handler_returns_stream_handler(self):
        """Test that _get_console_handler returns a StreamHandler instance."""
        handler = _get_console_handler("INFO")

        assert isinstance(handler, logging.StreamHandler)

    def test_get_console_handler_sets_level(self):
        """Test that _get_console_handler sets the correct log level."""
        handler = _get_console_handler("WARNING")

        assert handler.level == logging.WARNING

    def test_get_console_handler_has_formatter(self):
        """Test that _get_console_handler has a formatter attached."""
        handler = _get_console_handler("INFO")

        assert handler.formatter is not None

    def test_get_console_handler_different_levels(self):
        """Test _get_console_handler with different log levels."""
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in levels:
            handler = _get_console_handler(level)
            assert handler.level == getattr(logging, level)


class TestGetFileHandler:
    """Test suite for the _get_file_handler function."""

    def test_get_file_handler_returns_rotating_file_handler(self):
        """Test that _get_file_handler returns a RotatingFileHandler instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = _get_file_handler("INFO", temp_dir)

            assert isinstance(handler, logging.handlers.RotatingFileHandler)

    def test_get_file_handler_creates_directory(self):
        """Test that _get_file_handler creates the log directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = os.path.join(temp_dir, "logs", "nested")
            _ = _get_file_handler("INFO", log_dir)

            assert os.path.exists(log_dir)

    def test_get_file_handler_sets_level(self):
        """Test that _get_file_handler sets the correct log level."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = _get_file_handler("ERROR", temp_dir)

            assert handler.level == logging.ERROR

    def test_get_file_handler_has_json_formatter(self):
        """Test that _get_file_handler uses JsonFormatter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = _get_file_handler("INFO", temp_dir)

            assert isinstance(handler.formatter, JsonFormatter)

    def test_get_file_handler_creates_log_file(self):
        """Test that _get_file_handler creates the log file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = _get_file_handler("INFO", temp_dir)
            expected_file = os.path.join(temp_dir, "backtrain.log")

            # The file is created when the handler is instantiated
            assert handler.baseFilename == expected_file

    def test_get_file_handler_max_bytes_configuration(self):
        """Test that _get_file_handler configures maxBytes correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = _get_file_handler("INFO", temp_dir)

            # Should be 10MB
            assert handler.maxBytes == 10 * 1024 * 1024

    def test_get_file_handler_backup_count_configuration(self):
        """Test that _get_file_handler configures backupCount correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = _get_file_handler("INFO", temp_dir)

            # Should keep 5 backup files
            assert handler.backupCount == 5


class TestGetLogger:
    """Test suite for the get_logger function."""

    def test_get_logger_returns_logger_instance(self):
        """Test that get_logger returns a Logger instance."""
        logger = get_logger("test.module")

        assert isinstance(logger, logging.Logger)

    def test_get_logger_with_module_name(self):
        """Test that get_logger creates logger with correct name."""
        logger_name = "app.core.test_module"
        logger = get_logger(logger_name)

        assert logger.name == logger_name

    def test_get_logger_returns_same_instance(self):
        """Test that get_logger returns the same instance for the same name."""
        logger1 = get_logger("test.module")
        logger2 = get_logger("test.module")

        assert logger1 is logger2

    def test_get_logger_different_names(self):
        """Test that get_logger returns different instances for different names."""
        logger1 = get_logger("test.module1")
        logger2 = get_logger("test.module2")

        assert logger1 is not logger2
        assert logger1.name != logger2.name


class TestSetupLogging:
    """Test suite for the setup_logging function."""

    @patch("app.core.logger.settings")
    @patch("logging.basicConfig")
    def test_setup_logging_calls_basic_config(self, mock_basic_config, mock_settings):
        """Test that setup_logging calls logging.basicConfig."""
        mock_settings.log_console_handler = True
        mock_settings.log_console_level = "INFO"
        mock_settings.log_file_handler = False

        setup_logging()

        mock_basic_config.assert_called_once()

    @patch("app.core.logger.settings")
    @patch("logging.basicConfig")
    def test_setup_logging_with_console_handler_only(
        self, mock_basic_config, mock_settings
    ):
        """Test setup_logging with only console handler enabled."""
        mock_settings.log_console_handler = True
        mock_settings.log_console_level = "INFO"
        mock_settings.log_file_handler = False

        setup_logging()

        call_args = mock_basic_config.call_args
        handlers = call_args.kwargs["handlers"]
        assert len(handlers) == 1
        assert isinstance(handlers[0], logging.StreamHandler)

    @patch("app.core.logger.settings")
    @patch("logging.basicConfig")
    def test_setup_logging_with_file_handler_only(
        self, mock_basic_config, mock_settings
    ):
        """Test setup_logging with only file handler enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_settings.log_console_handler = False
            mock_settings.log_file_handler = True
            mock_settings.log_file_level = "DEBUG"
            mock_settings.log_file_dir = temp_dir

            setup_logging()

            call_args = mock_basic_config.call_args
            handlers = call_args.kwargs["handlers"]
            assert len(handlers) == 1
            assert isinstance(handlers[0], logging.handlers.RotatingFileHandler)

    @patch("app.core.logger.settings")
    @patch("logging.basicConfig")
    def test_setup_logging_with_both_handlers(self, mock_basic_config, mock_settings):
        """Test setup_logging with both console and file handlers enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_settings.log_console_handler = True
            mock_settings.log_console_level = "INFO"
            mock_settings.log_file_handler = True
            mock_settings.log_file_level = "DEBUG"
            mock_settings.log_file_dir = temp_dir

            setup_logging()

            call_args = mock_basic_config.call_args
            handlers = call_args.kwargs["handlers"]
            assert len(handlers) == 2

    @patch("app.core.logger.settings")
    @patch("logging.basicConfig")
    def test_setup_logging_with_no_handlers(self, mock_basic_config, mock_settings):
        """Test setup_logging with no handlers enabled."""
        mock_settings.log_console_handler = False
        mock_settings.log_file_handler = False

        setup_logging()

        call_args = mock_basic_config.call_args
        handlers = call_args.kwargs["handlers"]
        assert len(handlers) == 0

    @patch("app.core.logger.settings")
    @patch("logging.basicConfig")
    def test_setup_logging_sets_debug_level(self, mock_basic_config, mock_settings):
        """Test that setup_logging sets root level to DEBUG."""
        mock_settings.log_console_handler = True
        mock_settings.log_console_level = "INFO"
        mock_settings.log_file_handler = False

        setup_logging()

        call_args = mock_basic_config.call_args
        assert call_args.kwargs["level"] == "DEBUG"

    @patch("app.core.logger.settings")
    @patch("logging.basicConfig")
    def test_setup_logging_forces_reconfiguration(
        self, mock_basic_config, mock_settings
    ):
        """Test that setup_logging forces reconfiguration."""
        mock_settings.log_console_handler = True
        mock_settings.log_console_level = "INFO"
        mock_settings.log_file_handler = False

        setup_logging()

        call_args = mock_basic_config.call_args
        assert call_args.kwargs["force"] is True
