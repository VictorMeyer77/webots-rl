"""
Unit tests for corl.utils.logger

Covers:
- JsonFormatter.format           – standard record, exception, stack_info
- _get_console_handler           – level, stream, formatter type
- _get_file_handler              – directory creation, file path, level,
                                   formatter type, with and without train_id
- setup_logging                  – handler registration for console / file /
                                   both / neither, third-party logger silencing
"""

import json
import logging
import logging.handlers
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from corl.utils.logger import (
    JsonFormatter,
    _get_console_handler,
    _get_file_handler,
    setup_logging,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    msg: str = "hello",
    level: int = logging.INFO,
    exc_info=None,
    stack_info: str | None = None,
) -> logging.LogRecord:
    """Return a minimal LogRecord."""
    record = logging.LogRecord(
        name="test.logger",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=exc_info,
    )
    if stack_info:
        record.stack_info = stack_info
    return record


def _make_config(
    *,
    console_handler: bool = False,
    file_handler: bool = False,
    console_level: str = "DEBUG",
    file_level: str = "INFO",
    log_file_dir: str = ".log",
    log_file_name: str = "test",
    log_file_max_bytes: int = 1024,
    log_file_backup_count: int = 2,
    train_id: str | None = None,
) -> MagicMock:
    """Return a Config mock with sensible defaults for logger tests."""
    cfg = MagicMock()
    values = {
        "log_console_handler": console_handler,
        "log_file_handler": file_handler,
        "log_console_level": console_level,
        "log_file_level": file_level,
        "log_file_dir": log_file_dir,
        "log_file_name": log_file_name,
        "log_file_max_bytes": log_file_max_bytes,
        "log_file_backup_count": log_file_backup_count,
        "train_id": train_id,
    }
    cfg.get = MagicMock(side_effect=lambda key: values[key.lower()])
    return cfg


# ---------------------------------------------------------------------------
# JsonFormatter
# ---------------------------------------------------------------------------


class TestJsonFormatter:
    """Tests for JsonFormatter.format."""

    def setup_method(self):
        self.formatter = JsonFormatter()

    def test_returns_valid_json(self):
        record = _make_record("test message")
        result = self.formatter.format(record)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_required_fields_present(self):
        record = _make_record("test message")
        parsed = json.loads(self.formatter.format(record))
        for field in (
            "timestamp",
            "logger",
            "module",
            "function",
            "line",
            "thread",
            "process",
            "level",
            "message",
        ):
            assert field in parsed, f"Missing field: {field}"

    def test_message_content(self):
        record = _make_record("my log message")
        parsed = json.loads(self.formatter.format(record))
        assert parsed["message"] == "my log message"

    def test_level_field(self):
        record = _make_record(level=logging.WARNING)
        parsed = json.loads(self.formatter.format(record))
        assert parsed["level"] == "WARNING"

    def test_logger_name(self):
        record = _make_record()
        parsed = json.loads(self.formatter.format(record))
        assert parsed["logger"] == "test.logger"

    def test_no_exception_key_when_absent(self):
        record = _make_record()
        parsed = json.loads(self.formatter.format(record))
        assert "exception" not in parsed

    def test_exception_included_when_present(self):
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = _make_record(exc_info=exc_info)
        parsed = json.loads(self.formatter.format(record))
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]
        assert "boom" in parsed["exception"]

    def test_no_stack_info_key_when_absent(self):
        record = _make_record()
        parsed = json.loads(self.formatter.format(record))
        assert "stack_info" not in parsed

    def test_stack_info_included_when_present(self):
        record = _make_record(stack_info="Stack trace line 1\nStack trace line 2")
        parsed = json.loads(self.formatter.format(record))
        assert "stack_info" in parsed
        assert "Stack trace" in parsed["stack_info"]

    def test_line_number_is_integer(self):
        record = _make_record()
        parsed = json.loads(self.formatter.format(record))
        assert isinstance(parsed["line"], int)

    def test_process_is_integer(self):
        record = _make_record()
        parsed = json.loads(self.formatter.format(record))
        assert isinstance(parsed["process"], int)


# ---------------------------------------------------------------------------
# _get_console_handler
# ---------------------------------------------------------------------------


class TestGetConsoleHandler:
    """Tests for _get_console_handler."""

    def test_returns_stream_handler(self):
        handler = _get_console_handler("INFO")
        assert isinstance(handler, logging.StreamHandler)

    def test_writes_to_stdout(self):
        handler = _get_console_handler("DEBUG")
        assert handler.stream is sys.stdout

    def test_level_is_set_correctly(self):
        handler = _get_console_handler("WARNING")
        assert handler.level == logging.WARNING

    def test_debug_level(self):
        handler = _get_console_handler("DEBUG")
        assert handler.level == logging.DEBUG

    def test_error_level(self):
        handler = _get_console_handler("ERROR")
        assert handler.level == logging.ERROR

    def test_formatter_is_not_json(self):
        handler = _get_console_handler("INFO")
        assert not isinstance(handler.formatter, JsonFormatter)

    def test_formatter_uses_default_format(self):
        from corl.utils.logger import DEFAULT_DATE_FORMAT, DEFAULT_LOG_FORMAT

        handler = _get_console_handler("INFO")
        assert handler.formatter._fmt == DEFAULT_LOG_FORMAT
        assert handler.formatter.datefmt == DEFAULT_DATE_FORMAT


# ---------------------------------------------------------------------------
# _get_file_handler
# ---------------------------------------------------------------------------


class TestGetFileHandler:
    """Tests for _get_file_handler."""

    def test_returns_rotating_file_handler(self, tmp_path):
        cfg = _make_config(log_file_dir=str(tmp_path), log_file_name="app")
        handler = _get_file_handler(cfg)
        assert isinstance(handler, logging.handlers.RotatingFileHandler)
        handler.close()

    def test_creates_log_directory(self, tmp_path):
        log_dir = tmp_path / "logs" / "nested"
        cfg = _make_config(log_file_dir=str(log_dir), log_file_name="app")
        handler = _get_file_handler(cfg)
        assert log_dir.exists()
        handler.close()

    def test_log_file_in_correct_path_without_train_id(self, tmp_path):
        cfg = _make_config(log_file_dir=str(tmp_path), log_file_name="run")
        handler = _get_file_handler(cfg)
        assert Path(handler.baseFilename) == tmp_path / "run.log"
        handler.close()

    def test_log_file_in_subdirectory_with_train_id(self, tmp_path):
        cfg = _make_config(
            log_file_dir=str(tmp_path),
            log_file_name="run",
            train_id="session_42",
        )
        handler = _get_file_handler(cfg)
        expected = tmp_path / "session_42" / "run.log"
        assert Path(handler.baseFilename) == expected
        handler.close()

    def test_creates_train_id_subdirectory(self, tmp_path):
        cfg = _make_config(
            log_file_dir=str(tmp_path),
            log_file_name="run",
            train_id="session_42",
        )
        handler = _get_file_handler(cfg)
        assert (tmp_path / "session_42").is_dir()
        handler.close()

    def test_level_is_set_correctly(self, tmp_path):
        cfg = _make_config(log_file_dir=str(tmp_path), file_level="WARNING")
        handler = _get_file_handler(cfg)
        assert handler.level == logging.WARNING
        handler.close()

    def test_formatter_is_json(self, tmp_path):
        cfg = _make_config(log_file_dir=str(tmp_path))
        handler = _get_file_handler(cfg)
        assert isinstance(handler.formatter, JsonFormatter)
        handler.close()

    def test_max_bytes_applied(self, tmp_path):
        cfg = _make_config(log_file_dir=str(tmp_path), log_file_max_bytes=512)
        handler = _get_file_handler(cfg)
        assert handler.maxBytes == 512
        handler.close()

    def test_backup_count_applied(self, tmp_path):
        cfg = _make_config(log_file_dir=str(tmp_path), log_file_backup_count=7)
        handler = _get_file_handler(cfg)
        assert handler.backupCount == 7
        handler.close()


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    """Tests for setup_logging."""

    def test_console_handler_registered_when_enabled(self):
        cfg = _make_config(console_handler=True)
        with patch("logging.basicConfig") as mock_basic:
            setup_logging(cfg)
            _, kwargs = mock_basic.call_args
            handlers = kwargs["handlers"]
            assert any(isinstance(h, logging.StreamHandler) for h in handlers)

    def test_file_handler_registered_when_enabled(self, tmp_path):
        cfg = _make_config(file_handler=True, log_file_dir=str(tmp_path))
        with patch("logging.basicConfig") as mock_basic:
            setup_logging(cfg)
            _, kwargs = mock_basic.call_args
            handlers = kwargs["handlers"]
            assert any(
                isinstance(h, logging.handlers.RotatingFileHandler) for h in handlers
            )
            for h in handlers:
                h.close()

    def test_both_handlers_registered(self, tmp_path):
        cfg = _make_config(
            console_handler=True, file_handler=True, log_file_dir=str(tmp_path)
        )
        with patch("logging.basicConfig") as mock_basic:
            setup_logging(cfg)
            _, kwargs = mock_basic.call_args
            handlers = kwargs["handlers"]
            assert len(handlers) == 2
            for h in handlers:
                h.close()

    def test_no_handlers_when_both_disabled(self):
        cfg = _make_config(console_handler=False, file_handler=False)
        with patch("logging.basicConfig") as mock_basic:
            setup_logging(cfg)
            _, kwargs = mock_basic.call_args
            assert kwargs["handlers"] == []

    def test_root_logger_level_is_debug(self):
        cfg = _make_config()
        with patch("logging.basicConfig") as mock_basic:
            setup_logging(cfg)
            _, kwargs = mock_basic.call_args
            assert kwargs["level"] == "DEBUG"

    def test_force_true(self):
        cfg = _make_config()
        with patch("logging.basicConfig") as mock_basic:
            setup_logging(cfg)
            _, kwargs = mock_basic.call_args
            assert kwargs["force"] is True

    def test_h5py_logger_silenced(self):
        cfg = _make_config()
        with patch("logging.basicConfig"):
            setup_logging(cfg)
        assert logging.getLogger("h5py").level == logging.ERROR

    def test_urllib3_logger_silenced(self):
        cfg = _make_config()
        with patch("logging.basicConfig"):
            setup_logging(cfg)
        assert logging.getLogger("urllib3").level == logging.ERROR

    def test_mlflow_logger_silenced(self):
        cfg = _make_config()
        with patch("logging.basicConfig"):
            setup_logging(cfg)
        assert logging.getLogger("mlflow").level == logging.ERROR
