"""Unit tests for application configuration settings."""

import pytest
from app.core.config import Settings, settings
from pydantic import ValidationError


def test_settings_default_api_host():
    """Test that api_host has correct default value."""
    config = Settings()

    assert config.api_host == "0.0.0.0"


def test_settings_default_api_port():
    """Test that api_port has correct default value."""
    config = Settings()

    assert config.api_port == 8000


def test_settings_custom_api_host(monkeypatch):
    """Test that api_host can be overridden via environment variable."""
    monkeypatch.setenv("API_HOST", "localhost")

    config = Settings()

    assert config.api_host == "localhost"


def test_settings_custom_api_port(monkeypatch):
    """Test that api_port can be overridden via environment variable."""
    monkeypatch.setenv("API_PORT", "8080")

    config = Settings()

    assert config.api_port == 8080


def test_settings_invalid_api_port_type(monkeypatch):
    """Test that invalid api_port type raises validation error."""
    monkeypatch.setenv("API_PORT", "not_a_number")

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    assert "api_port" in str(exc_info.value)


def test_settings_negative_api_port(monkeypatch):
    """Test that negative api_port is accepted but may not be valid for server."""
    monkeypatch.setenv("API_PORT", "-1")

    config = Settings()

    assert config.api_port == -1


def test_settings_zero_api_port(monkeypatch):
    """Test that zero api_port is accepted (OS assigns random port)."""
    monkeypatch.setenv("API_PORT", "0")

    config = Settings()

    assert config.api_port == 0


def test_settings_api_host_ipv6(monkeypatch):
    """Test that api_host accepts IPv6 addresses."""
    monkeypatch.setenv("API_HOST", "::")

    config = Settings()

    assert config.api_host == "::"


def test_settings_api_port_max_value(monkeypatch):
    """Test that api_port accepts maximum valid port number."""
    monkeypatch.setenv("API_PORT", "65535")

    config = Settings()

    assert config.api_port == 65535


def test_settings_custom_memory_capacity(monkeypatch):
    """Test that memory capacity can be overridden via environment variable."""
    monkeypatch.setenv("MEMORY_CAPACITY", "50000")

    config = Settings()

    assert config.memory_capacity == 50000


def test_settings_custom_log_console_level(monkeypatch):
    """Test that console log level can be overridden via environment variable."""
    monkeypatch.setenv("LOG_CONSOLE_LEVEL", "DEBUG")

    config = Settings()

    assert config.log_console_level == "DEBUG"


def test_settings_custom_log_console_handler(monkeypatch):
    """Test that console log handler can be disabled via environment variable."""
    monkeypatch.setenv("LOG_CONSOLE_HANDLER", "False")

    config = Settings()

    assert config.log_console_handler is False


def test_settings_custom_log_file_level(monkeypatch):
    """Test that file log level can be overridden via environment variable."""
    monkeypatch.setenv("LOG_FILE_LEVEL", "WARNING")

    config = Settings()

    assert config.log_file_level == "WARNING"


def test_settings_custom_log_file_handler(monkeypatch):
    """Test that file log handler can be enabled via environment variable."""
    monkeypatch.setenv("LOG_FILE_HANDLER", "True")

    config = Settings()

    assert config.log_file_handler is True


def test_settings_custom_log_file_dir(monkeypatch):
    """Test that log file directory can be overridden via environment variable."""
    monkeypatch.setenv("LOG_FILE_DIR", "/var/log/app")

    config = Settings()

    assert config.log_file_dir == "/var/log/app"


def test_settings_all_custom_values(monkeypatch):
    """Test that all settings can be overridden simultaneously."""
    monkeypatch.setenv("MEMORY_CAPACITY", "25000")
    monkeypatch.setenv("LOG_CONSOLE_LEVEL", "ERROR")
    monkeypatch.setenv("LOG_CONSOLE_HANDLER", "False")
    monkeypatch.setenv("LOG_FILE_LEVEL", "INFO")
    monkeypatch.setenv("LOG_FILE_HANDLER", "True")
    monkeypatch.setenv("LOG_FILE_DIR", "/custom/log/path")

    config = Settings()

    assert config.memory_capacity == 25000
    assert config.log_console_level == "ERROR"
    assert config.log_console_handler is False
    assert config.log_file_level == "INFO"
    assert config.log_file_handler is True
    assert config.log_file_dir == "/custom/log/path"


def test_settings_invalid_memory_capacity_type(monkeypatch):
    """Test that invalid memory capacity type raises validation error."""
    monkeypatch.setenv("MEMORY_CAPACITY", "invalid_number")

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    assert "memory_capacity" in str(exc_info.value)


def test_settings_negative_memory_capacity(monkeypatch):
    """Test that negative memory capacity is accepted but not recommended."""
    monkeypatch.setenv("MEMORY_CAPACITY", "-1000")

    config = Settings()

    assert config.memory_capacity == -1000


def test_settings_zero_memory_capacity(monkeypatch):
    """Test that zero memory capacity is accepted."""
    monkeypatch.setenv("MEMORY_CAPACITY", "0")

    config = Settings()

    assert config.memory_capacity == 0


def test_settings_invalid_boolean_type(monkeypatch):
    """Test that invalid boolean values are handled gracefully."""
    monkeypatch.setenv("LOG_CONSOLE_HANDLER", "not_a_boolean")

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    assert "log_console_handler" in str(exc_info.value)


def test_settings_case_insensitive_boolean(monkeypatch):
    """Test that boolean environment variables are case-insensitive."""
    monkeypatch.setenv("LOG_FILE_HANDLER", "TRUE")

    config = Settings()

    assert config.log_file_handler is True


def test_settings_env_file_config():
    """Test that Settings has correct env_file configuration."""
    assert Settings.model_config["env_file"] == ".env"


def test_global_settings_instance():
    """Test that global settings instance is correctly initialized."""
    assert isinstance(settings, Settings)
    assert settings.memory_capacity == 500


def test_settings_immutability():
    """Test that settings values can be modified after creation."""
    config = Settings()
    original_capacity = config.memory_capacity

    config.memory_capacity = 5000

    assert config.memory_capacity == 5000
    assert config.memory_capacity != original_capacity


def test_settings_large_memory_capacity(monkeypatch):
    """Test that very large memory capacity values are accepted."""
    monkeypatch.setenv("MEMORY_CAPACITY", "10000000")

    config = Settings()

    assert config.memory_capacity == 10000000


def test_settings_empty_log_file_dir(monkeypatch):
    """Test that empty log file directory is accepted."""
    monkeypatch.setenv("LOG_FILE_DIR", "")

    config = Settings()

    assert config.log_file_dir == ""


def test_settings_multiple_instances_independence():
    """Test that multiple Settings instances are independent."""
    config1 = Settings()
    config2 = Settings()

    config1.memory_capacity = 1000
    config2.memory_capacity = 2000

    assert config1.memory_capacity == 1000
    assert config2.memory_capacity == 2000
