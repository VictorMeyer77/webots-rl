"""
Unit tests for configuration management module.

Tests cover:
- LogLevel enum validation
- ConfigItem dataclass initialization
- Config class initialization with different prefixes
- Environment variable loading and type casting
- Caching behavior
- Dictionary-style access (__getitem__, __contains__)
- Error handling for invalid keys and type casting
- Boolean value parsing
- Default value returns
"""

import os
import pytest
from unittest.mock import patch

from corl.utils.config import Config, ConfigItem, LogLevel


class TestLogLevel:
    """Test LogLevel enum."""

    def test_log_level_values(self):
        """Test that all log levels have correct string values."""
        assert LogLevel.DEBUG == "DEBUG"
        assert LogLevel.INFO == "INFO"
        assert LogLevel.WARNING == "WARNING"
        assert LogLevel.ERROR == "ERROR"
        assert LogLevel.CRITICAL == "CRITICAL"

    def test_log_level_is_string(self):
        """Test that LogLevel instances are strings."""
        assert isinstance(LogLevel.INFO, str)
        assert isinstance(LogLevel.DEBUG, str)

    def test_log_level_comparison(self):
        """Test LogLevel string comparison."""
        assert LogLevel.INFO == "INFO"
        assert LogLevel.DEBUG != "INFO"

    def test_log_level_iteration(self):
        """Test iterating over all log levels."""
        levels = [level for level in LogLevel]
        assert len(levels) == 5
        assert LogLevel.DEBUG in levels
        assert LogLevel.CRITICAL in levels


class TestConfigItem:
    """Test ConfigItem dataclass."""

    def test_minimal_initialization(self):
        """Test ConfigItem with only default value."""
        item = ConfigItem(default="test")
        assert item.default == "test"
        assert item.cast is None
        assert item.description == ""

    def test_full_initialization(self):
        """Test ConfigItem with all fields."""
        item = ConfigItem(default=8000, cast=int, description="Port number")
        assert item.default == 8000
        assert item.cast is int
        assert item.description == "Port number"

    def test_none_default(self):
        """Test ConfigItem with None as default."""
        item = ConfigItem(default=None, cast=str, description="Optional value")
        assert item.default is None
        assert item.cast is str

    def test_bool_type(self):
        """Test ConfigItem with bool type."""
        item = ConfigItem(default=True, cast=bool, description="Flag")
        assert item.default is True
        assert item.cast is bool

    def test_loglevel_type(self):
        """Test ConfigItem with LogLevel type."""
        item = ConfigItem(default="INFO", cast=LogLevel, description="Log level")
        assert item.default == "INFO"
        assert item.cast is LogLevel


class TestConfigInitialization:
    """Test Config class initialization."""

    def test_default_prefix(self):
        """Test Config initialization with default prefix."""
        config = Config()
        assert config.prefix == "webots"
        assert isinstance(config._cache, dict)
        assert len(config._cache) == 0

    def test_custom_prefix(self):
        """Test Config initialization with custom prefix."""
        config = Config(prefix="myapp")
        assert config.prefix == "myapp"
        assert isinstance(config._cache, dict)

    def test_empty_prefix(self):
        """Test Config initialization with empty prefix."""
        config = Config(prefix="")
        assert config.prefix == ""

    @patch("corl.utils.config.load_dotenv")
    def test_load_dotenv_called(self, mock_load_dotenv):
        """Test that load_dotenv is called during initialization."""
        Config()
        mock_load_dotenv.assert_called_once()


class TestConfigGet:
    """Test Config.get() method."""

    def test_get_default_value(self):
        """Test getting a config value that returns default."""
        config = Config()
        # No environment variable set, should return default
        host = config.get("API_HOST")
        assert host == "http://localhost"

    def test_get_default_int_value(self):
        """Test getting a default integer value."""
        config = Config()
        port = config.get("API_PORT")
        assert port == 8000
        assert isinstance(port, int)

    def test_get_default_bool_value(self):
        """Test getting a default boolean value."""
        config = Config()
        console_handler = config.get("LOG_CONSOLE_HANDLER")
        assert console_handler is True
        assert isinstance(console_handler, bool)

    @patch.dict(os.environ, {"WEBOTS_API_HOST": "http://example.com"})
    def test_get_from_environment(self):
        """Test getting a config value from environment variable."""
        config = Config()
        host = config.get("API_HOST")
        assert host == "http://example.com"

    @patch.dict(os.environ, {"WEBOTS_API_PORT": "9000"})
    def test_get_int_from_environment(self):
        """Test getting and casting integer from environment."""
        config = Config()
        port = config.get("API_PORT")
        assert port == 9000
        assert isinstance(port, int)

    @patch.dict(os.environ, {"WEBOTS_LOG_CONSOLE_HANDLER": "true"})
    def test_get_bool_true_from_environment(self):
        """Test getting boolean true from environment."""
        config = Config()
        handler = config.get("LOG_CONSOLE_HANDLER")
        assert handler is True

    @patch.dict(os.environ, {"WEBOTS_LOG_CONSOLE_HANDLER": "false"})
    def test_get_bool_false_from_environment(self):
        """Test getting boolean false from environment."""
        config = Config()
        handler = config.get("LOG_CONSOLE_HANDLER")
        assert handler is False

    @patch.dict(os.environ, {"WEBOTS_TRAIN": "1"})
    def test_get_bool_one_from_environment(self):
        """Test getting boolean from '1' string."""
        config = Config()
        train = config.get("TRAIN")
        assert train is True

    @patch.dict(os.environ, {"WEBOTS_TRAIN": "0"})
    def test_get_bool_zero_from_environment(self):
        """Test getting boolean from '0' string."""
        config = Config()
        train = config.get("TRAIN")
        assert train is False

    @patch.dict(os.environ, {"WEBOTS_TRAIN": "yes"})
    def test_get_bool_yes_from_environment(self):
        """Test getting boolean from 'yes' string."""
        config = Config()
        train = config.get("TRAIN")
        assert train is True

    @patch.dict(os.environ, {"WEBOTS_TRAIN": "on"})
    def test_get_bool_on_from_environment(self):
        """Test getting boolean from 'on' string."""
        config = Config()
        train = config.get("TRAIN")
        assert train is True

    @patch.dict(os.environ, {"WEBOTS_LOG_CONSOLE_LEVEL": "DEBUG"})
    def test_get_loglevel_from_environment(self):
        """Test getting LogLevel enum from environment."""
        config = Config()
        level = config.get("LOG_CONSOLE_LEVEL")
        assert level == LogLevel.DEBUG
        assert isinstance(level, LogLevel)

    @patch.dict(os.environ, {"CUSTOM_API_HOST": "http://custom.com"})
    def test_get_with_custom_prefix(self):
        """Test getting config value with custom prefix."""
        config = Config(prefix="CUSTOM")
        host = config.get("API_HOST")
        assert host == "http://custom.com"

    def test_get_case_insensitive(self):
        """Test that get() is case-insensitive."""
        config = Config()
        assert config.get("api_host") == config.get("API_HOST")
        assert config.get("Api_Host") == config.get("API_HOST")

    def test_get_invalid_key(self):
        """Test getting an invalid key raises ValueError."""
        config = Config()
        with pytest.raises(ValueError) as exc_info:
            config.get("INVALID_KEY")
        assert "not defined in DEFAULTS" in str(exc_info.value)

    @patch.dict(os.environ, {"WEBOTS_API_PORT": "invalid"})
    def test_get_invalid_int_cast(self):
        """Test that invalid integer cast raises ValueError with details."""
        config = Config()
        with pytest.raises(ValueError) as exc_info:
            config.get("API_PORT")
        error_msg = str(exc_info.value)
        assert "Failed to cast" in error_msg
        assert "WEBOTS_API_PORT=invalid" in error_msg
        assert "Default value:" in error_msg
        assert "Description:" in error_msg

    @patch.dict(os.environ, {"WEBOTS_LOG_CONSOLE_LEVEL": "INVALID_LEVEL"})
    def test_get_invalid_loglevel_cast(self):
        """Test that invalid LogLevel cast raises ValueError."""
        config = Config()
        with pytest.raises(ValueError) as exc_info:
            config.get("LOG_CONSOLE_LEVEL")
        assert "Failed to cast" in str(exc_info.value)


class TestConfigCaching:
    """Test Config caching behavior."""

    @patch.dict(os.environ, {"WEBOTS_API_HOST": "http://cached.com"})
    def test_value_cached_after_first_get(self):
        """Test that values are cached after first retrieval."""
        config = Config()

        # First call should cache the value
        first_call = config.get("API_HOST")
        assert "API_HOST" in config._cache
        assert config._cache["API_HOST"] == "http://cached.com"

        # Second call should return cached value
        second_call = config.get("API_HOST")
        assert first_call == second_call

    def test_default_value_cached(self):
        """Test that default values are cached."""
        config = Config()

        # Get default value
        _ = config.get("API_PORT")
        assert "API_PORT" in config._cache
        assert config._cache["API_PORT"] == 8000

    @patch.dict(os.environ, {})
    def test_cache_persists_across_calls(self):
        """Test that cache persists across multiple calls."""
        config = Config()

        # Call get multiple times
        for _ in range(3):
            config.get("API_HOST")

        # Should only have one cached entry
        assert config._cache["API_HOST"] == "http://localhost"

    @patch.dict(os.environ, {"WEBOTS_API_PORT": "9000"})
    def test_cache_with_environment_variable(self):
        """Test caching with environment variable override."""
        config = Config()
        port = config.get("API_PORT")

        assert port == 9000
        assert config._cache["API_PORT"] == 9000


class TestConfigDictAccess:
    """Test dictionary-style access to Config."""

    def test_getitem_method(self):
        """Test __getitem__ method for dict-style access."""
        config = Config()
        host = config["API_HOST"]
        assert host == "http://localhost"

    @patch.dict(os.environ, {"WEBOTS_API_PORT": "9000"})
    def test_getitem_with_environment(self):
        """Test __getitem__ with environment variable."""
        config = Config()
        port = config["API_PORT"]
        assert port == 9000

    def test_getitem_invalid_key(self):
        """Test __getitem__ with invalid key raises ValueError."""
        config = Config()
        with pytest.raises(ValueError):
            _ = config["INVALID_KEY"]

    def test_getitem_case_insensitive(self):
        """Test __getitem__ is case-insensitive."""
        config = Config()
        assert config["api_host"] == config["API_HOST"]

    def test_contains_valid_key(self):
        """Test __contains__ with valid key."""
        config = Config()
        assert "API_HOST" in config
        assert "API_PORT" in config
        assert "TRAIN" in config

    def test_contains_invalid_key(self):
        """Test __contains__ with invalid key."""
        config = Config()
        assert "INVALID_KEY" not in config
        assert "RANDOM_CONFIG" not in config

    def test_contains_case_insensitive(self):
        """Test __contains__ is case-insensitive."""
        config = Config()
        assert "api_host" in config
        assert "Api_Host" in config
        assert "API_HOST" in config

    def test_contains_does_not_check_environment(self):
        """Test __contains__ checks DEFAULTS, not environment variables."""
        config = Config()
        # Even if not set in environment, should return True if in DEFAULTS
        assert "BIN_PATH" in config


class TestConfigAllDefaults:
    """Test that all default config items are accessible."""

    def test_all_api_configs(self):
        """Test all API configuration items."""
        config = Config()
        assert config.get("API_HOST") == "http://localhost"
        assert config.get("API_PORT") == 8000

    def test_all_logging_configs(self):
        """Test all logging configuration items."""
        # Create fresh config instance to avoid cached values
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.get("LOG_CONSOLE_LEVEL") == "INFO"
            assert config.get("LOG_CONSOLE_HANDLER") is True
            assert config.get("LOG_FILE_LEVEL") == "INFO"
            assert config.get("LOG_FILE_HANDLER") is False
            assert config.get("LOG_FILE_DIR") == ".log"

    def test_all_trainer_configs(self):
        """Test all trainer configuration items."""
        config = Config()
        assert config.get("TRAIN") is False
        assert config.get("TRAIN_ID") is None
        assert config.get("WORKER_ID") is None
        assert config.get("TRAINER_TENSORBOARD_PATH") == ".train/tensorboard/"
        assert config.get("TRAINER_MODEL_DIR") == ".train/models/"
        assert config.get("TRAINER_MAX_WORKER") == 4

    def test_all_webots_configs(self):
        """Test all Webots configuration items."""
        config = Config()
        assert (
            config.get("BIN_PATH") == "/Applications/Webots.app/Contents/MacOS/webots"
        )
        assert config.get("WORLD_PATH") == "projects/worlds"
        assert config.get("CONTROLLER_PATH") == "projects/controllers"


class TestConfigEdgeCases:
    """Test edge cases and special scenarios."""

    @patch.dict(os.environ, {"WEBOTS_TRAIN_ID": ""})
    def test_empty_string_environment_variable(self):
        """Test handling of empty string environment variable."""
        config = Config()
        train_id = config.get("TRAIN_ID")
        # Empty string should be returned, not the default
        assert train_id == ""

    def test_none_default_values(self):
        """Test config items with None as default."""
        config = Config()
        assert config.get("TRAIN_ID") is None
        assert config.get("WORKER_ID") is None

    @patch.dict(os.environ, {"WEBOTS_WORKER_ID": "42"})
    def test_optional_value_with_environment(self):
        """Test optional value (None default) with environment variable."""
        config = Config()
        worker_id = config.get("WORKER_ID")
        assert worker_id == 42

    @patch.dict(os.environ, {"WEBOTS_API_PORT": "  8080  "})
    def test_whitespace_in_environment_value(self):
        """Test that whitespace in environment values is handled."""
        config = Config()
        # int() should handle whitespace
        port = config.get("API_PORT")
        assert port == 8080

    def test_multiple_config_instances(self):
        """Test multiple Config instances with different prefixes."""
        config1 = Config(prefix="app1")
        config2 = Config(prefix="app2")

        assert config1.prefix == "app1"
        assert config2.prefix == "app2"
        # Each should have independent caches
        assert config1._cache is not config2._cache


class TestConfigIntegration:
    """Integration tests for Config class."""

    @patch.dict(
        os.environ,
        {
            "WEBOTS_API_HOST": "http://production.com",
            "WEBOTS_API_PORT": "443",
            "WEBOTS_TRAIN": "true",
            "WEBOTS_LOG_CONSOLE_LEVEL": "ERROR",
            "WEBOTS_TRAINER_MAX_WORKER": "8",
        },
    )
    def test_multiple_environment_overrides(self):
        """Test multiple environment variables overriding defaults."""
        config = Config()

        assert config.get("API_HOST") == "http://production.com"
        assert config.get("API_PORT") == 443
        assert config.get("TRAIN") is True
        assert config.get("LOG_CONSOLE_LEVEL") == LogLevel.ERROR
        assert config.get("TRAINER_MAX_WORKER") == 8

    @patch.dict(os.environ, {"WEBOTS_API_HOST": "http://test.com"})
    def test_mixed_access_methods(self):
        """Test using both get() and [] access methods."""
        config = Config()

        # Access same value with different methods
        host1 = config.get("API_HOST")
        host2 = config["API_HOST"]

        assert host1 == host2 == "http://test.com"
        # Should be same cached value
        assert config._cache["API_HOST"] == "http://test.com"

    def test_check_before_access_pattern(self):
        """Test the pattern of checking key existence before access."""
        config = Config()

        if "API_HOST" in config:
            host = config["API_HOST"]
            assert host is not None

        if "NONEXISTENT" not in config:
            # This should execute
            assert True
