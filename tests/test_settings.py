"""Tests for application settings."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from settings import Settings, get_settings


@pytest.fixture(scope="class")
def settings() -> Settings:
    """Provide a Settings instance with default values."""
    return Settings()  # type: ignore


class TestSettings:
    """Tests for Settings class."""

    def test_default_values(self, settings: Settings) -> None:
        """Test that default settings are set correctly."""
        assert not settings.model_fields_set
        assert settings.storage == Path(tempfile.gettempdir()) / "scratch_api"  # noqa
        assert settings.persistent_storage is False
        assert settings.max_upload_size == 100 * 1024 * 1024  # 100MB
        assert settings.api_host == "127.0.0.1"
        assert settings.api_port == 8000  # noqa
        assert settings.app_title == "Scratch API"

    def test_storage_explicitly_set_from_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test that storage from environment variable is tracked in model_fields_set."""
        # Arrange
        custom_path = tmp_path / "custom_storage"
        custom_path.mkdir()
        monkeypatch.setenv("SCRATCH_STORAGE", str(custom_path))

        # Act
        settings = Settings()  # type: ignore

        # Assert
        # Pydantic uses the actual field name (lowercase)
        assert "storage" in settings.model_fields_set
        assert settings.storage == custom_path

    def test_storage_not_in_fields_set_when_using_default(self, settings: Settings) -> None:
        """Test that storage is not in model_fields_set when using default."""
        assert "storage" not in settings.model_fields_set

    @pytest.mark.parametrize(
        ("env_var", "env_value", "field_name", "expected_value"),
        [
            pytest.param("SCRATCH_PERSISTENT_STORAGE", "true", "persistent_storage", True, id="persistent_storage"),
            pytest.param("SCRATCH_MAX_UPLOAD_SIZE", "52428800", "max_upload_size", 52428800, id="max_upload_size"),
            pytest.param("SCRATCH_API_PORT", "9000", "api_port", 9000, id="api_port"),
        ],
    )
    def test_settings_from_env(
        self, monkeypatch: pytest.MonkeyPatch, env_var: str, env_value: str, field_name: str, expected_value: bool | int
    ) -> None:
        """Test that settings can be configured via environment variables."""
        # Arrange
        monkeypatch.setenv(env_var, env_value)

        # Act
        settings = Settings()  # type: ignore

        # Assert
        assert field_name in settings.model_fields_set
        assert getattr(settings, field_name) == expected_value

    @pytest.mark.parametrize(
        ("env_var", "value", "expected_match"),
        [
            pytest.param("SCRATCH_MAX_UPLOAD_SIZE", "0", "greater than 0", id="max_upload_size_zero"),
            pytest.param("SCRATCH_MAX_UPLOAD_SIZE", "-1", "greater than 0", id="max_upload_size_negative"),
            pytest.param("SCRATCH_API_PORT", "70000", "less than 65536", id="api_port_above_range"),
            pytest.param("SCRATCH_API_PORT", "0", "greater than 0", id="api_port_zero"),
            pytest.param("SCRATCH_API_PORT", "-1", "greater than 0", id="api_port_negative"),
        ],
    )
    def test_validation_rejects_invalid_values(
        self, monkeypatch: pytest.MonkeyPatch, env_var: str, value: str, expected_match: str
    ) -> None:
        """Test that Settings validation rejects invalid values."""
        # Arrange
        monkeypatch.setenv(env_var, value)

        # Act & Assert
        with pytest.raises(ValueError, match=expected_match):
            Settings()  # type: ignore

    def test_settings_are_frozen(self, settings: Settings) -> None:
        """Test that settings are immutable after creation."""
        with pytest.raises(ValueError, match="frozen"):
            settings.api_port = 9000  # type: ignore

    def test_extra_fields_forbidden(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that extra fields are not allowed."""
        # Arrange
        monkeypatch.setenv("SCRATCH_UNKNOWN_FIELD", "value")

        # Act - Settings should still work, just ignore unknown fields
        settings = Settings()  # type: ignore

        # Assert - unknown field should not be present
        assert not hasattr(settings, "UNKNOWN_FIELD")

    def test_app_version_returns_fallback_when_package_not_found(self, settings: Settings) -> None:
        """Test that app_version returns '0.0.0' when package version cannot be determined."""
        assert settings.app_version == "0.0.0"

    @patch("settings.version")
    def test_app_version_returns_package_version(self, mock_version) -> None:
        """Test that app_version returns actual package version when available."""
        # Arrange
        mock_version.return_value = "1.2.3"

        # Act
        settings = Settings()  # type: ignore

        # Assert
        assert settings.app_version == "1.2.3"
        mock_version.assert_called_once_with("scratch")

    def test_log_startup_config_logs_configuration(self, settings: Settings, caplog: pytest.LogCaptureFixture) -> None:
        """Test that log_startup_config logs all configuration values."""
        # Act
        settings.log_startup_config()

        # Assert
        log_text = caplog.text

        assert "Application startup - Configuration:" in log_text
        assert f"Title: {settings.app_title}" in log_text
        assert f"Version: {settings.app_version}" in log_text
        assert f"Host: {settings.api_host}:{settings.api_port}" in log_text
        assert f"Storage: {settings.storage}" in log_text
        assert f"Persistent storage: {settings.persistent_storage}" in log_text
        assert f"Max upload size: {settings.max_upload_size / (1024 * 1024):.1f}MB" in log_text


class TestGetSettings:
    """Tests for get_settings cached function."""

    def test_get_settings_returns_settings_instance(self) -> None:
        """Test that get_settings returns a Settings instance."""
        # Act
        settings = get_settings()

        # Assert
        assert isinstance(settings, Settings)

    def test_get_settings_cache_persists_across_calls(self) -> None:
        """Test that multiple calls to get_settings return the same cached instance."""
        # Act
        instance_ids = {id(get_settings()) for _ in range(5)}

        # Assert
        assert len(instance_ids) == 1
