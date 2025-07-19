# tests/unit/test_config.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from config.settings import Settings, get_settings


class TestSettings:
    """Test configuration settings."""

    def test_settings_initialization(self):
        """Test settings can be initialized with defaults."""
        settings = Settings()
        
        assert settings.DB_PATH == "./data/soma.duckdb"
        assert settings.API_PORT == 5001
        assert settings.WEB_PORT == 7860
        assert settings.ENVIRONMENT == "development"
        assert settings.DEBUG is True

    def test_settings_from_env(self, monkeypatch):
        """Test settings can be loaded from environment variables."""
        monkeypatch.setenv("DB_PATH", "/custom/path/db.duckdb")
        monkeypatch.setenv("API_PORT", "8080")
        monkeypatch.setenv("ENVIRONMENT", "production")
        
        settings = Settings()
        
        assert settings.DB_PATH == "/custom/path/db.duckdb"
        assert settings.API_PORT == 8080
        assert settings.ENVIRONMENT == "production"

    @patch('pathlib.Path.mkdir')
    def test_directory_creation(self, mock_mkdir):
        """Test that directories are created during initialization."""
        settings = Settings()
        settings.model_post_init(None)
        
        # Verify mkdir was called for required directories
        assert mock_mkdir.called

    def test_get_settings_singleton(self):
        """Test that get_settings returns the same instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2
