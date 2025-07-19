import logging
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.utils.helpers import setup_logging, ensure_directory, safe_division


class TestHelpers:
    """Test utility helper functions."""

    @patch('pathlib.Path.mkdir')
    @patch('logging.basicConfig')
    def test_setup_logging(self, mock_basic_config, mock_mkdir):
        """Test logging setup."""
        logger = setup_logging("test_logger", "DEBUG")
        
        assert logger.name == "test_logger"
        mock_mkdir.assert_called_once()
        mock_basic_config.assert_called_once()

    def test_ensure_directory(self, tmp_path):
        """Test directory creation."""
        test_dir = tmp_path / "test_directory"
        
        result = ensure_directory(test_dir)
        
        assert test_dir.exists()
        assert test_dir.is_dir()
        assert result == test_dir

    def test_safe_division_normal(self):
        """Test safe division with normal values."""
        result = safe_division(10, 2)
        assert result == 5.0

    def test_safe_division_by_zero(self):
        """Test safe division by zero returns default."""
        result = safe_division(10, 0)
        assert result == 0.0
        
        result = safe_division(10, 0, default=999)
        assert result == 999
