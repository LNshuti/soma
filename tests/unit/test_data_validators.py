import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.data.validators import DataValidator


class TestDataValidator:
    """Test data validation."""

    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager."""
        mock_db = MagicMock()
        return mock_db

    @pytest.fixture
    def data_validator(self, mock_db_manager):
        """Create data validator."""
        return DataValidator(mock_db_manager)

    def test_validate_table_success(self, data_validator):
        """Test successful table validation."""
        # Mock valid data
        valid_df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "value": [10, 20, 30]
        })
        
        data_validator.db.table_exists.return_value = True
        data_validator.db.get_table.return_value = valid_df
        
        result = data_validator._validate_table("test_table")
        
        assert result["overall_status"] == "PASS"
        assert result["record_count"] == 3
        assert result["column_count"] == 3

    def test_validate_table_with_issues(self, data_validator):
        """Test table validation with data quality issues."""
        # Mock data with nulls
        invalid_df = pd.DataFrame({
            "id": [1, 2, None],
            "name": ["A", None, "C"],
            "value": [10, 20, 30]
        })
        
        data_validator.db.table_exists.return_value = True
        data_validator.db.get_table.return_value = invalid_df
        
        result = data_validator._validate_table("test_table")
        
        assert result["overall_status"] == "FAIL"
        assert len(result["issues"]) > 0

    def test_validate_table_not_exists(self, data_validator):
        """Test validation when table doesn't exist."""
        data_validator.db.table_exists.return_value = False
        
        result = data_validator._validate_table("nonexistent_table")
        
        assert result["overall_status"] == "FAIL"
        assert "Table does not exist" in result["issues"]

    def test_validate_all_tables(self, data_validator):
        """Test validating all tables."""
        # Mock successful validation for all tables
        data_validator.db.table_exists.return_value = True
        data_validator.db.get_table.return_value = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })
        
        results = data_validator.validate_all_tables()
        
        assert isinstance(results, dict)
        assert len(results) == 5  # Number of tables being validated
        assert all(table in results for table in 
                  ["publishers", "books", "sales", "inventory", "campaign_events"])
