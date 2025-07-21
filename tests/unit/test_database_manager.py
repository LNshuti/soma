from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from src.utils.database import DatabaseManager


class TestDatabaseManager:
    """Test database manager functionality."""

    @pytest.fixture
    def mock_duckdb(self):
        """Mock DuckDB connection."""
        with patch("src.utils.database.duckdb") as mock:
            mock_conn = MagicMock()
            mock.connect.return_value = mock_conn
            yield mock, mock_conn

    def test_database_manager_initialization(self, mock_duckdb):
        """Test database manager initialization."""
        # Reset singleton to ensure clean state
        DatabaseManager._instance = None
        
        mock_duckdb_module, mock_conn = mock_duckdb

        db_manager = DatabaseManager("test.db")

        assert db_manager.db_path.name == "test.db"
        mock_duckdb_module.connect.assert_called()

    def test_singleton_pattern(self, mock_duckdb):
        """Test that DatabaseManager follows singleton pattern."""
        mock_duckdb_module, mock_conn = mock_duckdb

        db1 = DatabaseManager("test.db")
        db2 = DatabaseManager("test.db")

        assert db1 is db2

    def test_save_dataframe(self, mock_duckdb):
        """Test saving DataFrame to database."""
        mock_duckdb_module, mock_conn = mock_duckdb

        db_manager = DatabaseManager("test.db")
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})

        # Mock get_connection to return our mock connection
        with patch.object(db_manager, "get_connection", return_value=mock_conn):
            db_manager.save_dataframe(df, "test_table", "raw")

        expected_calls = [
            call("DROP TABLE IF EXISTS raw.test_table"),
            call("CREATE TABLE raw.test_table AS SELECT * FROM df"),
        ]
        mock_conn.execute.assert_has_calls(expected_calls)

    def test_fetch_dataframe(self, mock_duckdb):
        """Test fetching DataFrame from database."""
        mock_duckdb_module, mock_conn = mock_duckdb
        mock_result = MagicMock()
        mock_df = pd.DataFrame({"col1": [1, 2]})
        mock_result.df.return_value = mock_df
        mock_conn.execute.return_value = mock_result

        db_manager = DatabaseManager("test.db")

        # Mock get_connection to return our mock connection
        with patch.object(db_manager, "get_connection", return_value=mock_conn):
            result = db_manager.fetch_dataframe("SELECT * FROM test_table")

        assert result.equals(mock_df)
        mock_conn.execute.assert_called_with("SELECT * FROM test_table")

    def test_table_exists(self, mock_duckdb):
        """Test checking if table exists."""
        mock_duckdb_module, mock_conn = mock_duckdb

        db_manager = DatabaseManager("test.db")

        # Mock execute_query to return a result indicating table exists
        with patch.object(db_manager, "execute_query", return_value=[[1]]):
            result = db_manager.table_exists("test_table", "raw")

        assert result is True
