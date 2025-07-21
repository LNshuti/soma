import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.generators import ContentDataGenerator, DataGenerationConfig
from src.utils.database import DatabaseManager


class TestDatabaseOperations:
    """Test database operations integration."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database."""
        temp_file = tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False)
        temp_file.close()  # Close but don't delete
        # Remove the empty file so duckdb can create a fresh one
        Path(temp_file.name).unlink(missing_ok=True)
        yield temp_file.name
        Path(temp_file.name).unlink(missing_ok=True)

    @pytest.fixture
    def db_manager(self, temp_db_path):
        """Create database manager."""
        # Reset singleton to ensure clean state
        DatabaseManager._instance = None
        return DatabaseManager(temp_db_path)

    def test_database_schema_creation(self, db_manager):
        """Test database schema creation."""
        conn = db_manager.get_connection()

        # Check schemas exist
        schemas = conn.execute("SELECT schema_name FROM information_schema.schemata").fetchall()
        schema_names = [schema[0] for schema in schemas]

        assert "raw" in schema_names
        assert "staging" in schema_names
        assert "analytics" in schema_names

    def test_data_generation_and_storage(self, db_manager):
        """Test complete data generation and storage."""
        config = DataGenerationConfig(
            n_publishers=10, n_books=20, n_sales=50, n_inventory=30, n_campaigns=15
        )

        generator = ContentDataGenerator(config)

        # Generate publishers
        publisher_count = generator._generate_publishers(db_manager)
        assert publisher_count == 10
        assert db_manager.table_exists("publishers", "raw")

        # Generate books
        book_count = generator._generate_books(db_manager)
        assert book_count == 20
        assert db_manager.table_exists("books", "raw")

        # Verify data integrity
        publishers_df = db_manager.get_table("publishers", "raw")
        books_df = db_manager.get_table("books", "raw")

        assert len(publishers_df) == 10
        assert len(books_df) == 20
        assert set(books_df["publisher_id"]).issubset(
            set(publishers_df["publisher_id"])
        )

    def test_table_operations(self, db_manager):
        """Test table operations."""
        # Create test data
        test_df = pd.DataFrame(
            {"id": [1, 2, 3], "name": ["A", "B", "C"], "value": [10, 20, 30]}
        )

        # Save DataFrame
        db_manager.save_dataframe(test_df, "test_table", "raw")

        # Check table exists
        assert db_manager.table_exists("test_table", "raw")

        # Retrieve data
        retrieved_df = db_manager.get_table("test_table", "raw")

        assert len(retrieved_df) == 3
        assert list(retrieved_df.columns) == ["id", "name", "value"]

    def test_query_operations(self, db_manager):
        """Test query operations."""
        # Create test data
        test_df = pd.DataFrame(
            {
                "book_id": ["BOOK_001", "BOOK_002", "BOOK_003"],
                "title": ["Book A", "Book B", "Book C"],
                "price": [19.99, 24.99, 29.99],
            }
        )

        db_manager.save_dataframe(test_df, "books", "raw")

        # Test parameterized query
        result = db_manager.fetch_dataframe(
            "SELECT * FROM raw.books WHERE price > ?", [20.00]
        )

        assert len(result) == 2
        assert all(result["price"] > 20.00)

    def test_concurrent_access(self, db_manager):
        """Test concurrent database access."""
        import threading
        import time

        results = []

        def write_data(thread_id):
            df = pd.DataFrame(
                {
                    "thread_id": [thread_id] * 5,
                    "data": [f"data_{thread_id}_{i}" for i in range(5)],
                }
            )
            db_manager.save_dataframe(df, f"thread_table_{thread_id}", "raw")
            results.append(thread_id)

        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=write_data, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify all threads completed
        assert len(results) == 3
        assert set(results) == {0, 1, 2}

        # Verify all tables were created
        for i in range(3):
            assert db_manager.table_exists(f"thread_table_{i}", "raw")
