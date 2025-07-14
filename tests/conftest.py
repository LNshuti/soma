import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.utils.database import DatabaseManager


@pytest.fixture(scope="session")
def test_db():
    """Test database fixture"""
    with tempfile.NamedTemporaryFile(suffix=".duckdb") as f:
        db_manager = DatabaseManager(f.name)
        yield db_manager


@pytest.fixture
def sample_data():
    """Sample data for testing"""
    return pd.DataFrame(
        {
            "book_id": ["BOOK_001", "BOOK_002"],
            "daily_quantity": [10, 15],
            "sale_date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
        }
    )


@pytest.fixture
def temp_artifacts_dir():
    """Temporary artifacts directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)
