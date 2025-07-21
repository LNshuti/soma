import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.models.forecasting.demand_model import DemandForecastingModel
from src.models.recommendation.engine import RecommendationEngine
from src.utils.database import DatabaseManager


class TestModelTrainingPipeline:
    """Test complete model training pipeline."""

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
    def db_with_data(self, temp_db_path):
        """Create database with test data."""
        # Reset singleton to ensure clean state
        DatabaseManager._instance = None
        db = DatabaseManager(temp_db_path)

        # Create test data
        # Create publishers first
        publishers_df = pd.DataFrame(
            {
                "publisher_id": [f"PUB_{i:03d}" for i in range(1, 11)],
                "publisher_name": [f"Publisher {i}" for i in range(1, 11)],
                "publisher_type": (["Traditional", "Independent", "Academic"] * 4)[:10],
            }
        )
        
        books_df = pd.DataFrame(
            {
                "book_id": [f"BOOK_{i:03d}" for i in range(1, 101)],
                "title": [f"Book {i}" for i in range(1, 101)],
                "author": [f"Author {i}" for i in range(1, 101)],
                "publisher_id": ([f"PUB_{(i % 10) + 1:03d}" for i in range(100)]),
                "genre": (["Fiction", "Science", "Biography"] * 34)[:100],
                "price": ([19.99, 24.99, 29.99] * 34)[:100],
                "publication_year": ([2020, 2021, 2022] * 34)[:100],
                "format": (["Hardcover", "Paperback", "eBook"] * 34)[:100],
                "page_count": ([200, 300, 400] * 34)[:100],
            }
        )

        sales_df = pd.DataFrame(
            {
                "transaction_id": [f"TXN_{i:06d}" for i in range(1, 1001)],
                "book_id": [f"BOOK_{(i % 100) + 1:03d}" for i in range(1, 1001)],
                "sale_date": pd.date_range("2025-01-01", periods=1000, freq="h"),
                "quantity": ([1, 2, 3] * 334)[:1000],
                "unit_price": ([19.99, 24.99, 29.99] * 334)[:1000],
                "total_amount": ([19.99, 49.98, 89.97] * 334)[:1000],
                "customer_type": (["Individual", "Business", "Educational"] * 334)[:1000],
                "channel": (["Online", "Retail", "Wholesale"] * 334)[:1000],
                "region": (["North", "South", "East"] * 334)[:1000],
            }
        )

        db.save_dataframe(publishers_df, "publishers", "raw")
        db.save_dataframe(books_df, "books", "raw")
        db.save_dataframe(sales_df, "sales", "raw")

        return db

    def test_demand_forecasting_full_pipeline(self, db_with_data, temp_artifacts_dir):
        """Test complete demand forecasting pipeline."""
        model = DemandForecastingModel(
            model_path=str(temp_artifacts_dir / "demand_model.pkl")
        )
        model.db_manager = db_with_data

        # Test data loading
        df = model.load_data()
        assert not df.empty

        # Test feature preparation
        X, y = model.prepare_features(df)
        assert not X.empty
        assert not y.empty

        # Test training
        metrics = model.train(X, y)
        assert isinstance(metrics, dict)
        assert "mae" in metrics
        assert "rmse" in metrics

        # Test prediction
        predictions = model.predict(X.head(5))
        assert len(predictions) == 5

        # Test book-specific prediction
        result = model.predict_for_book("BOOK_001", 7)
        assert "predictions" in result
        assert len(result["predictions"]) == 7

    def test_recommendation_engine_full_pipeline(
        self, db_with_data, temp_artifacts_dir
    ):
        """Test complete recommendation pipeline."""
        engine = RecommendationEngine(db_with_data)

        # Test training
        metrics = engine.train()
        assert isinstance(metrics, dict)
        assert "num_users" in metrics
        assert "num_items" in metrics

        # Test similar books recommendation
        result = engine.get_similar_books("BOOK_001", 5)
        assert isinstance(result, dict)

        # Test popular books by user type
        result = engine.get_popular_by_user_type("Individual", 5)
        assert isinstance(result, dict)
