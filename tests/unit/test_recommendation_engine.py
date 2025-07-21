from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.recommendation.engine import RecommendationEngine


class TestRecommendationEngine:
    """Test recommendation engine."""

    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager."""
        mock_db = MagicMock()

        # Mock interactions data
        interactions_df = pd.DataFrame(
            {
                "user_id": ["user1", "user1", "user2", "user2"],
                "book_id": ["BOOK_001", "BOOK_002", "BOOK_001", "BOOK_003"],
                "rating": [5, 3, 4, 5],
                "interaction_count": [1, 1, 1, 1],
            }
        )

        # Mock book features
        features_df = pd.DataFrame(
            {
                "book_id": ["BOOK_001", "BOOK_002", "BOOK_003"],
                "genre": ["Fiction", "Science", "Fiction"],
                "price": [19.99, 24.99, 22.99],
                "publication_year": [2020, 2021, 2022],
            }
        )

        # Mock popular items
        popular_df = pd.DataFrame(
            {
                "book_id": ["BOOK_001", "BOOK_002", "BOOK_003"],
                "title": ["Book 1", "Book 2", "Book 3"],
                "total_sales": [100, 80, 60],
            }
        )

        mock_db.fetch_dataframe.side_effect = [interactions_df, features_df, popular_df]

        return mock_db

    @pytest.fixture
    def recommendation_engine(self, mock_db_manager):
        """Create recommendation engine."""
        engine = RecommendationEngine(mock_db_manager)
        return engine

    def test_load_data(self, recommendation_engine):
        """Test data loading."""
        df = recommendation_engine.load_data()

        assert isinstance(df, pd.DataFrame)
        assert "user_id" in df.columns
        assert "book_id" in df.columns
        assert "rating" in df.columns

    def test_train(self, recommendation_engine):
        """Test training the recommendation engine."""
        with patch.object(recommendation_engine, "save_model"):
            metrics = recommendation_engine.train()

        assert isinstance(metrics, dict)
        assert "num_users" in metrics
        assert "num_items" in metrics
        assert recommendation_engine.is_trained is True

    def test_get_similar_items(self, recommendation_engine):
        """Test getting similar items."""
        # Train the model first
        with patch.object(recommendation_engine, "save_model"):
            recommendation_engine.train()

        # Mock similarity matrix
        recommendation_engine.similarity_matrix = pd.DataFrame(
            {
                "BOOK_001": [1.0, 0.8, 0.6],
                "BOOK_002": [0.8, 1.0, 0.4],
                "BOOK_003": [0.6, 0.4, 1.0],
            },
            index=["BOOK_001", "BOOK_002", "BOOK_003"],
        )

        similar_items = recommendation_engine.get_similar_items("BOOK_001", 2)

        assert len(similar_items) == 2
        assert all(isinstance(item, tuple) for item in similar_items)
        assert all(len(item) == 2 for item in similar_items)

    def test_get_similar_books(self, recommendation_engine):
        """Test getting similar books with details."""
        # Setup trained model
        recommendation_engine.is_trained = True
        recommendation_engine.similarity_matrix = pd.DataFrame(
            {"BOOK_001": [1.0, 0.8], "BOOK_002": [0.8, 1.0]},
            index=["BOOK_001", "BOOK_002"],
        )

        # Reset the mock and set up specific return value
        recommendation_engine.db_manager.reset_mock()
        book_details_df = pd.DataFrame(
            {
                "book_id": ["BOOK_002"],
                "title": ["Similar Book"],
                "author": ["Author Name"],
                "genre": ["Fiction"],
                "price": [19.99],
            }
        )
        # Make sure book_id is string type to match comparison
        book_details_df["book_id"] = book_details_df["book_id"].astype(str)
        recommendation_engine.db_manager.fetch_dataframe.return_value = book_details_df

        result = recommendation_engine.get_similar_books("BOOK_001", 5)

        assert isinstance(result, dict)
        assert "recommendations" in result
        assert "type" in result
        assert result["type"] == "content_based"

    def test_get_popular_items(self, recommendation_engine):
        """Test getting popular items."""
        # Setup popular items
        recommendation_engine.popular_items = pd.DataFrame(
            {"book_id": ["BOOK_001", "BOOK_002"], "total_sales": [100, 80]}
        )

        popular_items = recommendation_engine.get_popular_items(2)

        assert len(popular_items) == 2
        assert popular_items[0][1] > popular_items[1][1]  # Sorted by popularity

    def test_get_popular_by_user_type(self, recommendation_engine):
        """Test getting popular books by user type."""
        # Reset the mock and set up specific return value
        recommendation_engine.db_manager.reset_mock()
        popular_df = pd.DataFrame(
            {
                "book_id": ["BOOK_001", "BOOK_002"],
                "title": ["Popular Book 1", "Popular Book 2"],
                "author": ["Author 1", "Author 2"],
                "genre": ["Fiction", "Science"],
                "price": [19.99, 24.99],
                "popularity_score": [100, 80],
            }
        )
        # Make sure book_id is string type to match comparison
        popular_df["book_id"] = popular_df["book_id"].astype(str)
        # First call will be for the main query, second might be for fallback
        recommendation_engine.db_manager.fetch_dataframe.return_value = popular_df

        result = recommendation_engine.get_popular_by_user_type("Individual", 5)

        assert isinstance(result, dict)
        assert "recommendations" in result
        assert "user_type" in result
        assert result["user_type"] == "Individual"
        assert len(result["recommendations"]) >= 2
