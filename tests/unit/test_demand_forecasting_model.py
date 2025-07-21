from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.forecasting.demand_model import DemandForecastingModel


class TestDemandForecastingModel:
    """Test demand forecasting model."""

    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager."""
        mock_db = MagicMock()
        mock_db.fetch_dataframe.return_value = pd.DataFrame(
            {
                "book_id": ["BOOK_001", "BOOK_002"],
                "sale_date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
                "quantity": [10, 15],
                "price": [19.99, 24.99],
                "page_count": [200, 300],
            }
        )
        return mock_db

    @pytest.fixture
    def demand_model(self, mock_db_manager):
        """Create demand forecasting model."""
        with patch("src.models.forecasting.demand_model.DatabaseManager") as mock_dm:
            mock_dm.return_value = mock_db_manager
            model = DemandForecastingModel()
            model.db_manager = mock_db_manager
            return model

    def test_load_data(self, demand_model):
        """Test data loading."""
        df = demand_model.load_data()

        assert isinstance(df, pd.DataFrame)
        assert "book_id" in df.columns
        assert "quantity" in df.columns
        assert len(df) == 2

    def test_prepare_features(self, demand_model):
        """Test feature preparation."""
        df = demand_model.load_data()
        X, y = demand_model.prepare_features(df)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(demand_model.feature_columns) > 0
        assert "quantity" not in demand_model.feature_columns

    @patch("src.models.forecasting.demand_model.RandomForestRegressor")
    def test_train(self, mock_rf, demand_model):
        """Test model training."""
        mock_model = MagicMock()
        # Mock predict to return array matching test data size
        mock_model.predict.return_value = [10]  # Single prediction for test data
        mock_rf.return_value = mock_model

        # Mock the data loading and preparation
        df = pd.DataFrame(
            {
                "book_id": ["BOOK_001"] * 10,
                "sale_date": pd.date_range("2023-01-01", periods=10),
                "quantity": range(1, 11),
                "price": [19.99] * 10,
                "page_count": [200] * 10,
            }
        )
        demand_model.db_manager.fetch_dataframe.return_value = df

        with patch.object(demand_model, "save_model"):
            metrics = demand_model.train()

        assert isinstance(metrics, dict)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert mock_model.fit.called

    def test_predict_for_book(self, demand_model):
        """Test prediction for specific book."""
        # Mock the model
        demand_model.model = MagicMock()
        demand_model.model.predict.return_value = np.array([5.0])
        demand_model.feature_columns = ["price", "page_count"]

        # Mock database responses
        demand_model.db_manager.fetch_dataframe.side_effect = [
            pd.DataFrame(
                {
                    "book_id": ["BOOK_001"],
                    "sale_date": [pd.to_datetime("2023-01-01")],
                    "daily_quantity": [5],
                    "price": [19.99],
                    "page_count": [200],
                }
            )
        ]

        result = demand_model.predict_for_book("BOOK_001", 7)

        assert isinstance(result, dict)
        assert "book_id" in result
        assert "predictions" in result
        assert len(result["predictions"]) == 7
        assert all("day" in pred for pred in result["predictions"])
        assert all("predicted_demand" in pred for pred in result["predictions"])

    def test_predict_for_book_no_data(self, demand_model):
        """Test prediction when no data available."""
        demand_model.db_manager.fetch_dataframe.return_value = pd.DataFrame()

        result = demand_model.predict_for_book("BOOK_999", 5)

        assert result["book_id"] == "BOOK_999"
        assert len(result["predictions"]) == 5
        assert result["method"] == "error_fallback"
