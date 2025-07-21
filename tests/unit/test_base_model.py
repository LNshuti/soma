from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.base import BaseModel, ModelType
from src.utils.database import DatabaseManager


class ConcreteModel(BaseModel):
    """Concrete implementation for testing."""

    def load_data(self) -> pd.DataFrame:
        return pd.DataFrame({"feature1": [1, 2, 3], "target": [10, 20, 30]})

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        X = df[["feature1"]]
        y = df["target"]
        return X, y

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model = MagicMock()
        self.model.feature_importances_ = np.array([0.8])
        self.metadata["feature_names"] = ["feature1"]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([15, 25, 35])


class TestBaseModel:
    """Test base model functionality."""

    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager."""
        with patch("src.utils.database.DatabaseManager") as mock:
            yield mock.return_value

    @pytest.fixture
    def concrete_model(self, mock_db_manager):
        """Create concrete model instance."""
        return ConcreteModel(
            model_name="test_model",
            model_type=ModelType.REGRESSION,
            db_manager=mock_db_manager,
        )

    def test_model_initialization(self, concrete_model):
        """Test model initialization."""
        assert concrete_model.model_name == "test_model"
        assert concrete_model.model_type == ModelType.REGRESSION
        assert concrete_model.model is None
        assert concrete_model.last_trained is None

    def test_load_data(self, concrete_model):
        """Test data loading."""
        df = concrete_model.load_data()

        assert isinstance(df, pd.DataFrame)
        assert "feature1" in df.columns
        assert "target" in df.columns
        assert len(df) == 3

    def test_prepare_features(self, concrete_model):
        """Test feature preparation."""
        df = concrete_model.load_data()
        X, y = concrete_model.prepare_features(df)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert X.shape == (3, 1)
        assert len(y) == 3

    def test_train(self, concrete_model):
        """Test model training."""
        df = concrete_model.load_data()
        X, y = concrete_model.prepare_features(df)

        concrete_model.train(X, y)

        assert concrete_model.model is not None
        assert hasattr(concrete_model.model, "feature_importances_")

    def test_predict(self, concrete_model):
        """Test model prediction."""
        df = concrete_model.load_data()
        X, y = concrete_model.prepare_features(df)
        concrete_model.train(X, y)

        predictions = concrete_model.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 3

    @patch("joblib.dump")
    def test_save_model(self, mock_dump, concrete_model):
        """Test model saving."""
        concrete_model.model = MagicMock()
        concrete_model.preprocessors = {"scaler": MagicMock()}
        concrete_model.metadata = {"version": "1.0"}

        concrete_model.save_model()

        # Should call joblib.dump for model, preprocessors, and metadata
        assert mock_dump.call_count == 3

    @patch("joblib.load")
    @patch("pathlib.Path.exists")
    def test_load_model(self, mock_exists, mock_load, concrete_model):
        """Test model loading."""
        mock_exists.return_value = True
        mock_load.side_effect = [
            MagicMock(),
            {"scaler": MagicMock()},
            {"version": "1.0"},
        ]

        result = concrete_model.load_model()

        assert result is True
        assert concrete_model.model is not None
        assert len(concrete_model.preprocessors) == 1
        assert concrete_model.metadata["version"] == "1.0"

    def test_get_feature_importance(self, concrete_model):
        """Test feature importance extraction."""
        df = concrete_model.load_data()
        X, y = concrete_model.prepare_features(df)
        concrete_model.train(X, y)

        importance_df = concrete_model.get_feature_importance()

        assert isinstance(importance_df, pd.DataFrame)
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns
        assert len(importance_df) == 1

    def test_string_representation(self, concrete_model):
        """Test string representation."""
        assert str(concrete_model) == "RegressionModel(test_model)"
        assert repr(concrete_model) == "RegressionModel(test_model)"
