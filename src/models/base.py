# src/models/base.py
"""Base model class and interfaces."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import pandas as pd

from src.utils.database import DatabaseManager
from src.utils.metrics import ModelMetrics

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types."""

    FORECASTING = "forecasting"
    RECOMMENDATION = "recommendation"
    RAG = "rag"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class BaseModel(ABC):
    """Abstract base class for all ML models."""

    def __init__(
        self,
        model_name: str = "base_model",
        model_type: ModelType = ModelType.REGRESSION,
        db_manager: Optional[DatabaseManager] = None,
        model_dir: str = "./artifacts/models",
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.db_manager = db_manager or DatabaseManager()
        self.metrics = ModelMetrics(model_name)
        self.last_trained = None

        # Model artifacts
        self.model = None
        self.preprocessors = {}
        self.metadata = {}

        # Paths
        self.model_dir = Path(model_dir) / model_name
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized {model_type.value} model: {model_name}")

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load training data."""
        pass

    @abstractmethod
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features for training."""
        pass

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Any:
        """Make predictions."""
        pass

    def save_model(self) -> None:
        """Save model and artifacts."""
        try:
            # Save main model
            if self.model is not None:
                model_path = self.model_dir / "model.pkl"
                joblib.dump(self.model, model_path)
                logger.info(f"Model saved to {model_path}")

            # Save preprocessors
            if self.preprocessors:
                preprocessors_path = self.model_dir / "preprocessors.pkl"
                joblib.dump(self.preprocessors, preprocessors_path)
                logger.info(f"Preprocessors saved to {preprocessors_path}")

            # Save metadata
            if self.metadata:
                metadata_path = self.model_dir / "metadata.pkl"
                joblib.dump(self.metadata, metadata_path)
                logger.info(f"Metadata saved to {metadata_path}")

        except Exception as e:
            logger.error(f"Error saving model {self.model_name}: {e}")
            raise

    def load_model(self) -> bool:
        """Load saved model and artifacts."""
        try:
            model_path = self.model_dir / "model.pkl"
            if model_path.exists():
                self.model = joblib.load(model_path)
                logger.info(f"Model loaded from {model_path}")

            preprocessors_path = self.model_dir / "preprocessors.pkl"
            if preprocessors_path.exists():
                self.preprocessors = joblib.load(preprocessors_path)
                logger.info(f"Preprocessors loaded from {preprocessors_path}")

            metadata_path = self.model_dir / "metadata.pkl"
            if metadata_path.exists():
                self.metadata = joblib.load(metadata_path)
                logger.info(f"Metadata loaded from {metadata_path}")

            return self.model is not None

        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            return False

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        predictions = self.predict(X_test)

        # Calculate metrics based on model type
        if self.model_type in [ModelType.FORECASTING, ModelType.REGRESSION]:
            metrics = self.metrics.calculate_regression_metrics(y_test, predictions)
        elif self.model_type == ModelType.CLASSIFICATION:
            metrics = self.metrics.calculate_classification_metrics(y_test, predictions)
        else:
            metrics = {}

        # Log metrics
        self.metrics.log_metrics(metrics, {"model_type": self.model_type.value})

        return metrics

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance if available."""
        if hasattr(self.model, "feature_importances_"):
            importance_df = pd.DataFrame(
                {
                    "feature": self.metadata.get("feature_names", []),
                    "importance": self.model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)
            return importance_df
        return None

    def __str__(self) -> str:
        return f"{self.model_type.value.title()}Model({self.model_name})"

    def __repr__(self) -> str:
        return self.__str__()
