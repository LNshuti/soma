# src/utils/metrics.py
"""Model metrics and evaluation utilities."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)

logger = logging.getLogger(__name__)


class ModelMetrics:
    """Model performance metrics tracking."""

    def __init__(self, model_name: str, metrics_dir: str = "./artifacts/metrics"):
        self.model_name = model_name
        self.metrics_dir = Path(metrics_dir)
        self.metrics_file = self.metrics_dir / f"{model_name}_metrics.json"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def calculate_regression_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression metrics."""
        metrics = {
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2": r2_score(y_true, y_pred),
            "mape": np.mean(
                np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))
            )
            * 100,
        }
        return metrics

    def calculate_classification_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate classification metrics."""
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted"
        )

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
        return metrics

    def log_metrics(
        self, metrics: Dict[str, float], metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log metrics to file."""
        metric_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "metrics": metrics,
            "metadata": metadata or {},
        }

        # Load existing metrics or create new list
        if self.metrics_file.exists():
            with open(self.metrics_file, "r") as f:
                all_metrics = json.load(f)
        else:
            all_metrics = []

        all_metrics.append(metric_entry)

        # Save updated metrics
        with open(self.metrics_file, "w") as f:
            json.dump(all_metrics, f, indent=2)

        logger.info(f"Metrics logged for {self.model_name}: {metrics}")

    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Get the latest metrics for this model."""
        if not self.metrics_file.exists():
            return None

        with open(self.metrics_file, "r") as f:
            all_metrics = json.load(f)

        return all_metrics[-1] if all_metrics else None

    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get all metrics history for this model."""
        if not self.metrics_file.exists():
            return []

        with open(self.metrics_file, "r") as f:
            return json.load(f)

    def compare_models(self, metric_name: str = "rmse") -> pd.DataFrame:
        """Compare different models based on a specific metric."""
        comparison_data = []

        for metrics_file in self.metrics_dir.glob("*_metrics.json"):
            model_name = metrics_file.stem.replace("_metrics", "")

            with open(metrics_file, "r") as f:
                metrics_history = json.load(f)

            if metrics_history:
                latest_metrics = metrics_history[-1]
                if metric_name in latest_metrics["metrics"]:
                    comparison_data.append(
                        {
                            "model": model_name,
                            "metric_value": latest_metrics["metrics"][metric_name],
                            "timestamp": latest_metrics["timestamp"],
                        }
                    )

        if comparison_data:
            df = pd.DataFrame(comparison_data)
            return df.sort_values("metric_value")
        else:
            return pd.DataFrame()
