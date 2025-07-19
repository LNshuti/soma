import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, mock_open
import json
from src.utils.metrics import ModelMetrics


class TestModelMetrics:
    """Test model metrics functionality."""

    @pytest.fixture
    def metrics_instance(self, tmp_path):
        """Create ModelMetrics instance with temporary directory."""
        return ModelMetrics("test_model", str(tmp_path))

    def test_calculate_regression_metrics(self, metrics_instance):
        """Test regression metrics calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.2])
        
        metrics = metrics_instance.calculate_regression_metrics(y_true, y_pred)
        
        assert "mae" in metrics
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "mape" in metrics
        
        assert metrics["mae"] > 0
        assert metrics["mse"] > 0
        assert metrics["rmse"] > 0
        assert -1 <= metrics["r2"] <= 1

    def test_calculate_classification_metrics(self, metrics_instance):
        """Test classification metrics calculation."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        metrics = metrics_instance.calculate_classification_metrics(y_true, y_pred)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1

    def test_log_metrics(self, metrics_instance):
        """Test logging metrics to file."""
        test_metrics = {"mae": 0.5, "r2": 0.8}
        metadata = {"model_type": "test"}
        
        metrics_instance.log_metrics(test_metrics, metadata)
        
        assert metrics_instance.metrics_file.exists()
        
        with open(metrics_instance.metrics_file, 'r') as f:
            saved_data = json.load(f)
        
        assert len(saved_data) == 1
        assert saved_data[0]["metrics"] == test_metrics
        assert saved_data[0]["metadata"] == metadata

    def test_get_latest_metrics(self, metrics_instance):
        """Test getting latest metrics."""
        # First log some metrics
        metrics_instance.log_metrics({"mae": 0.5}, {"version": 1})
        metrics_instance.log_metrics({"mae": 0.3}, {"version": 2})
        
        latest = metrics_instance.get_latest_metrics()
        
        assert latest is not None
        assert latest["metrics"]["mae"] == 0.3
        assert latest["metadata"]["version"] == 2
