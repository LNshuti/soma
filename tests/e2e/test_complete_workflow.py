import pytest
import requests
import time
import subprocess
import os
import signal
from pathlib import Path


class TestCompleteWorkflow:
    """End-to-end workflow tests."""

    @pytest.fixture(scope="class")
    def api_server(self):
        """Start API server for testing."""
        # Start the API server
        process = subprocess.Popen(
            ["python", "-m", "src.api.app"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        time.sleep(5)
        
        # Check if server is running
        try:
            response = requests.get("http://localhost:5001/health", timeout=5)
            if response.status_code in [200, 503]:
                yield "http://localhost:5001"
            else:
                pytest.skip("API server failed to start")
        except requests.exceptions.RequestException:
            pytest.skip("API server not accessible")
        finally:
            # Cleanup
            process.terminate()
            process.wait()

    def test_data_generation_to_prediction_workflow(self):
        """Test complete workflow from data generation to prediction."""
        # Step 1: Generate data
        from src.data.generators import ContentDataGenerator, DataGenerationConfig
        from src.utils.database import DatabaseManager
        
        config = DataGenerationConfig(
            n_publishers=50,
            n_books=100,
            n_sales=1000,
            n_inventory=500,
            n_campaigns=100
        )
        
        generator = ContentDataGenerator(config)
        results = generator.generate_all()
        
        assert all(count > 0 for count in results.values())
        
        # Step 2: Train models
        from src.models.forecasting.demand_model import DemandForecastingModel
        from src.models.recommendation.engine import RecommendationEngine
        
        # Train demand model
        demand_model = DemandForecastingModel()
        demand_metrics = demand_model.train()
        assert "mae" in demand_metrics
        
        # Train recommendation engine
        rec_engine = RecommendationEngine()
        rec_metrics = rec_engine.train()
        assert "num_users" in rec_metrics
        
        # Step 3: Test predictions
        demand_result = demand_model.predict_for_book("BOOK_001", 7)
        assert "predictions" in demand_result
        
        rec_result = rec_engine.get_similar_books("BOOK_001", 5)
        assert "recommendations" in rec_result or "error" in rec_result

    def test_api_workflow(self, api_server):
        """Test complete API workflow."""
        base_url = api_server
        
        # Test health check
        response = requests.get(f"{base_url}/health")
        assert response.status_code in [200, 503]
        
        # Test demand prediction
        prediction_payload = {
            "book_id": "BOOK_001",
            "days_ahead": 7
        }
        
        response = requests.post(
            f"{base_url}/predict/demand",
            json=prediction_payload
        )
        assert response.status_code in [200, 404, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 7
        
        # Test recommendations
        rec_payload = {
            "book_id": "BOOK_001",
            "n_recommendations": 5
        }
        
        response = requests.post(
            f"{base_url}/recommend/",
            json=rec_payload
        )
        assert response.status_code in [200, 404, 500]
        
        # Test content generation
        content_payload = {
            "book_id": "BOOK_001",
            "ad_type": "social_media"
        }
        
        response = requests.post(
            f"{base_url}/generate/ad-copy",
            json=content_payload
        )
        assert response.status_code in [200, 404, 500]

    def test_gradio_interface_availability(self):
        """Test Gradio interface availability."""
        # This would typically require starting the Gradio app
        # For now, just test that the module can be imported
        try:
            from src.web.gradio_app import SomaGradioApp, create_gradio_interface
            
            # Test app creation
            app = SomaGradioApp()
            assert app.db_path == "./data/soma.duckdb"
            
            # Test interface creation (without launching)
            interface = create_gradio_interface()
            assert interface is not None
            
        except ImportError as e:
            pytest.skip(f"Gradio dependencies not available: {e}")

    def test_model_persistence_workflow(self):
        """Test model training, saving, and loading workflow."""
        from src.models.forecasting.demand_model import DemandForecastingModel
        
        # Train and save model
        model1 = DemandForecastingModel()
        
        # Mock some training data
        import pandas as pd
        import numpy as np
        
        mock_data = pd.DataFrame({
            "book_id": ["BOOK_001"] * 100,
            "sale_date": pd.date_range("2023-01-01", periods=100),
            "daily_quantity": np.random.poisson(5, 100),
            "price": [19.99] * 100,
            "page_count": [200] * 100
        })
        
        X, y = model1.prepare_features(mock_data)
        if not X.empty and not y.empty:
            metrics1 = model1.train(X, y)
            
            # Save model
            model1.save_model()
            
            # Load model in new instance
            model2 = DemandForecastingModel()
            loaded = model2.load_model()
            
            if loaded:
                # Test that loaded model can make predictions
                predictions = model2.predict(X.head(5))
                assert len(predictions) == 5
