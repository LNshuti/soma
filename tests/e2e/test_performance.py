import pytest
import time
import concurrent.futures
import requests
from src.models.forecasting.demand_model import DemandForecastingModel
from src.models.recommendation.engine import RecommendationEngine


class TestPerformance:
    """Performance and load tests."""

    def test_model_training_performance(self):
        """Test model training performance."""
        start_time = time.time()
        
        # Train demand model
        demand_model = DemandForecastingModel()
        demand_metrics = demand_model.train()
        
        training_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust as needed)
        assert training_time < 300  # 5 minutes
        assert "mae" in demand_metrics

    def test_prediction_performance(self):
        """Test prediction performance."""
        demand_model = DemandForecastingModel()
        
        # Try to load existing model or train quickly
        if not demand_model.load_model():
            pytest.skip("No trained model available for performance test")
        
        start_time = time.time()
        
        # Make multiple predictions
        for i in range(100):
            result = demand_model.predict_for_book(f"BOOK_{i:03d}", 7)
            assert "predictions" in result or "error" in result
        
        total_time = time.time() - start_time
        avg_time = total_time / 100
        
        # Each prediction should be fast
        assert avg_time < 0.1  # 100ms per prediction

    def test_concurrent_api_requests(self, api_server=None):
        """Test concurrent API request handling."""
        if not api_server:
            pytest.skip("API server not available")
        
        base_url = "http://localhost:5001"
        
        def make_request(book_id):
            try:
                response = requests.post(
                    f"{base_url}/predict/demand",
                    json={"book_id": f"BOOK_{book_id:03d}", "days_ahead": 7},
                    timeout=10
                )
                return response.status_code
            except:
                return 500
        
        start_time = time.time()
        
        # Make concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, i) for i in range(1, 21)]
            results = [future.result() for future in futures]
        
        total_time = time.time() - start_time
        
        # Most requests should succeed
        success_rate = sum(1 for status in results if status in [200, 404]) / len(results)
        assert success_rate > 0.7  # 70% success rate
        
        # Should handle requests reasonably fast
        assert total_time < 30  # 30 seconds for 20 concurrent requests

    def test_memory_usage(self):
        """Test memory usage during operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform memory-intensive operations
        demand_model = DemandForecastingModel()
        rec_engine = RecommendationEngine()
        
        # Make multiple predictions
        for i in range(50):
            demand_model.predict_for_book(f"BOOK_{i:03d}", 7)
            if rec_engine.is_trained:
                rec_engine.get_similar_books(f"BOOK_{i:03d}", 5)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (adjust as needed)
        assert memory_increase < 500 * 1024 * 1024  # 500MB increase max
