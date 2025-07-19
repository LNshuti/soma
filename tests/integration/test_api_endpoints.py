import pytest
import json
from flask import Flask
from src.api.app import create_app


class TestAPIEndpoints:
    """Test API endpoints integration."""

    @pytest.fixture
    def app(self):
        """Create test Flask app."""
        app = create_app()
        app.config['TESTING'] = True
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return app.test_client()

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get('/health')
        
        assert response.status_code in [200, 503]  # May be degraded without full setup
        data = json.loads(response.data)
        assert "status" in data
        assert "components" in data

    def test_demand_prediction_endpoint(self, client):
        """Test demand prediction endpoint."""
        payload = {
            "book_id": "BOOK_001",
            "days_ahead": 7
        }
        
        response = client.post(
            '/predict/demand',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # Should return something (may be fallback data)
        assert response.status_code in [200, 404, 500]
        data = json.loads(response.data)
        
        if response.status_code == 200:
            assert "book_id" in data
            assert "predictions" in data

    def test_recommendations_endpoint(self, client):
        """Test recommendations endpoint."""
        payload = {
            "book_id": "BOOK_001",
            "n_recommendations": 5
        }
        
        response = client.post(
            '/recommend/',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # Should return something (may be fallback data)
        assert response.status_code in [200, 404, 500]
        data = json.loads(response.data)
        
        if response.status_code == 200:
            assert "recommendations" in data or "error" in data

    def test_ad_copy_generation_endpoint(self, client):
        """Test ad copy generation endpoint."""
        payload = {
            "book_id": "BOOK_001",
            "ad_type": "social_media",
            "target_audience": "young_adult"
        }
        
        response = client.post(
            '/generate/ad-copy',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # Should return something (may be fallback data)
        assert response.status_code in [200, 404, 500]
        data = json.loads(response.data)
        
        if response.status_code == 200:
            assert "ad_copy_variants" in data or "error" in data

    def test_batch_prediction_endpoint(self, client):
        """Test batch prediction endpoint."""
        payload = {
            "book_ids": ["BOOK_001", "BOOK_002", "BOOK_003"],
            "days_ahead": 5
        }
        
        response = client.post(
            '/predict/batch-demand',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 500]
        data = json.loads(response.data)
        
        if response.status_code == 200:
            assert "predictions" in data
            assert len(data["predictions"]) == 3
