# Testing Guide

This guide covers the testing strategy, frameworks, and best practices for the SOMA Content Analytics Platform.

## Testing Strategy

### Testing Pyramid
```
    /\
   /  \     E2E Tests (Few)
  /____\    Integration Tests (Some)
 /______\   Unit Tests (Many)
```

### Test Categories

1. **Unit Tests**: Individual components and functions
2. **Integration Tests**: Component interactions and API endpoints  
3. **End-to-End Tests**: Complete user workflows
4. **Performance Tests**: Load and stress testing

## Test Framework Setup

### Dependencies
```bash
# Core testing
pytest>=7.0
pytest-cov>=4.0
pytest-mock>=3.10
pytest-asyncio>=0.21

# Test utilities
factory-boy>=3.2
faker>=18.0
responses>=0.23

# Performance testing
locust>=2.0
```

### Configuration

#### pytest.ini
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow running tests
    database: Tests requiring database
```

#### conftest.py
```python
import pytest
import pandas as pd
from unittest.mock import Mock
from src.utils.database import DatabaseManager
from src.api.app import create_app

@pytest.fixture
def app():
    """Create test Flask application."""
    app = create_app(testing=True)
    return app

@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()

@pytest.fixture
def db_manager():
    """Create test database manager."""
    return DatabaseManager(db_path=":memory:")

@pytest.fixture
def sample_books_data():
    """Sample books data for testing."""
    return pd.DataFrame({
        'book_id': ['BOOK_000001', 'BOOK_000002'],
        'title': ['Test Book 1', 'Test Book 2'],
        'author': ['Author 1', 'Author 2'],
        'genre': ['Fiction', 'Science'],
        'price': [29.99, 39.99]
    })

@pytest.fixture
def mock_model():
    """Mock ML model for testing."""
    model = Mock()
    model.predict.return_value = [1.0, 2.0, 3.0]
    model.is_trained = True
    return model
```

## Unit Tests

### Testing Individual Functions

#### Data Processing Tests
```python
# tests/unit/test_data_generators.py
import pytest
import pandas as pd
from src.data.generators import ContentDataGenerator, DataGenerationConfig

class TestContentDataGenerator:
    """Test suite for ContentDataGenerator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = DataGenerationConfig(
            n_publishers=5,
            n_books=10,
            n_sales=20
        )
        self.generator = ContentDataGenerator(self.config)
    
    def test_config_initialization(self):
        """Test configuration initialization."""
        assert self.config.n_publishers == 5
        assert self.config.n_books == 10
        assert self.config.n_sales == 20
    
    @pytest.mark.unit
    def test_publisher_generation(self, db_manager):
        """Test publisher data generation."""
        # Setup
        self.generator.db_manager = db_manager
        
        # Execute
        count = self.generator._generate_publishers(db_manager)
        
        # Assert
        assert count == 5
        publishers = db_manager.get_table("publishers", "raw")
        assert len(publishers) == 5
        assert all(col in publishers.columns for col in [
            'publisher_id', 'publisher_name', 'publisher_type'
        ])
    
    @pytest.mark.unit
    def test_book_generation_requires_publishers(self, db_manager):
        """Test that book generation requires publishers."""
        # Setup
        self.generator.db_manager = db_manager
        
        # Execute & Assert
        with pytest.raises(ValueError, match="Publishers table not found"):
            self.generator._generate_books(db_manager)
```

#### Model Tests
```python
# tests/unit/test_demand_forecasting_model.py
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from src.models.forecasting.demand_model import DemandForecastingModel

class TestDemandForecastingModel:
    """Test suite for DemandForecastingModel."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = DemandForecastingModel()
    
    @pytest.mark.unit
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model.model_name == "demand_forecasting"
        assert not self.model.is_trained
        assert self.model.model is None
    
    @pytest.mark.unit
    def test_feature_engineering(self):
        """Test feature engineering functionality."""
        # Setup test data
        data = pd.DataFrame({
            'sale_date': pd.date_range('2024-01-01', periods=10),
            'quantity': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'unit_price': [29.99] * 10
        })
        
        # Execute
        features = self.model._engineer_features(data)
        
        # Assert
        expected_columns = [
            'year', 'month', 'quarter', 'day_of_week',
            'quantity_lag_1', 'quantity_ma_7'
        ]
        for col in expected_columns:
            assert col in features.columns
    
    @pytest.mark.unit
    @patch('src.models.forecasting.demand_model.RandomForestRegressor')
    def test_model_training(self, mock_rf):
        """Test model training process."""
        # Setup
        mock_rf_instance = Mock()
        mock_rf.return_value = mock_rf_instance
        
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [5, 6, 7, 8]
        })
        y = pd.Series([10, 20, 30, 40])
        
        # Execute
        metrics = self.model.train(X, y)
        
        # Assert
        mock_rf_instance.fit.assert_called_once()
        assert self.model.is_trained
        assert 'mse' in metrics
        assert 'r2_score' in metrics
```

### Testing Utilities

#### Database Tests
```python
# tests/unit/test_database_manager.py
import pytest
import pandas as pd
from src.utils.database import DatabaseManager

class TestDatabaseManager:
    """Test suite for DatabaseManager."""
    
    @pytest.mark.unit
    @pytest.mark.database
    def test_singleton_behavior(self):
        """Test that DatabaseManager is a singleton."""
        db1 = DatabaseManager(":memory:")
        db2 = DatabaseManager(":memory:")
        assert db1 is db2
    
    @pytest.mark.unit
    @pytest.mark.database
    def test_save_and_retrieve_dataframe(self, db_manager):
        """Test saving and retrieving DataFrames."""
        # Setup test data
        test_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'value': [10.5, 20.3, 30.1]
        })
        
        # Execute
        db_manager.save_dataframe(test_data, "test_table", "raw")
        retrieved_data = db_manager.get_table("test_table", "raw")
        
        # Assert
        pd.testing.assert_frame_equal(test_data, retrieved_data)
    
    @pytest.mark.unit
    @pytest.mark.database
    def test_fetch_dataframe_with_params(self, db_manager):
        """Test parameterized queries."""
        # Setup
        test_data = pd.DataFrame({
            'id': [1, 2, 3],
            'category': ['A', 'B', 'A']
        })
        db_manager.save_dataframe(test_data, "test_table", "raw")
        
        # Execute
        query = "SELECT * FROM raw.test_table WHERE category = ?"
        result = db_manager.fetch_dataframe(query, ['A'])
        
        # Assert
        assert len(result) == 2
        assert all(result['category'] == 'A')
```

## Integration Tests

### API Endpoint Tests

#### Health Endpoint Tests
```python
# tests/integration/test_api_endpoints.py
import pytest
import json
from src.api.app import create_app

class TestAPIEndpoints:
    """Integration tests for API endpoints."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test client."""
        self.app = create_app(testing=True)
        self.client = self.app.test_client()
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get('/health')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'version' in data
    
    def test_health_endpoint_with_database_failure(self, monkeypatch):
        """Test health endpoint when database fails."""
        # Mock database failure
        def mock_db_failure(*args, **kwargs):
            raise Exception("Database connection failed")
        
        monkeypatch.setattr(
            'src.utils.database.DatabaseManager.get_connection',
            mock_db_failure
        )
        
        response = self.client.get('/health')
        
        assert response.status_code == 503
        data = response.get_json()
        assert data['status'] == 'unhealthy'
```

#### Recommendation Endpoint Tests
```python
# tests/integration/test_recommendations.py
class TestRecommendationEndpoints:
    """Integration tests for recommendation endpoints."""
    
    def test_similar_books_endpoint(self, client, sample_books_data):
        """Test similar books recommendation endpoint."""
        # Setup test data in database
        # (This would require actual database setup)
        
        response = client.get('/api/recommendations/similar/BOOK_000001')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['type'] == 'content_based'
        assert data['source_book_id'] == 'BOOK_000001'
        assert 'recommendations' in data
        assert isinstance(data['recommendations'], list)
    
    def test_popular_books_endpoint(self, client):
        """Test popular books endpoint."""
        response = client.get('/api/recommendations/popular/Individual')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['type'] == 'popular_by_user_type'
        assert data['user_type'] == 'Individual'
        assert 'recommendations' in data
    
    def test_personalized_recommendations(self, client):
        """Test personalized recommendations endpoint."""
        request_data = {
            'user_id': 'test_user',
            'user_type': 'Individual',
            'previous_purchases': ['BOOK_000001'],
            'preferences': {
                'genres': ['Science'],
                'price_range': {'min': 10.0, 'max': 50.0}
            },
            'limit': 5
        }
        
        response = client.post(
            '/api/recommendations/personalized',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'recommendations' in data
        assert len(data['recommendations']) <= 5
```

### Database Integration Tests

#### dbt Model Tests
```python
# tests/integration/test_dbt_integration.py
import pytest
import subprocess
import pandas as pd
from src.utils.database import DatabaseManager

class TestDbtIntegration:
    """Integration tests for dbt models."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_dbt_run_success(self, db_manager):
        """Test that dbt models run successfully."""
        # Setup test data
        self._setup_raw_data(db_manager)
        
        # Run dbt
        result = subprocess.run(
            ['dbt', 'run', '--profiles-dir', 'dbt', '--target', 'test'],
            cwd='.',
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert 'Completed successfully' in result.stdout
    
    @pytest.mark.integration
    def test_fact_sales_model(self, db_manager):
        """Test fact_sales model output."""
        # Setup and run dbt
        self._setup_raw_data(db_manager)
        self._run_dbt_model('marts.fact_sales')
        
        # Verify output
        fact_sales = db_manager.get_table('fact_sales', 'main')
        
        assert len(fact_sales) > 0
        expected_columns = [
            'transaction_id', 'book_id', 'sale_date',
            'quantity', 'total_amount', 'channel'
        ]
        for col in expected_columns:
            assert col in fact_sales.columns
    
    def _setup_raw_data(self, db_manager):
        """Setup raw test data."""
        # Create minimal test data
        publishers = pd.DataFrame({
            'publisher_id': ['PUB_001'],
            'publisher_name': ['Test Publisher'],
            'publisher_type': ['Traditional']
        })
        
        books = pd.DataFrame({
            'book_id': ['BOOK_001'],
            'title': ['Test Book'],
            'author': ['Test Author'],
            'publisher_id': ['PUB_001'],
            'genre': ['Fiction'],
            'price': [29.99]
        })
        
        sales = pd.DataFrame({
            'transaction_id': ['TXN_001'],
            'book_id': ['BOOK_001'],
            'sale_date': ['2024-01-01'],
            'quantity': [1],
            'unit_price': [29.99],
            'total_amount': [29.99],
            'channel': ['Online'],
            'customer_type': ['Individual']
        })
        
        db_manager.save_dataframe(publishers, 'publishers', 'raw')
        db_manager.save_dataframe(books, 'books', 'raw')
        db_manager.save_dataframe(sales, 'sales', 'raw')
```

## End-to-End Tests

### Complete Workflow Tests

#### User Journey Tests
```python
# tests/e2e/test_complete_workflow.py
import pytest
import requests
import time

class TestCompleteWorkflow:
    """End-to-end tests for complete user workflows."""
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_recommendation_workflow(self, base_url):
        """Test complete recommendation workflow."""
        # Step 1: Check system health
        health_response = requests.get(f"{base_url}/health")
        assert health_response.status_code == 200
        
        # Step 2: Get popular books
        popular_response = requests.get(
            f"{base_url}/api/recommendations/popular/Individual"
        )
        assert popular_response.status_code == 200
        popular_books = popular_response.json()['recommendations']
        assert len(popular_books) > 0
        
        # Step 3: Get similar books for first popular book
        book_id = popular_books[0]['book_id']
        similar_response = requests.get(
            f"{base_url}/api/recommendations/similar/{book_id}"
        )
        assert similar_response.status_code == 200
        similar_books = similar_response.json()['recommendations']
        
        # Step 4: Verify similar books are different from source
        for book in similar_books:
            assert book['book_id'] != book_id
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_prediction_workflow(self, base_url):
        """Test demand forecasting workflow."""
        # Get available books
        health_response = requests.get(f"{base_url}/health")
        assert health_response.status_code == 200
        
        # Make prediction request
        prediction_data = {
            'book_ids': ['BOOK_000001'],
            'forecast_horizon': 7,
            'confidence_level': 0.95
        }
        
        prediction_response = requests.post(
            f"{base_url}/api/predictions/demand",
            json=prediction_data
        )
        
        assert prediction_response.status_code == 200
        predictions = prediction_response.json()['predictions']
        assert len(predictions) == 1
        
        prediction = predictions[0]
        assert prediction['book_id'] == 'BOOK_000001'
        assert 'forecasted_demand' in prediction
        assert 'confidence_interval' in prediction
```

### Performance Tests

#### Load Testing with Locust
```python
# tests/performance/locustfile.py
from locust import HttpUser, task, between

class SomaAPIUser(HttpUser):
    """Locust user for load testing SOMA API."""
    
    wait_time = between(1, 3)
    
    def on_start(self):
        """Setup user session."""
        # Check if system is healthy
        response = self.client.get("/health")
        if response.status_code != 200:
            raise Exception("System not healthy")
    
    @task(3)
    def get_popular_books(self):
        """Test popular books endpoint."""
        user_types = ['Individual', 'Business', 'Educational']
        user_type = self.environment.random.choice(user_types)
        
        with self.client.get(
            f"/api/recommendations/popular/{user_type}",
            name="/api/recommendations/popular/[user_type]"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                assert 'recommendations' in data
    
    @task(2)
    def get_similar_books(self):
        """Test similar books endpoint."""
        # Use a known book ID
        book_id = "BOOK_000001"
        
        with self.client.get(
            f"/api/recommendations/similar/{book_id}",
            name="/api/recommendations/similar/[book_id]"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                assert 'recommendations' in data
    
    @task(1)
    def make_prediction(self):
        """Test prediction endpoint."""
        prediction_data = {
            'book_ids': ['BOOK_000001'],
            'forecast_horizon': 7
        }
        
        with self.client.post(
            "/api/predictions/demand",
            json=prediction_data
        ) as response:
            if response.status_code == 200:
                data = response.json()
                assert 'predictions' in data
```

## Test Data Management

### Test Fixtures

#### Factory Pattern
```python
# tests/factories.py
import factory
import pandas as pd
from datetime import datetime, timedelta

class BookFactory(factory.Factory):
    """Factory for creating test book data."""
    
    class Meta:
        model = dict
    
    book_id = factory.Sequence(lambda n: f"BOOK_{n:06d}")
    title = factory.Faker('catch_phrase')
    author = factory.Faker('name')
    genre = factory.Faker('random_element', 
                         elements=['Fiction', 'Science', 'Biography'])
    price = factory.Faker('pydecimal', 
                         left_digits=2, right_digits=2, 
                         min_value=10, max_value=100)

class SalesFactory(factory.Factory):
    """Factory for creating test sales data."""
    
    class Meta:
        model = dict
    
    transaction_id = factory.Sequence(lambda n: f"TXN_{n:08d}")
    book_id = factory.LazyAttribute(lambda obj: BookFactory().book_id)
    sale_date = factory.Faker('date_between', 
                             start_date='-1y', end_date='today')
    quantity = factory.Faker('random_int', min=1, max=5)
    unit_price = factory.Faker('pydecimal', 
                              left_digits=2, right_digits=2,
                              min_value=10, max=50)

# Usage in tests
def test_with_factory_data():
    books = [BookFactory() for _ in range(10)]
    sales = [SalesFactory() for _ in range(20)]
    
    books_df = pd.DataFrame(books)
    sales_df = pd.DataFrame(sales)
```

### Test Database Management

#### Database Fixtures
```python
# tests/fixtures/database.py
import pytest
import tempfile
import os
from src.utils.database import DatabaseManager
from src.data.generators import ContentDataGenerator, DataGenerationConfig

@pytest.fixture(scope="session")
def test_database():
    """Create test database for session."""
    # Create temporary database file
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.duckdb')
    temp_db.close()
    
    try:
        # Generate test data
        db_manager = DatabaseManager(temp_db.name)
        config = DataGenerationConfig(
            n_publishers=5,
            n_books=20,
            n_sales=50,
            n_inventory=10,
            n_campaigns=10
        )
        generator = ContentDataGenerator(config)
        generator.generate_all()
        
        yield temp_db.name
        
    finally:
        # Cleanup
        os.unlink(temp_db.name)

@pytest.fixture
def clean_database(test_database):
    """Provide clean database for each test."""
    # Each test gets a copy of the session database
    # This could be optimized with transactions/rollbacks
    yield test_database
```

## Test Automation

### CI/CD Integration

#### GitHub Actions
```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ --cov=src --cov-report=xml
    
    - name: Run integration tests
      run: |
        pytest tests/integration/
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Test Reporting

#### Coverage Reports
```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# Generate XML for CI
pytest --cov=src --cov-report=xml

# Fail if coverage below threshold
pytest --cov=src --cov-fail-under=80
```

#### Test Results
```bash
# JUnit XML for CI integration
pytest --junitxml=test-results.xml

# Generate test report
pytest --html=test-report.html --self-contained-html
```

## Best Practices

### Test Organization
1. **Naming**: Use descriptive test names that explain the scenario
2. **Structure**: Arrange tests using Arrange-Act-Assert pattern
3. **Independence**: Each test should be independent and idempotent
4. **Speed**: Keep unit tests fast, mark slow tests appropriately

### Mocking Guidelines
```python
# Good: Mock external dependencies
@patch('src.models.forecasting.requests.get')
def test_external_api_call(mock_get):
    mock_get.return_value.json.return_value = {'data': 'test'}
    result = fetch_external_data()
    assert result == {'data': 'test'}

# Good: Mock at the boundary
@patch('src.utils.database.duckdb.connect')
def test_database_operation(mock_connect):
    mock_conn = Mock()
    mock_connect.return_value = mock_conn
    # Test database operation
```

### Assertion Best Practices
```python
# Good: Specific assertions
assert response.status_code == 200
assert len(recommendations) == 5
assert recommendation['book_id'].startswith('BOOK_')

# Good: Use pytest assertions for better error messages
assert 'error' not in response.json()
assert response.json()['status'] == 'success'

# Good: Use pandas testing for DataFrames
pd.testing.assert_frame_equal(expected_df, actual_df)
```

---

*For continuous integration setup, see the [Development Guide](development.md)*