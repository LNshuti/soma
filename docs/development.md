# Development Guide

This guide covers the development workflow, code standards, and best practices for contributing to the SOMA Content Analytics Platform.

## Development Environment Setup

### Prerequisites
- Python 3.11+
- Docker and Docker Compose
- Git
- Node.js 16+ (for documentation)

### Local Setup
```bash
# Clone repository
git clone https://github.com/LNshuti/soma.git

cd soma

# Create virtual environment
conda env create --file=environment.yaml

conda activate soma

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Setup database
python -m src.data.generators
cd dbt && dbt run --profiles-dir . --target dev && cd ..
```

### Development Dependencies
```bash
# Testing
pytest>=7.0
pytest-cov>=4.0
pytest-mock>=3.10

# Code quality
black>=23.0
isort>=5.12
flake8>=6.0
mypy>=1.0

# Pre-commit hooks
pre-commit>=3.0

# Development tools
jupyterlab>=4.0
ipython>=8.0
```

## Project Structure

### Directory Layout
```
soma/
├── src/                    # Source code
│   ├── api/               # REST API
│   ├── web/               # Web interface
│   ├── data/              # Data pipeline
│   ├── models/            # ML models
│   └── utils/             # Utilities
├── dbt/                   # dbt models
├── tests/                 # Test suite
├── docs/                  # Documentation
├── deployment/            # Deployment configs
├── artifacts/             # Generated files
├── data/                  # Database files
└── notebooks/             # Jupyter notebooks
```

### Code Organization

#### API Layer (`src/api/`)
```
src/api/
├── __init__.py
├── app.py                 # Flask application factory
└── routes/
    ├── __init__.py
    ├── health.py          # Health check endpoints
    ├── predictions.py     # ML prediction endpoints
    └── recommendations.py # Recommendation endpoints
```

#### Models Layer (`src/models/`)
```
src/models/
├── __init__.py
├── base.py               # Base model interface
├── forecasting/
│   └── demand_model.py   # Demand forecasting
├── recommendation/
│   └── engine.py         # Recommendation engine
└── rag/
    └── generators.py     # RAG system
```

#### Utilities (`src/utils/`)
```
src/utils/
├── __init__.py
├── database.py           # Database management
├── helpers.py            # Common utilities
└── validators.py         # Data validation
```

## Code Standards

### Python Style Guide

#### Formatting
```python
# Use Black for code formatting
black src/ tests/

# Import organization with isort
isort src/ tests/

# Line length: 88 characters (Black default)
# Use double quotes for strings
# Use type hints for all functions
```

#### Type Hints
```python
from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np

def process_data(
    data: pd.DataFrame,
    columns: List[str],
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Process DataFrame with specified columns."""
    pass

class DataProcessor:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform input data."""
        pass
```

#### Documentation
```python
def calculate_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    metrics: List[str]
) -> Dict[str, float]:
    """Calculate evaluation metrics for predictions.
    
    Args:
        y_true: True values
        y_pred: Predicted values  
        metrics: List of metric names to calculate
        
    Returns:
        Dictionary mapping metric names to values
        
    Raises:
        ValueError: If metrics list is empty
        
    Example:
        >>> y_true = np.array([1, 2, 3])
        >>> y_pred = np.array([1.1, 2.1, 2.9])
        >>> calculate_metrics(y_true, y_pred, ['mse', 'mae'])
        {'mse': 0.013, 'mae': 0.1}
    """
    pass
```

#### Error Handling
```python
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def safe_division(a: float, b: float) -> Optional[float]:
    """Safely divide two numbers."""
    try:
        if b == 0:
            logger.warning("Division by zero attempted")
            return None
        return a / b
    except TypeError as e:
        logger.error(f"Type error in division: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in division: {e}")
        return None
```

### Database Patterns

#### Connection Management
```python
from contextlib import contextmanager
from src.utils.database import DatabaseManager

@contextmanager
def get_db_connection():
    """Get database connection with automatic cleanup."""
    db = DatabaseManager()
    conn = db.get_connection()
    try:
        yield conn
    finally:
        conn.close()

# Usage
with get_db_connection() as conn:
    result = conn.execute("SELECT * FROM books").fetchall()
```

#### Query Patterns
```python
def get_books_by_genre(genre: str) -> pd.DataFrame:
    """Get books filtered by genre."""
    query = """
    SELECT book_id, title, author, price
    FROM dim_books
    WHERE genre = ?
    ORDER BY title
    """
    
    db = DatabaseManager()
    return db.fetch_dataframe(query, params=[genre])
```

### Testing Patterns

#### Unit Tests
```python
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from src.models.base import BaseModel

class TestBaseModel:
    """Test suite for BaseModel class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = BaseModel("test_model", "classification")
    
    def test_initialization(self):
        """Test model initialization."""
        assert self.model.model_name == "test_model"
        assert self.model.model_type.value == "classification"
        assert not self.model.is_trained
    
    @patch('src.models.base.DatabaseManager')
    def test_load_data(self, mock_db):
        """Test data loading functionality."""
        # Setup mock
        mock_db.return_value.fetch_dataframe.return_value = pd.DataFrame({
            'feature1': [1, 2, 3],
            'target': [0, 1, 0]
        })
        
        # Test
        data = self.model.load_data()
        
        # Assert
        assert len(data) == 3
        assert 'feature1' in data.columns
```

#### Integration Tests
```python
import pytest
from src.api.app import create_app

@pytest.fixture
def client():
    """Create test client."""
    app = create_app(testing=True)
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get('/health')
    assert response.status_code == 200
    
    data = response.get_json()
    assert data['status'] == 'healthy'
    assert 'timestamp' in data

def test_recommendations_endpoint(client):
    """Test recommendations endpoint."""
    response = client.get('/api/recommendations/popular/Individual')
    assert response.status_code == 200
    
    data = response.get_json()
    assert 'recommendations' in data
    assert data['user_type'] == 'Individual'
```

## Development Workflow

### Git Workflow

#### Branch Strategy
```bash
# Main branches
main          # Production-ready code
develop       # Integration branch

# Feature branches
feature/add-new-model
feature/improve-api
bugfix/fix-database-connection
hotfix/critical-security-patch
```

#### Commit Messages
```bash
# Format: type(scope): description

feat(api): add new recommendation endpoint
fix(db): resolve connection pool exhaustion
docs(readme): update installation instructions
test(models): add unit tests for demand forecasting
refactor(utils): simplify database connection logic
```

#### Pull Request Process
1. Create feature branch from `develop`
2. Implement changes with tests
3. Run full test suite locally
4. Create pull request to `develop`
5. Code review and approval
6. Merge to `develop`
7. Deploy to staging for testing
8. Merge to `main` for production

### Development Commands

#### Code Quality
```bash
# Format code
make format
# or
black src/ tests/
isort src/ tests/

# Run linting
make lint
# or
flake8 src/ tests/
mypy src/

# Type checking
mypy src/ --strict
```

#### Testing
```bash
# Run all tests
make test
# or
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit/           # Unit tests only
pytest tests/integration/    # Integration tests only
pytest tests/e2e/           # End-to-end tests only

# Run tests in parallel
pytest -n auto
```

#### Database Operations
```bash
# Generate new data
python -m src.data.generators

# Run dbt models
cd dbt
dbt run --profiles-dir . --target dev
dbt test --profiles-dir . --target dev

# Reset database
rm -f data/soma.duckdb
python -m src.data.generators
cd dbt && dbt run --profiles-dir . --target dev
```

#### Model Training
```bash
# Train all models
python -m src.models.train_all

# Train specific model
python -m src.models.forecasting.demand_model
python -m src.models.recommendation.engine

# Model evaluation
python -m src.models.evaluate_all
```

### Development Tools

#### Makefile
```makefile
.PHONY: install format lint test clean

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

format:
	black src/ tests/
	isort src/ tests/

lint:
	flake8 src/ tests/
	mypy src/

test:
	pytest --cov=src --cov-report=html

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/

docker-build:
	docker compose build

docker-test:
	docker compose run --rm api pytest

setup-db:
	python -m src.data.generators
	cd dbt && dbt run --profiles-dir . --target dev

reset-db:
	rm -f data/soma.duckdb
	make setup-db
```

#### Pre-commit Configuration (`.pre-commit-config.yaml`)
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]
```

#### IDE Configuration

##### VS Code (`settings.json`)
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

### Debugging

#### Local Debugging
```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Remote debugging with debugpy
import debugpy
debugpy.listen(('0.0.0.0', 5678))
debugpy.wait_for_client()
```

#### Docker Debugging
```bash
# Run container in debug mode
docker compose -f docker-compose.yml -f docker-compose.debug.yml up

# Attach debugger (VS Code)
# Use "Python: Remote Attach" configuration
```

#### Database Debugging
```bash
# Connect to DuckDB CLI
duckdb data/soma.duckdb

# Query debugging
.explain SELECT * FROM fact_sales WHERE sale_date > '2024-01-01';
.timer on
```

## Performance Guidelines

### Code Optimization

#### Database Queries
```python
# Good: Use efficient queries
def get_sales_summary(start_date: str, end_date: str) -> pd.DataFrame:
    query = """
    SELECT 
        book_id,
        SUM(quantity) as total_quantity,
        SUM(total_amount) as total_revenue
    FROM fact_sales
    WHERE sale_date BETWEEN ? AND ?
    GROUP BY book_id
    """
    return db.fetch_dataframe(query, [start_date, end_date])

# Bad: Load all data then filter
def get_sales_summary_bad(start_date: str, end_date: str) -> pd.DataFrame:
    all_sales = db.fetch_dataframe("SELECT * FROM fact_sales")
    filtered = all_sales[
        (all_sales['sale_date'] >= start_date) & 
        (all_sales['sale_date'] <= end_date)
    ]
    return filtered.groupby('book_id').agg({
        'quantity': 'sum',
        'total_amount': 'sum'
    })
```

#### Memory Management
```python
# Use generators for large datasets
def process_large_dataset():
    for chunk in pd.read_csv('large_file.csv', chunksize=1000):
        yield process_chunk(chunk)

# Clear variables when done
def memory_efficient_processing():
    large_data = load_large_dataset()
    result = process_data(large_data)
    del large_data  # Free memory
    return result
```

### Monitoring

#### Performance Metrics
```python
import time
import functools
import logging

def timing_decorator(func):
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger = logging.getLogger(func.__module__)
        logger.info(
            f"{func.__name__} executed in {end_time - start_time:.2f} seconds"
        )
        return result
    return wrapper

@timing_decorator
def expensive_operation():
    """Some expensive operation."""
    pass
```

## Documentation Standards

### Code Documentation
- Use docstrings for all classes and functions
- Include type hints for all parameters
- Provide usage examples for complex functions
- Document error conditions and exceptions

### API Documentation
- Document all endpoints with examples
- Include request/response schemas
- Provide curl examples
- Document error codes and messages

### Architecture Documentation
- Maintain up-to-date architecture diagrams
- Document design decisions and trade-offs
- Include deployment and scaling considerations
- Document security and compliance requirements

## Security Guidelines

### Code Security
```python
# Input validation
from pydantic import BaseModel, validator

class BookRequest(BaseModel):
    book_id: str
    genre: str
    price: float
    
    @validator('book_id')
    def validate_book_id(cls, v):
        if not v.startswith('BOOK_'):
            raise ValueError('Invalid book ID format')
        return v
    
    @validator('price')
    def validate_price(cls, v):
        if v <= 0:
            raise ValueError('Price must be positive')
        return v
```

### Secrets Management
```python
import os
from typing import Optional

def get_secret(key: str) -> Optional[str]:
    """Get secret from environment variables."""
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Required secret {key} not found")
    return value

# Usage
API_KEY = get_secret('OPENAI_API_KEY')
```

## Contributing Guidelines

### Code Review Checklist
- [ ] Code follows style guide and passes linting
- [ ] All tests pass and coverage is maintained
- [ ] Documentation is updated for new features
- [ ] Security considerations are addressed
- [ ] Performance impact is considered
- [ ] Error handling is comprehensive
- [ ] Logging is appropriate and informative

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests pass locally
```

---

*For deployment and production considerations, see the [Deployment Guide](deployment/README.md)*