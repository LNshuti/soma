# Configuration Guide

This guide covers all configuration options for the SOMA Content Analytics Platform.

## Configuration Overview

SOMA uses a hierarchical configuration system:
1. **Default Values**: Built-in defaults in `config/settings.py`
2. **Environment Variables**: Override defaults via environment
3. **Configuration Files**: Optional `.env` files
4. **Runtime Parameters**: Command-line arguments and API parameters

## Configuration File (`config/settings.py`)

### Settings Class
```python
class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
        case_sensitive=False
    )
```

### Configuration Categories

#### Database Settings
```python
# Database configuration
DB_PATH: str = "./data/soma.duckdb"
```

**Environment Override**:
```bash
DB_PATH="/custom/path/soma.duckdb"
```

#### dbt Settings
```python
# dbt configuration
DBT_PROFILES_DIR: str = "./dbt"
DBT_PROJECT_DIR: str = "./dbt"
```

**Environment Override**:
```bash
DBT_PROFILES_DIR="/app/dbt/profiles"
DBT_PROJECT_DIR="/app/dbt"
```

#### API Settings
```python
# API configuration
API_DEBUG: bool = True
API_HOST: str = "0.0.0.0"
API_PORT: int = 5001
```

**Environment Override**:
```bash
API_DEBUG=false
API_HOST=127.0.0.1
API_PORT=8080
```

#### Web Interface Settings
```python
# Web interface configuration
WEB_HOST: str = "0.0.0.0"
WEB_PORT: int = 7860
WEB_SHARE: bool = False
```

**Environment Override**:
```bash
WEB_HOST=localhost
WEB_PORT=8860
WEB_SHARE=true
```

#### Directory Settings
```python
# Directory configuration
MODEL_DIR: str = "./artifacts/models"
LOG_DIR: str = "./artifacts/logs"
METRICS_DIR: str = "./artifacts/metrics"
```

**Environment Override**:
```bash
MODEL_DIR="/app/models"
LOG_DIR="/app/logs"
METRICS_DIR="/app/metrics"
```

#### External API Keys
```python
# External service configuration
OPENAI_API_KEY: str = "your_openai_key_here"
```

**Environment Override**:
```bash
OPENAI_API_KEY="sk-your-actual-key-here"
```

#### Logging Settings
```python
# Logging configuration
LOG_FORMAT: str = "json"
LOG_LEVEL: str = "INFO"
```

**Environment Override**:
```bash
LOG_FORMAT=text
LOG_LEVEL=DEBUG
```

#### Environment Settings
```python
# Environment configuration
ENVIRONMENT: str = "development"
DEBUG: bool = True
```

**Environment Override**:
```bash
ENVIRONMENT=production
DEBUG=false
```

#### Celery Settings
```python
# Celery configuration (for async tasks)
CELERY_BROKER_URL: str = "redis://localhost:6379/0"
CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
```

## Environment Files

### Local Development (`.env`)
```bash
# Local development configuration
ENVIRONMENT=development
DEBUG=true

# Database
DB_PATH=./data/soma.duckdb

# API
API_DEBUG=true
API_HOST=0.0.0.0
API_PORT=5001

# Web Interface
WEB_HOST=0.0.0.0
WEB_PORT=7860
WEB_SHARE=false

# Logging
LOG_LEVEL=DEBUG
LOG_FORMAT=text

# External APIs
OPENAI_API_KEY=your_key_here
```

### Production (`.env.production`)
```bash
# Production configuration
ENVIRONMENT=production
DEBUG=false

# Database
DB_PATH=/app/data/soma.duckdb

# API
API_DEBUG=false
API_HOST=0.0.0.0
API_PORT=5001

# Web Interface
WEB_HOST=0.0.0.0
WEB_PORT=7860
WEB_SHARE=false

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# External APIs
OPENAI_API_KEY=${OPENAI_API_KEY}

# Security
SESSION_SECRET_KEY=${SESSION_SECRET_KEY}
```

### Docker Environment (`.env.docker`)
```bash
# Docker-specific configuration
ENVIRONMENT=docker
DEBUG=false

# Database (mounted volume)
DB_PATH=/app/data/soma.duckdb

# API
API_HOST=0.0.0.0
API_PORT=5001

# Web Interface
WEB_HOST=0.0.0.0
WEB_PORT=7860

# Paths (container paths)
MODEL_DIR=/app/artifacts/models
LOG_DIR=/app/artifacts/logs
DBT_PROFILES_DIR=/app/dbt
```

## Kubernetes Configuration

### ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: soma-config
  namespace: soma-local
data:
  # Environment
  ENVIRONMENT: "production"
  DEBUG: "false"
  
  # Database
  DB_PATH: "/app/data/soma.duckdb"
  
  # API Settings
  API_HOST: "0.0.0.0"
  API_PORT: "5001"
  API_DEBUG: "false"
  
  # Web Interface
  WEB_HOST: "0.0.0.0"
  WEB_PORT: "7860"
  WEB_SHARE: "false"
  
  # Paths
  MODEL_DIR: "/app/artifacts/models"
  LOG_DIR: "/app/artifacts/logs"
  METRICS_DIR: "/app/artifacts/metrics"
  DBT_PROFILES_DIR: "/app/dbt"
  
  # Logging
  LOG_LEVEL: "INFO"
  LOG_FORMAT: "json"
```

### Secrets
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: soma-secrets
  namespace: soma-local
type: Opaque
data:
  # Base64 encoded values
  OPENAI_API_KEY: <base64-encoded-key>
  SESSION_SECRET_KEY: <base64-encoded-secret>
```

## dbt Configuration

### Profiles (`dbt/profiles.yml`)
```yaml
soma_content_analytics:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: ../data/soma.duckdb
      schema: main
      
    prod:
      type: duckdb
      path: /app/data/soma.duckdb
      schema: main
      
    test:
      type: duckdb
      path: :memory:
      schema: main
```

### Project Configuration (`dbt/dbt_project.yml`)
```yaml
name: 'soma_content_analytics'
version: '1.0.0'
config-version: 2

profile: 'soma_content_analytics'

model-paths: ["models"]
analysis-paths: ["analysis"]
test-paths: ["tests"]
seed-paths: ["seeds"]
macro-paths: ["macros"]
snapshot-paths: ["snapshots"]

target-path: "target"
clean-targets:
  - "target"
  - "dbt_packages"

models:
  soma_content_analytics:
    staging:
      +materialized: view
    marts:
      +materialized: table
    ml_features:
      +materialized: table
```

## Machine Learning Configuration

### Model Configuration
```python
# Demand Forecasting Model
DEMAND_MODEL_CONFIG = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'n_jobs': -1
}

# Recommendation Engine
RECOMMENDATION_CONFIG = {
    'similarity_threshold': 0.1,
    'max_recommendations': 100,
    'min_interactions': 5,
    'popularity_window_days': 180
}
```

### Training Configuration
```python
TRAINING_CONFIG = {
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_state': 42,
    'cross_validation_folds': 5,
    'hyperparameter_tuning': True,
    'early_stopping': True
}
```

## Logging Configuration

### Structured Logging
```python
import logging
import sys
from src.utils.helpers import setup_logging

# Configure based on settings
logger = setup_logging(__name__)

# Log levels by environment
LOG_LEVELS = {
    'development': 'DEBUG',
    'testing': 'INFO', 
    'production': 'WARNING'
}
```

### Log Format Examples

#### JSON Format (Production)
```json
{
  "timestamp": "2024-01-19T10:30:00.123Z",
  "level": "INFO",
  "logger": "src.api.app",
  "message": "Application started successfully",
  "environment": "production",
  "version": "1.0.0",
  "request_id": "req_12345"
}
```

#### Text Format (Development)
```
2024-01-19 10:30:00,123 [INFO] src.api.app: Application started successfully
```

## Performance Configuration

### Resource Limits
```python
# Database connection pool
DB_POOL_SIZE = 10
DB_MAX_OVERFLOW = 20
DB_POOL_TIMEOUT = 30

# API request limits
API_RATE_LIMIT = "100/minute"
API_REQUEST_TIMEOUT = 30

# ML model limits
MODEL_PREDICTION_TIMEOUT = 10
MODEL_BATCH_SIZE = 1000
```

### Caching Configuration
```python
# Redis cache settings
CACHE_TYPE = "redis"
CACHE_REDIS_URL = "redis://localhost:6379/0"
CACHE_DEFAULT_TIMEOUT = 300

# Cache keys
CACHE_KEY_PREFIX = "soma:"
CACHE_VERSION = "v1"
```

## Security Configuration

### Authentication (Planned)
```python
# JWT settings
JWT_SECRET_KEY = "${JWT_SECRET_KEY}"
JWT_ACCESS_TOKEN_EXPIRES = 3600  # 1 hour
JWT_REFRESH_TOKEN_EXPIRES = 86400  # 24 hours

# Session settings
SESSION_SECRET_KEY = "${SESSION_SECRET_KEY}"
SESSION_COOKIE_SECURE = True
SESSION_COOKIE_HTTPONLY = True
```

### CORS Settings
```python
# CORS configuration
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:7860",
    "https://your-domain.com"
]
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["GET", "POST", "PUT", "DELETE"]
```

## Monitoring Configuration

### Health Check Settings
```python
# Health check configuration
HEALTH_CHECK_TIMEOUT = 5
HEALTH_CHECK_DATABASE = True
HEALTH_CHECK_MODELS = True
HEALTH_CHECK_EXTERNAL_APIS = False
```

### Metrics Configuration
```python
# Prometheus metrics
METRICS_ENABLED = True
METRICS_PATH = "/metrics"
METRICS_PORT = 8000

# Custom metrics
CUSTOM_METRICS = [
    "prediction_requests_total",
    "prediction_latency_seconds",
    "model_accuracy_score"
]
```

## Validation and Defaults

### Configuration Validation
```python
def validate_config(settings: Settings) -> None:
    """Validate configuration settings"""
    
    # Check required directories exist
    for directory in [settings.MODEL_DIR, settings.LOG_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Validate database path
    db_dir = Path(settings.DB_PATH).parent
    if not db_dir.exists():
        raise ValueError(f"Database directory does not exist: {db_dir}")
    
    # Validate port ranges
    if not (1024 <= settings.API_PORT <= 65535):
        raise ValueError(f"Invalid API port: {settings.API_PORT}")
```

### Environment-Specific Defaults
```python
def get_environment_defaults(environment: str) -> Dict[str, Any]:
    """Get default values for specific environment"""
    
    defaults = {
        'development': {
            'DEBUG': True,
            'LOG_LEVEL': 'DEBUG',
            'API_DEBUG': True
        },
        'production': {
            'DEBUG': False,
            'LOG_LEVEL': 'INFO',
            'API_DEBUG': False
        },
        'testing': {
            'DEBUG': False,
            'LOG_LEVEL': 'WARNING',
            'DB_PATH': ':memory:'
        }
    }
    
    return defaults.get(environment, {})
```

## Configuration Best Practices

### Security
1. **Never commit secrets**: Use environment variables for sensitive data
2. **Use strong defaults**: Secure by default configuration
3. **Validate inputs**: Check configuration values at startup
4. **Rotate secrets**: Regular key rotation procedures

### Maintainability
1. **Document changes**: Comment configuration changes
2. **Version control**: Track configuration changes
3. **Environment parity**: Keep environments consistent
4. **Backup configurations**: Store configuration backups

### Performance
1. **Cache settings**: Cache frequently accessed configuration
2. **Lazy loading**: Load configuration when needed
3. **Optimize defaults**: Choose performance-optimal defaults
4. **Monitor usage**: Track configuration usage patterns

---

*For deployment-specific configuration, see the [Deployment Guide](deployment/README.md)*