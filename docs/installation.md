# Installation Guide

This guide covers the installation and setup of the SOMA Content Analytics Platform.

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Kubernetes (kind for local development)
- Git

## Quick Start with Docker Compose

The fastest way to get SOMA running locally:

```bash
# Clone the repository
git clone https://github.com/LNshuti/soma.git
cd soma

# Start services with Docker Compose
docker compose up -d

# Wait for services to be ready
docker compose logs -f
```

Access the application:
- Web Interface: http://localhost:7860
- API: http://localhost:5001

## Local Development Setup

### 1. Environment Setup

```bash
# Create Python virtual environment
conda env create --file=environment.yaml

conda activate soma

# Install dependencies
pip install -r requirements.txt
```

### 2. Database Setup

```bash
# Generate synthetic data
python -m src.data.generators

# Run dbt transformations
cd dbt
dbt run --profiles-dir . --target dev
cd ..
```

### 3. Start Services

```bash
# Start API server
python -m src.api.app

# In another terminal, start web interface
python -m src.web.gradio_app
```

## Kubernetes Deployment

### Prerequisites
- kind cluster (for local development)
- kubectl configured

### Local Kubernetes Setup

```bash
# Clean and setup kind cluster
bash 01_deployment_cleanup_and_setup.sh

# Deploy to kind cluster
bash 02_deployment_local_kind_serve.sh
```

### Access Services

```bash
# Port forward web interface
kubectl port-forward -n soma-local svc/soma-web-service 7860:7860

# Port forward API
kubectl port-forward -n soma-local svc/soma-api-service 5001:5001
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Database
DB_PATH=./data/soma.duckdb

# API Settings
API_DEBUG=true
API_HOST=0.0.0.0
API_PORT=5001

# Web Interface
WEB_HOST=0.0.0.0
WEB_PORT=7860
WEB_SHARE=false

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Environment
ENVIRONMENT=development
```

### Directory Structure

The application expects the following directory structure:

```
soma/
├── src/              # Source code
├── data/             # Database files
├── dbt/              # dbt models and configurations
├── artifacts/        # ML models and logs
├── deployment/       # Kubernetes manifests
├── tests/           # Test suite
└── docs/            # Documentation
```

## Verification

After installation, verify the setup:

```bash
# Check API health
curl http://localhost:5001/health

# Check database
python -c "
import duckdb
conn = duckdb.connect('data/soma.duckdb')
print('Books:', conn.execute('SELECT COUNT(*) FROM raw.books').fetchone()[0])
"

# Run tests
pytest tests/
```

## Troubleshooting

### Common Issues

1. **Database Permission Errors**
   ```bash
   # Fix permissions
   chmod 644 data/soma.duckdb
   ```

2. **Port Already in Use**
   ```bash
   # Find and kill process using port 5001
   lsof -ti:5001 | xargs kill -9
   ```

3. **Docker Build Issues**
   ```bash
   # Clean Docker cache
   docker system prune -a
   ```

4. **Kubernetes Pod Crashes**
   ```bash
   # Check logs
   kubectl logs -n soma-local <pod-name>
   
   # Check events
   kubectl get events -n soma-local --sort-by='.lastTimestamp'
   ```

### Getting Help

- Check the [troubleshooting guide](troubleshooting.md)
- Review application logs
- Submit an issue on GitHub

## Next Steps

- [Architecture Overview](architecture.md)
- [API Documentation](api/README.md)
- [Development Guide](development.md)