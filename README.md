# Soma ML Platform

 This project consists of a machine learning platform for book recommendations, and Retrieval Augmented Generation(RAG) powered advertising content generation based on synthetic book metadata. 

## Features

- **Recommendation Engine**: Hybrid collaborative and content-based recommendations  
- **RAG System**: AI-powered ad copy and image prompt generation
- **Data Pipeline**: dbt-powered data transformation and modeling
- **Web Interface**: Interactive Gradio-based dashboard

## Quick Start

### Prerequisites
- Python 3.9+
- Docker (optional)

### Installation

1. **Clone and setup**
```bash
git clone https://github.com/LNshuti/soma.git
cd soma
conda env create --file=environment.yaml
conda activate soma
pip install -e .
```

2. **Configure environment**
```bash
cp .env.example .env

```

3. **Generate synthetic data**
```bash
python generate_data.py
```

4. **Build data models**
```bash
cd dbt
dbt deps
```

**Set required environment variable**
```bash
set -o allexport; source ../.env; set +o allexport; 
```

**Run dbt models** 
```bash
cd ..
dbt run
dbt test
```

5. **Train ML models**
```bash
python -m src.models.forecasting.demand_model
python -m src.models.recommendation.engine
```

6. **Start services**
```bash
python -m src.api.app &
python -m src.web.gradio_app
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│   dbt Models    │───▶│   ML Models     │
│   (Synthetic)   │    │   (Transform)   │    │ (Train/Predict) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐              │
│   Web Interface │◀───│   REST APIs     │◀─────────────┘
│    (Gradio)     │    │    (Flask)      │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────┬───────────────┘
                 │
    ┌─────────────────┐    ┌─────────────────┐
    │  Background     │    │    Database     │
    │   Workers       │    │   (DuckDB)      │
    │   (Celery)      │    │                 │
    └─────────────────┘    └─────────────────┘
```

## API Endpoints

- `GET /health` - System health check
- `POST /api/v1/predictions/demand` - Demand forecasting
- `POST /api/v1/recommendations` - Book recommendations
- `POST /api/v1/rag/ad-copy` - Generate ad copy
- `POST /api/v1/rag/image-prompts` - Generate image prompts

## Testing

```bash
pytest
pytest --cov=src tests/
```

## Deployment

### Development
```bash
make setup-dev
make run-api
```

### Production
```bash
docker compose -f docker-compose.yml up -d
```

```bash
# Build and start all services
docker compose up --build

# Run in background
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```