.PHONY: help install install-dev setup clean test lint format docker-build docker-up docker-down data-generate dbt-run api web all

# Default target
help:
	@echo "Available commands:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  setup        Complete setup for development"
	@echo "  clean        Clean up generated files"
	@echo "  test         Run tests"
	@echo "  lint         Run linting"
	@echo "  format       Format code"
	@echo "  data-generate Generate synthetic data"
	@echo "  dbt-run      Run dbt models"
	@echo "  api          Start API server"
	@echo "  web          Start web interface"
	@echo "  all          Full setup and run"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,rag]"
	pre-commit install

# Setup
setup: install-dev
	mkdir -p data/raw data/processed data/external
	mkdir -p artifacts/models artifacts/logs artifacts/metrics
	cp .env.example .env
	@echo "Setup complete! Edit .env file as needed."

# Cleaning
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/
	rm -rf dbt/target/ dbt/logs/

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=html

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-e2e:
	pytest tests/e2e/ -v

# Code quality
lint:
	flake8 src/ tests/
	mypy src/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

# Data pipeline
data-generate:
	python -m src.data.generators

dbt-deps:
	cd dbt && dbt deps

dbt-run: dbt-deps
	cd dbt && dbt run

dbt-test: dbt-run
	cd dbt && dbt test

# Model training
train-models:
	python -m src.models.forecasting.demand_model
	python -m src.models.recommendation.engine
	python -m src.models.rag.system

# Services
api:
	python -m src.api.app

web:
	python -m src.web.gradio_app

# Docker
docker-build:
	docker compose build --no-cache

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

# Full pipeline
all: setup data-generate dbt-run train-models
	@echo "Full setup complete! Run 'make api' or 'make web' to start services."