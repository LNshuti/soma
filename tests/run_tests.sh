#!/bin/bash
# tests/run_tests.sh

set -e

echo "Running Soma ML Platform Tests"

# Set up test environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export ENVIRONMENT="testing"

# Run different test suites
echo "Running Unit Tests..."
pytest tests/unit/ -m unit -v

echo "Running Integration Tests..."  
pytest tests/integration/ -m integration -v

echo "Running E2E Tests..."
pytest tests/e2e/ -m e2e -v --tb=short

echo "Generating Coverage Report..."
pytest tests/ --cov=src --cov-report=html --cov-report=term

echo "All tests completed!"
