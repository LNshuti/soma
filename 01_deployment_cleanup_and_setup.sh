#!/bin/bash
set -e

echo "Cleaning up existing resources..."

# Stop existing port-forwards
echo "Stopping existing port-forwards..."
pkill -f "kubectl port-forward" || true
pkill -f "port-forward" || true

# Clean up Kubernetes resources first
echo "Cleaning up Kubernetes resources..."
kubectl delete namespace soma-local --ignore-not-found=true --timeout=60s || true
kubectl delete pv soma-data-pv --ignore-not-found=true || true

# Clean up kind cluster
echo "Cleaning up existing kind cluster..."
kind delete cluster --name kind || true

# Clean up Docker resources
echo "Cleaning up Docker resources..."
docker system prune -f
docker volume prune -f

# Clean up temporary files and build artifacts
echo "Cleaning up temporary files..."
rm -rf .pytest_cache/
rm -rf __pycache__/
rm -rf .coverage
rm -rf htmlcov/
rm -rf dist/
rm -rf build/
rm -rf *.egg-info/
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Clean up local data files
rm -rf data/processed/
rm -rf data/raw/
rm -rf artifacts/
rm -rf logs/

# Clean up any Docker build cache
docker builder prune -f

echo "Cleanup complete!"

echo "Setting up fresh environment..."

# Create kind cluster
echo "Creating kind cluster..."
kind create cluster --name kind --wait 5m

echo "Waiting for cluster to be ready..."
kubectl wait --for=condition=Ready nodes --all --timeout=300s

echo "Fresh environment setup complete!"