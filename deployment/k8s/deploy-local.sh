#!/bin/bash
set -e

echo "Deploying Soma to local Kubernetes..."

# Build Docker images
echo "Building Docker images..."
docker compose build

# Apply Kubernetes configurations
echo "Applying Kubernetes configurations..."
kubectl apply -f deployment/k8s/local/namespace.yaml
kubectl apply -f deployment/k8s/local/configmap.yaml
kubectl apply -f deployment/k8s/local/data-setup-job.yaml
kubectl apply -f deployment/k8s/local/api-deployment.yaml
kubectl apply -f deployment/k8s/local/web-deployment.yaml

echo "Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/soma-api -n soma-local
kubectl wait --for=condition=available --timeout=300s deployment/soma-web -n soma-local

echo "Deployment complete!"
echo "Access the web interface at: http://localhost:7860"
echo "API available at: http://localhost:5001"

# Port forward for local access
kubectl port-forward service/soma-web-service 7860:7860 -n soma-local &
kubectl port-forward service/soma-api-service 5001:5001 -n soma-local &

echo "Port forwarding started. Press Ctrl+C to stop."
wait
