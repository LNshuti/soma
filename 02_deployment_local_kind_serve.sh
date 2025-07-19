#!/bin/bash
set -euo pipefail

CLUSTER_NAME="kind"
NAMESPACE="soma-local"

echo "Verifying kind cluster exists..."
if ! kind get clusters | grep -q "^$CLUSTER_NAME$"; then
  echo "Kind cluster '$CLUSTER_NAME' not found. Please run 01_deployment_cleanup_and_setup.sh first"
  exit 1
fi

echo "Building Docker images..."
docker compose build

echo "Loading Docker images into kind cluster..."
kind load docker-image soma-web:latest --name "$CLUSTER_NAME"
kind load docker-image soma-api:latest --name "$CLUSTER_NAME"

echo "Deploying to Kubernetes in namespace '$NAMESPACE'..."

# Apply configurations in proper order
echo "Creating namespace..."
kubectl apply -f deployment/k8s/local/namespace.yaml

echo "Setting up storage..."
kubectl apply -f deployment/k8s/local/persistent-volume.yaml
kubectl apply -f deployment/k8s/local/pvc.yaml

echo "Applying configurations..."
kubectl apply -f deployment/k8s/local/configmap.yaml

echo "Running data setup job..."
kubectl apply -f deployment/k8s/local/data-setup-job.yaml

echo "Waiting for data-setup job to complete..."
kubectl wait --for=condition=complete --timeout=300s job/soma-data-setup -n "$NAMESPACE"

echo "Checking data setup logs..."
kubectl logs job/soma-data-setup -n "$NAMESPACE"

echo "Deploying services..."
kubectl apply -f deployment/k8s/local/api-deployment.yaml
kubectl apply -f deployment/k8s/local/api-service.yaml
kubectl apply -f deployment/k8s/local/web-deployment.yaml
kubectl apply -f deployment/k8s/local/web-service.yaml

echo "Waiting for deployments to become available..."
kubectl wait --for=condition=available --timeout=300s deployment/soma-api -n "$NAMESPACE"
kubectl wait --for=condition=available --timeout=300s deployment/soma-web -n "$NAMESPACE"

echo "Setting up port-forwarding..."

cleanup() {
  echo "Cleaning up background processes..."
  kill $(jobs -p) 2>/dev/null || true
  exit
}
trap cleanup SIGINT SIGTERM

kubectl port-forward service/soma-web-service 7860:7860 -n "$NAMESPACE" &
kubectl port-forward service/soma-api-service 5001:5001 -n "$NAMESPACE" &

echo "Deployment complete!"
echo "Web interface: http://localhost:7860"
echo "API available at: http://localhost:5001"
echo "Press Ctrl+C to stop."

wait
