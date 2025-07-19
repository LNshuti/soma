# Deployment Guide

This guide covers deployment options for the SOMA Content Analytics Platform across different environments.

## Deployment Options

1. [Docker Compose](docker-compose.md) - Single machine deployment
2. [Kubernetes](kubernetes.md) - Container orchestration
3. [Cloud Deployment](cloud.md) - AWS, GCP, Azure deployment

## Quick Start

### Docker Compose (Recommended for Development)
```bash
# Clone repository
git clone https://github.com/LNshuti/soma.git

cd soma

# Start all services
docker compose up -d

# Access services
# Web Interface: http://localhost:7860
# API: http://localhost:5001
```

### Local Kubernetes (kind)
```bash
# Setup kind cluster
bash 01_deployment_cleanup_and_setup.sh

# Deploy services
bash 02_deployment_local_kind_serve.sh

# Port forward services
kubectl port-forward -n soma-local svc/soma-web-service 7860:7860
kubectl port-forward -n soma-local svc/soma-api-service 5001:5001
```

## Architecture Overview

### Service Components
```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Web Service   │  │   API Service   │  │  Data Setup     │
│   (Gradio)      │  │   (Flask)       │  │  (Job)          │
│   Port: 7860    │  │   Port: 5001    │  │                 │
└─────────┬───────┘  └─────────┬───────┘  └─────────┬───────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                    ┌─────────────────┐
                    │ Persistent      │
                    │ Volume          │
                    │ (Database)      │
                    └─────────────────┘
```

### Container Images
- **soma-web**: Gradio web interface
- **soma-api**: Flask REST API  
- **soma-data-setup**: Data generation and setup

## Configuration Management

### Environment Variables
```bash
# Application settings
ENVIRONMENT=development
DEBUG=true

# Database
DB_PATH=/app/data/soma.duckdb
DBT_PROFILES_DIR=/app/dbt

# API settings
API_HOST=0.0.0.0
API_PORT=5001

# Web interface
WEB_HOST=0.0.0.0
WEB_PORT=7860
WEB_SHARE=false

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### ConfigMaps (Kubernetes)
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: soma-config
  namespace: soma-local
data:
  ENVIRONMENT: "development"
  DB_PATH: "/app/data/soma.duckdb"
  API_HOST: "0.0.0.0"
  API_PORT: "5001"
  WEB_HOST: "0.0.0.0"
  WEB_PORT: "7860"
```

## Storage Configuration

### Persistent Volumes
```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: soma-data-pv
spec:
  capacity:
    storage: 1Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: local-storage
  hostPath:
    path: /tmp/soma-data
```

### Volume Claims
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: soma-data-pvc
  namespace: soma-local
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
  storageClassName: local-storage
```

## Service Configuration

### Web Service
```yaml
apiVersion: v1
kind: Service
metadata:
  name: soma-web-service
  namespace: soma-local
spec:
  selector:
    app: soma-web
  ports:
    - protocol: TCP
      port: 7860
      targetPort: 7860
  type: ClusterIP
```

### API Service
```yaml
apiVersion: v1
kind: Service
metadata:
  name: soma-api-service
  namespace: soma-local
spec:
  selector:
    app: soma-api
  ports:
    - protocol: TCP
      port: 5001
      targetPort: 5001
  type: ClusterIP
```

## Health Monitoring

### Health Checks
```yaml
# Liveness probe
livenessProbe:
  httpGet:
    path: /health
    port: 5001
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

# Readiness probe
readinessProbe:
  httpGet:
    path: /health
    port: 5001
  initialDelaySeconds: 5
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 3
```

### Resource Limits
```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"
```

## Security Configuration

### Pod Security Context
```yaml
securityContext:
  runAsUser: 1000
  runAsGroup: 1000
  fsGroup: 1000
  runAsNonRoot: true
```

### Network Policies
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: soma-network-policy
  namespace: soma-local
spec:
  podSelector:
    matchLabels:
      app: soma
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from: []
    ports:
    - protocol: TCP
      port: 7860
    - protocol: TCP
      port: 5001
```

## Scaling Configuration

### Horizontal Pod Autoscaler
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: soma-api-hpa
  namespace: soma-local
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: soma-api
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Vertical Pod Autoscaler
```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: soma-api-vpa
  namespace: soma-local
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: soma-api
  updatePolicy:
    updateMode: "Auto"
```

## Backup and Disaster Recovery

### Database Backup
```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups/soma"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DB_PATH="/app/data/soma.duckdb"

# Create backup
kubectl exec -n soma-local deployment/soma-api -- \
  cp $DB_PATH "/tmp/soma_backup_${TIMESTAMP}.duckdb"

# Copy to backup location
kubectl cp soma-local/soma-api-pod:/tmp/soma_backup_${TIMESTAMP}.duckdb \
  ${BACKUP_DIR}/soma_backup_${TIMESTAMP}.duckdb

# Cleanup old backups (keep last 7 days)
find $BACKUP_DIR -name "soma_backup_*.duckdb" -mtime +7 -delete
```

### Disaster Recovery Plan
1. **Database Restoration**: Restore from latest backup
2. **Configuration Recovery**: Apply saved ConfigMaps and Secrets
3. **Service Restart**: Redeploy affected services
4. **Validation**: Run health checks and integration tests

## Monitoring and Logging

### Logging Configuration
```yaml
# Fluent Bit DaemonSet for log collection
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluent-bit
  namespace: kube-system
spec:
  selector:
    matchLabels:
      k8s-app: fluent-bit-logging
  template:
    spec:
      containers:
      - name: fluent-bit
        image: fluent/fluent-bit:latest
        ports:
        - containerPort: 2020
        volumeMounts:
        - name: varlog
          mountPath: /var/log
        - name: varlibdockercontainers
          mountPath: /var/lib/docker/containers
          readOnly: true
```

### Metrics Collection
```yaml
# Prometheus ServiceMonitor
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: soma-metrics
  namespace: soma-local
spec:
  selector:
    matchLabels:
      app: soma
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

## Deployment Scripts

### Automated Deployment
```bash
#!/bin/bash
# deploy.sh - Automated deployment script

set -e

NAMESPACE="soma-local"
ENVIRONMENT=${1:-development}

echo "Deploying SOMA to $ENVIRONMENT environment..."

# Create namespace
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply configurations
kubectl apply -f deployment/k8s/local/

# Wait for deployments
kubectl wait --for=condition=available --timeout=300s \
  deployment/soma-api deployment/soma-web -n $NAMESPACE

# Run data setup job
kubectl apply -f deployment/k8s/local/data-setup-job.yaml

# Wait for job completion
kubectl wait --for=condition=complete --timeout=600s \
  job/soma-data-setup -n $NAMESPACE

echo "Deployment completed successfully!"
```

### Rollback Script
```bash
#!/bin/bash
# rollback.sh - Rollback to previous version

NAMESPACE="soma-local"
DEPLOYMENT=${1:-soma-api}

echo "Rolling back $DEPLOYMENT..."

# Rollback deployment
kubectl rollout undo deployment/$DEPLOYMENT -n $NAMESPACE

# Wait for rollback
kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE

echo "Rollback completed!"
```

## CI/CD Integration

### GitHub Actions
```yaml
name: Deploy to Kubernetes
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Build and push images
      run: |
        docker build -t soma-api .
        docker build -t soma-web -f Dockerfile.web .
    
    - name: Deploy to cluster
      run: |
        kubectl apply -f deployment/k8s/local/
        kubectl wait --for=condition=available deployment/soma-api deployment/soma-web
```

## Troubleshooting

### Common Issues

1. **Pod Startup Failures**
   ```bash
   # Check pod status
   kubectl get pods -n soma-local
   
   # View pod logs
   kubectl logs -f deployment/soma-api -n soma-local
   
   # Describe pod for events
   kubectl describe pod <pod-name> -n soma-local
   ```

2. **Service Connection Issues**
   ```bash
   # Test service connectivity
   kubectl exec -it deployment/soma-web -n soma-local -- \
     curl http://soma-api-service:5001/health
   
   # Check service endpoints
   kubectl get endpoints -n soma-local
   ```

3. **Storage Issues**
   ```bash
   # Check PV/PVC status
   kubectl get pv,pvc -n soma-local
   
   # Check volume mounts
   kubectl describe pod <pod-name> -n soma-local
   ```

### Debug Commands
```bash
# Get all resources
kubectl get all -n soma-local

# Check events
kubectl get events -n soma-local --sort-by='.lastTimestamp'

# Shell into pod
kubectl exec -it deployment/soma-api -n soma-local -- /bin/bash

# Port forward for debugging
kubectl port-forward deployment/soma-api 5001:5001 -n soma-local
```

---

*For specific deployment instructions, see the individual deployment guides*