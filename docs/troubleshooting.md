# Troubleshooting Guide

This guide helps diagnose and resolve common issues with the SOMA Content Analytics Platform.

## Quick Diagnostics

### System Health Check
```bash
# Check API health
curl http://localhost:5001/health

# Check web interface
curl http://localhost:7860

# Check database connectivity
python -c "
import duckdb
conn = duckdb.connect('data/soma.duckdb')
print('Books:', conn.execute('SELECT COUNT(*) FROM raw.books').fetchone()[0])
"
```

### Service Status
```bash
# Docker Compose
docker compose ps

# Kubernetes
kubectl get pods -n soma-local
kubectl get services -n soma-local
```

## Common Issues

### 1. Application Won't Start

#### Symptoms
- Services fail to start
- Connection refused errors
- Import errors

#### Diagnosis
```bash
# Check logs
docker compose logs api
docker compose logs web

# Kubernetes logs
kubectl logs deployment/soma-api -n soma-local
kubectl logs deployment/soma-web -n soma-local
```

#### Solutions

**Missing Dependencies**
```bash
# Rebuild images
docker compose build --no-cache

# Reinstall Python dependencies
pip install -r requirements.txt
```

**Port Conflicts**
```bash
# Find processes using ports
lsof -ti:5001 | xargs kill -9  # API port
lsof -ti:7860 | xargs kill -9  # Web port

# Use different ports
API_PORT=5002 WEB_PORT=7861 docker compose up
```

**Environment Variables**
```bash
# Check environment
docker compose config

# Verify .env file
cat .env
```

### 2. Database Issues

#### Symptoms
- "Database file not found" errors
- "Permission denied" errors
- Database corruption warnings

#### Diagnosis
```bash
# Check database file
ls -la data/soma.duckdb

# Test database connectivity
python -c "
import duckdb
try:
    conn = duckdb.connect('data/soma.duckdb')
    tables = conn.execute('SHOW TABLES').fetchall()
    print('Tables:', tables)
except Exception as e:
    print('Error:', e)
"
```

#### Solutions

**File Permissions**
```bash
# Fix permissions
chmod 644 data/soma.duckdb
chown $USER:$USER data/soma.duckdb

# Docker permissions
docker run --rm -v $(pwd)/data:/data alpine chown -R 1000:1000 /data
```

**Regenerate Database**
```bash
# Remove corrupted database
rm -f data/soma.duckdb

# Regenerate data
python -m src.data.generators

# Run dbt transformations
cd dbt && dbt run --profiles-dir . --target dev
```

**Docker Volume Issues**
```bash
# Remove and recreate volumes
docker compose down -v
docker compose up -d
```

### 3. API Issues

#### Symptoms
- 503 Service Unavailable
- Slow response times
- Model prediction errors

#### Diagnosis
```bash
# Check API logs
curl -v http://localhost:5001/health

# Test specific endpoints
curl http://localhost:5001/api/recommendations/popular/Individual

# Check model availability
ls -la artifacts/models/
```

#### Solutions

**Model Loading Issues**
```bash
# Retrain models
python -m src.models.forecasting.demand_model
python -m src.models.recommendation.engine

# Check model artifacts
ls -la artifacts/models/*/
```

**Database Connection Issues**
```bash
# Test database connection
python -c "
from src.utils.database import DatabaseManager
db = DatabaseManager()
try:
    conn = db.get_connection()
    print('Database connected successfully')
except Exception as e:
    print('Database error:', e)
"
```

**Memory Issues**
```bash
# Check memory usage
docker stats

# Increase memory limits (docker-compose.yml)
deploy:
  resources:
    limits:
      memory: 2G
```

### 4. Web Interface Issues

#### Symptoms
- White screen or blank page
- Interface not responsive
- Charts not displaying

#### Diagnosis
```bash
# Check web service logs
docker compose logs web

# Test web service directly
curl http://localhost:7860

# Check browser console for JavaScript errors
```

#### Solutions

**Browser Compatibility**
- Use Chrome 90+ or Firefox 88+
- Clear browser cache and cookies
- Disable ad blockers temporarily

**JavaScript Errors**
```bash
# Check Gradio version
pip show gradio

# Update dependencies
pip install --upgrade gradio plotly pandas
```

**API Connectivity**
```bash
# Test API from web container
docker compose exec web curl http://api:5001/health

# Check network connectivity
docker compose exec web ping api
```

### 5. Kubernetes Issues

#### Symptoms
- Pods stuck in pending state
- CrashLoopBackOff errors
- Service unavailable

#### Diagnosis
```bash
# Check pod status
kubectl get pods -n soma-local

# Check events
kubectl get events -n soma-local --sort-by='.lastTimestamp'

# Describe problematic pods
kubectl describe pod <pod-name> -n soma-local

# Check logs
kubectl logs <pod-name> -n soma-local
```

#### Solutions

**Resource Issues**
```bash
# Check node resources
kubectl top nodes
kubectl describe nodes

# Adjust resource requests/limits
kubectl edit deployment soma-api -n soma-local
```

**Image Pull Issues**
```bash
# Load images into kind
kind load docker-image soma-api:latest
kind load docker-image soma-web:latest

# Check image availability
docker images | grep soma
```

**Storage Issues**
```bash
# Check PVC status
kubectl get pvc -n soma-local

# Recreate PVC if needed
kubectl delete pvc soma-data-pvc -n soma-local
kubectl apply -f deployment/k8s/local/pvc.yaml
```

### 6. Data Pipeline Issues

#### Symptoms
- dbt run failures
- Data generation errors
- Missing or incorrect data

#### Diagnosis
```bash
# Test data generation
python -m src.data.generators

# Run dbt with debug
cd dbt
dbt run --debug --profiles-dir . --target dev

# Check dbt logs
cat dbt/logs/dbt.log
```

#### Solutions

**dbt Compilation Errors**
```bash
# Test dbt compilation
dbt compile --profiles-dir . --target dev

# Check model dependencies
dbt list --profiles-dir . --target dev

# Validate SQL syntax
dbt parse --profiles-dir . --target dev
```

**Data Quality Issues**
```bash
# Run dbt tests
dbt test --profiles-dir . --target dev

# Check data freshness
dbt source freshness --profiles-dir . --target dev
```

**Profile Configuration**
```bash
# Verify dbt profiles
cat dbt/profiles.yml

# Test connection
dbt debug --profiles-dir . --target dev
```

### 7. Performance Issues

#### Symptoms
- Slow application response
- High memory usage
- CPU bottlenecks

#### Diagnosis
```bash
# Monitor resource usage
docker stats

# Check system resources
top
htop
free -h
df -h

# Profile Python applications
python -m cProfile -o profile.stats -m src.api.app
```

#### Solutions

**Database Optimization**
```sql
-- Add indexes for common queries
CREATE INDEX idx_sales_date ON raw.sales(sale_date);
CREATE INDEX idx_sales_book ON raw.sales(book_id);
CREATE INDEX idx_books_genre ON raw.books(genre);
```

**Memory Optimization**
```python
# Reduce batch sizes
BATCH_SIZE = 100  # Instead of 1000

# Use generators for large datasets
def process_data():
    for chunk in pd.read_csv('large_file.csv', chunksize=1000):
        yield process_chunk(chunk)
```

**Caching**
```bash
# Install Redis for caching
docker run -d --name redis -p 6379:6379 redis:latest

# Configure application caching
CACHE_TYPE=redis
CACHE_REDIS_URL=redis://localhost:6379/0
```

## Debugging Tools

### Application Debugging

#### Enable Debug Mode
```bash
# Environment variable
DEBUG=true python -m src.api.app

# Docker Compose
DEBUG=true docker compose up
```

#### Python Debugger
```python
# Add breakpoints in code
import pdb; pdb.set_trace()

# Remote debugging with debugpy
import debugpy
debugpy.listen(('0.0.0.0', 5678))
debugpy.wait_for_client()
```

### Database Debugging

#### DuckDB CLI
```bash
# Connect to database
duckdb data/soma.duckdb

# Useful queries
.tables
.schema raw.books
SELECT COUNT(*) FROM raw.sales;
PRAGMA table_info('raw.books');
```

#### Query Analysis
```sql
-- Explain query plans
EXPLAIN SELECT * FROM fact_sales WHERE sale_date > '2024-01-01';

-- Check table statistics
SELECT COUNT(*), MIN(sale_date), MAX(sale_date) FROM raw.sales;
```

### Kubernetes Debugging

#### Pod Debugging
```bash
# Shell into pod
kubectl exec -it deployment/soma-api -n soma-local -- /bin/bash

# Copy files from pod
kubectl cp soma-local/soma-api-pod:/app/data/soma.duckdb ./debug.duckdb

# Port forward for debugging
kubectl port-forward deployment/soma-api 5001:5001 -n soma-local
```

#### Network Debugging
```bash
# Test service connectivity
kubectl run debug --image=alpine --rm -it -- sh
/ # wget -qO- http://soma-api-service.soma-local:5001/health

# Check DNS resolution
/ # nslookup soma-api-service.soma-local
```

## Error Messages and Solutions

### Common Error Messages

#### "ImportError: No module named 'src'"
```bash
# Solution: Set PYTHONPATH
export PYTHONPATH="/app:$PYTHONPATH"

# Or run from project root
cd /path/to/soma
python -m src.api.app
```

#### "PermissionError: [Errno 13] Permission denied"
```bash
# Solution: Fix file permissions
chmod 644 data/soma.duckdb
chown $USER:$USER data/soma.duckdb

# Docker: Run as correct user
docker run --user $(id -u):$(id -g) ...
```

#### "ConnectionError: Database is locked"
```bash
# Solution: Close other connections
pkill -f python  # Kill other Python processes
rm -f data/soma.duckdb-wal  # Remove WAL file if safe
```

#### "ValidationError: Invalid configuration"
```bash
# Solution: Check environment variables
env | grep -E "(API_|WEB_|DB_)"

# Validate configuration
python -c "from config.settings import settings; print(settings)"
```

## Prevention and Monitoring

### Health Monitoring
```bash
# Setup monitoring script
#!/bin/bash
while true; do
    if ! curl -s http://localhost:5001/health > /dev/null; then
        echo "$(date): API health check failed"
        # Send alert or restart service
    fi
    sleep 60
done
```

### Log Monitoring
```bash
# Monitor for errors
tail -f artifacts/logs/app.log | grep -i error

# Setup log rotation
logrotate -f /etc/logrotate.d/soma
```

### Automated Testing
```bash
# Regular health checks
pytest tests/integration/test_health.py

# Data quality tests
dbt test --profiles-dir dbt --target dev

# Performance benchmarks
python tests/performance/benchmark.py
```

## Getting Help

### Information to Gather
When reporting issues, include:

1. **Environment Details**
   ```bash
   # System information
   uname -a
   docker --version
   python --version
   
   # Application versions
   pip show gradio flask duckdb dbt-core
   
   # Configuration
   env | grep -E "(API_|WEB_|DB_|ENVIRONMENT)"
   ```

2. **Error Logs**
   ```bash
   # Application logs
   docker compose logs --tail=100
   
   # System logs
   journalctl -u docker --since="1 hour ago"
   ```

3. **Resource Usage**
   ```bash
   # System resources
   free -h
   df -h
   docker stats --no-stream
   ```

### Support Channels
- **Documentation**: Check this troubleshooting guide
- **Logs**: Review application and system logs
- **GitHub Issues**: Create detailed issue reports
- **Community**: Join project discussions

### Escalation Process
1. **Self-diagnosis**: Follow this troubleshooting guide
2. **Log analysis**: Examine detailed error logs
3. **Minimal reproduction**: Create minimal failing example
4. **Issue reporting**: Submit GitHub issue with details
5. **Community support**: Engage with project community

---

*For specific deployment issues, see the [Deployment Guide](deployment/README.md)*