#!/bin/bash
set -e

echo "Starting Soma ML Workers..."

# Wait for Redis
echo "Waiting for Redis..."
while ! python -c "import redis; redis.Redis(host='redis', port=6379).ping()"; do
    sleep 2
done

# Start Celery worker
echo "Starting Celery worker..."
exec celery -A src.workers.celery worker --loglevel=info --concurrency=4