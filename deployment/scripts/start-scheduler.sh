#!/bin/bash
set -e

echo "Starting Soma ML Scheduler..."

# Wait for Redis
echo "Waiting for Redis..."
while ! python -c "import redis; redis.Redis(host='redis', port=6379).ping()"; do
    sleep 2
done

# Start Celery beat scheduler
echo " Starting Celery beat..."
exec celery -A src.workers.celery beat --loglevel=info