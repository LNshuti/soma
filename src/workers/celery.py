"""Celery configuration for distributed task processing."""

import os

from celery import Celery
from kombu import Queue

from config.settings import get_settings

settings = get_settings()

# Initialize Celery app
celery_app = Celery(
    "soma_ml_platform",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "src.workers.tasks.data_tasks",
        "src.workers.tasks.ml_tasks",
        "src.workers.tasks.rag_tasks",
    ],
)

# Configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_ignore_result=False,
    result_expires=3600,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_hijack_root_logger=False,
    task_default_queue="default",
    task_routes={
        "src.workers.tasks.data_tasks.*": {"queue": "data_processing"},
        "src.workers.tasks.ml_tasks.*": {"queue": "ml_training"},
        "src.workers.tasks.rag_tasks.*": {"queue": "rag_generation"},
    },
    task_queues=(
        Queue("default", routing_key="default"),
        Queue("data_processing", routing_key="data_processing"),
        Queue("ml_training", routing_key="ml_training"),
        Queue("rag_generation", routing_key="rag_generation"),
    ),
)


# Health check task
@celery_app.task(bind=True)
def health_check(self):
    """Health check task for monitoring."""
    return {"status": "healthy", "worker_id": self.request.id}


if __name__ == "__main__":
    celery_app.start()
