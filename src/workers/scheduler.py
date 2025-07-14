"""Task scheduler for periodic background jobs."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict

from celery import Celery
from celery.schedules import crontab

from .celery import celery_app

logger = logging.getLogger(__name__)


class TaskScheduler:
    """Manages scheduled tasks and periodic jobs."""

    def __init__(self, celery_app: Celery):
        self.celery_app = celery_app
        self.setup_periodic_tasks()

    def setup_periodic_tasks(self):
        """Configure periodic tasks."""

        # Configure Celery Beat schedule
        self.celery_app.conf.beat_schedule = {
            # Data refresh tasks
            "refresh_data_daily": {
                "task": "src.workers.tasks.data_tasks.refresh_synthetic_data",
                "schedule": crontab(hour=2, minute=0),  # Daily at 2 AM
            },
            # Model retraining
            "retrain_models_weekly": {
                "task": "src.workers.tasks.ml_tasks.retrain_all_models",
                "schedule": crontab(
                    hour=3, minute=0, day_of_week=1
                ),  # Weekly on Monday
            },
            # Data quality checks
            "data_quality_check": {
                "task": "src.workers.tasks.data_tasks.run_data_quality_checks",
                "schedule": crontab(hour=1, minute=30),  # Daily at 1:30 AM
            },
            # Model performance monitoring
            "model_performance_check": {
                "task": "src.workers.tasks.ml_tasks.monitor_model_performance",
                "schedule": crontab(minute="*/30"),  # Every 30 minutes
            },
            # RAG system maintenance
            "update_rag_embeddings": {
                "task": "src.workers.tasks.rag_tasks.update_embeddings",
                "schedule": crontab(hour=4, minute=0),  # Daily at 4 AM
            },
            # Cleanup tasks
            "cleanup_old_artifacts": {
                "task": "src.workers.tasks.maintenance.cleanup_old_files",
                "schedule": crontab(
                    hour=5, minute=0, day_of_week=0
                ),  # Weekly on Sunday
            },
        }

        self.celery_app.conf.timezone = "UTC"

    def schedule_custom_task(
        self, task_name: str, task_path: str, schedule: Dict[str, Any]
    ) -> bool:
        """Schedule a custom task."""
        try:
            self.celery_app.conf.beat_schedule[task_name] = {
                "task": task_path,
                "schedule": schedule,
            }
            logger.info(f"Scheduled custom task: {task_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to schedule task {task_name}: {e}")
            return False

    def get_scheduled_tasks(self) -> Dict[str, Any]:
        """Get all scheduled tasks."""
        return self.celery_app.conf.beat_schedule


# Create global scheduler instance
task_scheduler = TaskScheduler(celery_app)
