"""Health check endpoints."""

from datetime import datetime

from flask import Blueprint, jsonify

from config.settings import settings
from src.utils.database import DatabaseManager

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health_check():
    """Comprehensive health check."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "components": {},
    }

    # Check database
    try:
        db_manager = DatabaseManager()
        with db_manager.get_connection() as conn:
            conn.execute("SELECT 1").fetchone()
        health_status["components"]["database"] = {"status": "healthy"}
    except Exception as e:
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "error": str(e),
        }
        health_status["status"] = "degraded"

    # Check model artifacts
    model_dir = settings.models.model_dir
    if model_dir.exists():
        model_count = len(list(model_dir.glob("*/model.pkl")))
        health_status["components"]["models"] = {
            "status": "healthy" if model_count > 0 else "warning",
            "loaded_models": str(model_count),
        }
    else:
        health_status["components"]["models"] = {
            "status": "warning",
            "message": "Model directory not found",
        }

    status_code = 200 if health_status["status"] == "healthy" else 503
    return jsonify(health_status), status_code


@health_bp.route("/ready", methods=["GET"])
def readiness_check():
    """Readiness probe for Kubernetes."""
    return jsonify({"status": "ready"}), 200


@health_bp.route("/live", methods=["GET"])
def liveness_check():
    """Liveness probe for Kubernetes."""
    return jsonify({"status": "alive"}), 200
