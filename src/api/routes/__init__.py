# src/api/routes/__init__.py
"""API routes package."""

from flask import Flask

from .content import content_bp
from .health import health_bp
from .predictions import predictions_bp
from .recommendations import recommendations_bp


def register_routes(app: Flask) -> None:
    """Register all route blueprints."""
    app.register_blueprint(health_bp)
    app.register_blueprint(predictions_bp, url_prefix="/predict")
    app.register_blueprint(recommendations_bp, url_prefix="/recommend")
    app.register_blueprint(content_bp, url_prefix="/generate")
