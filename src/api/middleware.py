"""Middleware for request/response processing."""

import logging
import time

from flask import Flask, g, request
from werkzeug.exceptions import HTTPException

logger = logging.getLogger(__name__)


def setup_middleware(app: Flask) -> None:
    """Setup middleware for the Flask app."""

    @app.before_request
    def before_request():
        """Before request processing."""
        g.start_time = time.time()
        logger.debug(f"Request started: {request.method} {request.path}")

    @app.after_request
    def after_request(response):
        """After request processing."""
        duration = time.time() - g.get("start_time", time.time())
        logger.info(
            f"Request completed: {request.method} {request.path} "
            f"- Status: {response.status_code} - Duration: {duration:.3f}s"
        )

        # Add response headers
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        response.headers["X-API-Version"] = "1.0"

        return response

    @app.errorhandler(Exception)
    def handle_exception(e):
        """Global exception handler."""
        if isinstance(e, HTTPException):
            logger.warning(f"HTTP exception: {e}")
            return {"error": e.description, "code": e.code}, e.code

        logger.error(f"Unhandled exception: {e}", exc_info=True)
        return {"error": "Internal server error", "code": 500}, 500
