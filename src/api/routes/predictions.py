"""Prediction endpoints."""

import logging

from flask import Blueprint, jsonify, request

from src.models.forecasting.demand_model import DemandForecastingModel

logger = logging.getLogger(__name__)
predictions_bp = Blueprint("predictions", __name__)

# Initialize models (lazy loading)
_demand_model = None


def get_demand_model():
    """Get or initialize demand forecasting model."""
    global _demand_model
    if _demand_model is None:
        _demand_model = DemandForecastingModel()
        _demand_model.load_model()
    return _demand_model


@predictions_bp.route("/demand", methods=["POST"])
def predict_demand():
    """Predict book demand."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        book_id = data.get("book_id")
        days_ahead = data.get("days_ahead", 7)

        if not book_id:
            return jsonify({"error": "book_id is required"}), 400

        if not isinstance(days_ahead, int) or days_ahead < 1 or days_ahead > 30:
            return jsonify({"error": "days_ahead must be between 1 and 30"}), 400

        # Get model and make prediction
        model = get_demand_model()
        result = model.predict_for_book(book_id, days_ahead)

        if "error" in result:
            return jsonify(result), (
                404 if "not found" in result["error"].lower() else 400
            )

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in demand prediction: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@predictions_bp.route("/batch-demand", methods=["POST"])
def predict_batch_demand():
    """Predict demand for multiple books."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        book_ids = data.get("book_ids", [])
        days_ahead = data.get("days_ahead", 7)

        if not book_ids or not isinstance(book_ids, list):
            return jsonify({"error": "book_ids must be a non-empty list"}), 400

        if len(book_ids) > 100:
            return jsonify({"error": "Maximum 100 books per batch request"}), 400

        # Get model and make predictions
        model = get_demand_model()
        results = []

        for book_id in book_ids:
            result = model.predict_for_book(book_id, days_ahead)
            results.append({"book_id": book_id, "prediction": result})

        return jsonify(
            {
                "predictions": results,
                "total_books": len(book_ids),
                "days_ahead": days_ahead,
            }
        )

    except Exception as e:
        logger.error(f"Error in batch demand prediction: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500
