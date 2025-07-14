"""Recommendation endpoints."""

import logging

from flask import Blueprint, jsonify, request

from src.models.recommendation.engine import RecommendationEngine

logger = logging.getLogger(__name__)
recommendations_bp = Blueprint("recommendations", __name__)

# Initialize models (lazy loading)
_recommendation_engine = None


def get_recommendation_engine():
    """Get or initialize recommendation engine."""
    global _recommendation_engine
    if _recommendation_engine is None:
        _recommendation_engine = RecommendationEngine()
        _recommendation_engine.load_model()
    return _recommendation_engine


@recommendations_bp.route("/", methods=["POST"])
def get_recommendations():
    """Get book recommendations."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        book_id = data.get("book_id")
        user_type = data.get("user_type")
        n_recommendations = data.get("n_recommendations", 10)

        if not book_id and not user_type:
            return jsonify({"error": "Either book_id or user_type is required"}), 400

        if n_recommendations < 1 or n_recommendations > 50:
            return jsonify({"error": "n_recommendations must be between 1 and 50"}), 400

        # Get engine and make recommendations
        engine = get_recommendation_engine()

        if book_id:
            result = engine.get_similar_books(book_id, n_recommendations)
        else:
            result = engine.get_popular_by_user_type(user_type, n_recommendations)

        if "error" in result:
            return jsonify(result), (
                404 if "not found" in result["error"].lower() else 400
            )

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in recommendations: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@recommendations_bp.route("/similar", methods=["POST"])
def get_similar_books():
    """Get similar books based on content."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        book_id = data.get("book_id")
        n_recommendations = data.get("n_recommendations", 10)

        if not book_id:
            return jsonify({"error": "book_id is required"}), 400

        # Get engine and make recommendations
        engine = get_recommendation_engine()
        result = engine.get_content_based_recommendations(book_id, n_recommendations)

        if "error" in result:
            return jsonify(result), (
                404 if "not found" in result["error"].lower() else 400
            )

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in similar books: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500
