# src/api/routes/content.py
"""Content generation endpoints using RAG system."""

import logging

from flask import Blueprint, jsonify, request

from src.models.rag.system import RAGSystem

logger = logging.getLogger(__name__)
content_bp = Blueprint("content", __name__)

# Initialize RAG system (lazy loading)
_rag_system = None


def get_rag_system():
    """Get or initialize RAG system."""
    global _rag_system
    if _rag_system is None:
        _rag_system = RAGSystem()
        _rag_system.initialize()
    return _rag_system


@content_bp.route("/ad-copy", methods=["POST"])
def generate_ad_copy():
    """Generate advertising copy for a book."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        book_id = data.get("book_id")
        ad_type = data.get("ad_type", "social_media")
        target_audience = data.get("target_audience")

        if not book_id:
            return jsonify({"error": "book_id is required"}), 400

        if ad_type not in ["social_media", "email", "display", "print"]:
            return jsonify({"error": "Invalid ad_type"}), 400

        # Get RAG system and generate ad copy
        rag = get_rag_system()
        result = rag.generate_ad_copy(book_id, ad_type, target_audience)

        if "error" in result:
            return jsonify(result), (
                404 if "not found" in result["error"].lower() else 400
            )

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in ad copy generation: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@content_bp.route("/image-prompts", methods=["POST"])
def generate_image_prompts():
    """Generate AI image prompts for a book."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        book_id = data.get("book_id")
        style = data.get("style", "modern")

        if not book_id:
            return jsonify({"error": "book_id is required"}), 400

        if style not in [
            "modern",
            "vintage",
            "minimalist",
            "dramatic",
            "artistic",
            "commercial",
        ]:
            return jsonify({"error": "Invalid style"}), 400

        # Get RAG system and generate prompts
        rag = get_rag_system()
        result = rag.generate_image_prompts(book_id, style)

        if "error" in result:
            return jsonify(result), (
                404 if "not found" in result["error"].lower() else 400
            )

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in image prompt generation: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@content_bp.route("/recommendation-text", methods=["POST"])
def generate_recommendation_text():
    """Generate recommendation text for a book."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        book_id = data.get("book_id")
        user_context = data.get("user_context", {})

        if not book_id:
            return jsonify({"error": "book_id is required"}), 400

        # Get RAG system and generate recommendation
        rag = get_rag_system()
        result = rag.generate_recommendation(book_id, user_context)

        if "error" in result:
            return jsonify(result), (
                404 if "not found" in result["error"].lower() else 400
            )

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in recommendation generation: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500
