# src/models/rag/system.py
"""RAG System for content generation and recommendations."""

import logging
from typing import Any, Dict, List, Optional

from src.models.rag.generators import (
    AdCopyGenerator,
    BookRecommendationGenerator,
    ContentGenerator,
    ImagePromptGenerator,
)
from src.utils.database import DatabaseManager
from src.utils.helpers import setup_logging

logger = setup_logging(__name__)


class RAGSystem:
    """RAG (Retrieval-Augmented Generation) System for content creation"""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager or DatabaseManager()
        self.ad_generator = AdCopyGenerator(self.db_manager)
        self.image_generator = ImagePromptGenerator(self.db_manager)
        self.recommendation_generator = BookRecommendationGenerator(self.db_manager)
        self.is_initialized = False

    def initialize(self) -> bool:
        """Initialize the RAG system"""
        try:
            self.ad_generator.initialize()
            self.image_generator.initialize()
            self.recommendation_generator.initialize()
            self.is_initialized = True
            logger.info("RAG System initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            return False

    def generate_ad_copy(
        self,
        book_id: str,
        ad_type: str = "social_media",
        target_audience: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate advertising copy for a book"""
        if not self.is_initialized:
            self.initialize()

        return self.ad_generator.generate_ad_copy(book_id, ad_type, target_audience)

    def generate_image_prompts(
        self, book_id: str, style: str = "modern"
    ) -> Dict[str, Any]:
        """Generate AI image prompts for a book"""
        if not self.is_initialized:
            self.initialize()

        return self.image_generator.generate_image_prompts(book_id, style)

    def generate_recommendation(
        self, book_id: str, user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate book recommendation text"""
        if not self.is_initialized:
            self.initialize()

        try:
            # Get book info
            query = "SELECT * FROM dim_books WHERE book_id = ?"
            result = self.db_manager.fetch_dataframe(query, [book_id])

            if result.empty:
                return {"error": f"Book {book_id} not found"}

            book_info = result.iloc[0].to_dict()
            user_context = user_context or {"user_type": "general"}

            recommendation_text = (
                self.recommendation_generator.generate_recommendation_text(
                    book_info, user_context
                )
            )

            return {
                "book_id": book_id,
                "recommendation_text": recommendation_text,
                "book_info": book_info,
                "user_context": user_context,
            }

        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return {"error": str(e)}

    def get_context_for_book(self, book_id: str) -> List[str]:
        """Get contextual information for a book"""
        try:
            # Get book details
            query = """
            SELECT b.title, b.author, b.genre, b.price,
                   p.publisher_name, p.publisher_type
            FROM dim_books b
            JOIN dim_publishers p ON b.publisher_id = p.publisher_id
            WHERE b.book_id = ?
            """
            result = self.db_manager.fetch_dataframe(query, [book_id])

            if result.empty:
                return []

            book = result.iloc[0]
            context = [
                f"Title: {book['title']}",
                f"Author: {book['author']}",
                f"Genre: {book['genre']}",
                f"Publisher: {book['publisher_name']} ({book['publisher_type']})",
                f"Price: ${book['price']:.2f}",
            ]

            # Get similar books in same genre
            similar_query = """
            SELECT title, author 
            FROM dim_books 
            WHERE genre = ? AND book_id != ?
            LIMIT 3
            """
            similar = self.db_manager.fetch_dataframe(
                similar_query, [book["genre"], book_id]
            )

            if not similar.empty:
                similar_titles = [
                    f"{row['title']} by {row['author']}"
                    for _, row in similar.iterrows()
                ]
                context.append(f"Similar books: {', '.join(similar_titles)}")

            return context

        except Exception as e:
            logger.error(f"Failed to get context for book {book_id}: {e}")
            return []

    def health_check(self) -> Dict[str, Any]:
        """Health check for RAG system"""
        return {
            "status": "healthy" if self.is_initialized else "not_initialized",
            "components": {
                "ad_generator": self.ad_generator.is_initialized,
                "image_generator": self.image_generator.is_initialized,
                "recommendation_generator": self.recommendation_generator.is_initialized,
                "database": self.db_manager is not None,
            },
        }


def main():
    """Test the RAG system"""
    rag = RAGSystem()
    rag.initialize()

    # Test ad copy generation
    ad_result = rag.generate_ad_copy("BOOK_000001", "social_media")
    print(f"Ad copy: {ad_result}")

    # Test image prompts
    img_result = rag.generate_image_prompts("BOOK_000001", "modern")
    print(f"Image prompts: {img_result}")

    # Test recommendations
    rec_result = rag.generate_recommendation("BOOK_000001")
    print(f"Recommendation: {rec_result}")


if __name__ == "__main__":
    main()
