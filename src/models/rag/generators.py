# src/models/rag/generators.py
"""RAG content generators for marketing copy and recommendations."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.database import DatabaseManager
from src.utils.helpers import setup_logging

logger = setup_logging(__name__)


class ContentGenerator:
    """Base content generator for RAG system"""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager or DatabaseManager()
        self.is_initialized = False

    def generate(self, prompt: str, context: List[str]) -> str:
        """Generate content based on prompt and context"""
        # Template-based generation for fallback
        if not context:
            return f"Generated response for: {prompt[:50]}..."

        context_str = " ".join(context[:3])  # Use first 3 context items
        return f"Based on {context_str}, here's a response to: {prompt[:50]}..."

    def initialize(self) -> bool:
        """Initialize the generator"""
        self.is_initialized = True
        logger.info("Content generator initialized")
        return True


class AdCopyGenerator(ContentGenerator):
    """Generator for advertising copy"""

    def generate_ad_copy(
        self,
        book_id: str,
        ad_type: str = "social_media",
        target_audience: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate ad copy for a book"""
        try:
            # Get book information
            book_info = self._get_book_info(book_id)
            if not book_info:
                return {"error": f"Book {book_id} not found"}

            # Get similar books for context
            similar_books = self._get_similar_books(book_id)

            # Generate ad copy variants
            variants = self._generate_variants(book_info, ad_type, target_audience)

            return {
                "book_id": book_id,
                "ad_type": ad_type,
                "target_audience": target_audience or "general",
                "ad_copy_variants": variants,
                "similar_books_context": [
                    b.get("title", "") for b in similar_books[:3]
                ],
                "generated_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Ad copy generation failed: {e}")
            return {"error": str(e)}

    def _get_book_info(self, book_id: str) -> Optional[Dict[str, Any]]:
        """Get book information from database"""
        try:
            query = "SELECT * FROM dim_books WHERE book_id = ?"
            result = self.db_manager.fetch_dataframe(query, [book_id])
            return result.iloc[0].to_dict() if not result.empty else None
        except Exception as e:
            logger.error(f"Failed to get book info: {e}")
            return None

    def _get_similar_books(self, book_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get similar books for context"""
        try:
            query = """
            SELECT b1.title, b1.author, b1.genre 
            FROM dim_books b1 
            JOIN dim_books b2 ON b1.genre = b2.genre 
            WHERE b2.book_id = ? AND b1.book_id != ?
            LIMIT ?
            """
            result = self.db_manager.fetch_dataframe(query, [book_id, book_id, limit])
            return result.to_dict("records")
        except Exception as e:
            logger.error(f"Failed to get similar books: {e}")
            return []

    def _generate_variants(
        self, book_info: Dict[str, Any], ad_type: str, target_audience: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Generate ad copy variants"""
        title = book_info.get("title", "Unknown Title")
        author = book_info.get("author", "Unknown Author")
        genre = book_info.get("genre", "Unknown Genre")
        price = book_info.get("price", 0)

        templates = {
            "social_media": [
                f"📚 Discover {title} by {author} - A captivating {genre.lower()} story! ✨ #BookLovers #Reading",
                f"🔥 New read alert! {title} will take you on an unforgettable journey. Get your copy today! 📖",
                f"⭐ Why readers love {title}: Engaging plot, brilliant writing, {genre} at its best! 🎯",
            ],
            "email": [
                f"Subject: Don't Miss Out - {title} Now Available\n\nDear Reader,\n\nExperience the compelling world of {title} by {author}. This {genre.lower()} masterpiece is priced at just ${price:.2f}.",
                f"Subject: Your Next Great Read Awaits\n\n{title} by {author} has arrived! Join thousands of readers discovering this {genre.lower()} gem.",
            ],
            "display": [
                f"{title} - The {genre} Novel Everyone's Talking About",
                f"From acclaimed author {author} comes {title} - Available Now",
            ],
        }

        selected_templates = templates.get(ad_type, templates["social_media"])

        variants = []
        for i, template in enumerate(selected_templates):
            variants.append(
                {
                    "text": template,
                    "type": "template",
                    "confidence": 0.8,
                    "length": len(template),
                    "call_to_action": (
                        "Order now!" if ad_type == "social_media" else "Learn more"
                    ),
                }
            )

        return variants


class ImagePromptGenerator(ContentGenerator):
    """Generator for AI image prompts"""

    def generate_image_prompts(
        self, book_id: str, style: str = "modern"
    ) -> Dict[str, Any]:
        """Generate image prompts for a book"""
        try:
            # Get book information
            book_info = self._get_book_info(book_id)
            if not book_info:
                return {"error": f"Book {book_id} not found"}

            # Generate prompts
            prompts = self._generate_prompts(book_info, style)

            return {
                "book_id": book_id,
                "style": style,
                "image_prompts": prompts,
                "color_palette": self._suggest_colors(book_info),
                "recommended_dimensions": {
                    "cover": "1600x2400",
                    "social": "1080x1080",
                    "banner": "1200x628",
                },
                "generated_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Image prompt generation failed: {e}")
            return {"error": str(e)}

    def _get_book_info(self, book_id: str) -> Optional[Dict[str, Any]]:
        """Get book information from database"""
        try:
            query = "SELECT * FROM dim_books WHERE book_id = ?"
            result = self.db_manager.fetch_dataframe(query, [book_id])
            return result.iloc[0].to_dict() if not result.empty else None
        except Exception as e:
            logger.error(f"Failed to get book info: {e}")
            return None

    def _generate_prompts(
        self, book_info: Dict[str, Any], style: str
    ) -> List[Dict[str, Any]]:
        """Generate image prompts"""
        title = book_info.get("title", "Unknown Title")
        author = book_info.get("author", "Unknown Author")
        genre = book_info.get("genre", "Unknown Genre")

        base_prompts = {
            "modern": [
                f"Modern book cover design for '{title}' by {author} - clean, minimalist {genre.lower()} style",
                f"Contemporary book cover featuring themes from '{title}' - sleek typography and modern aesthetics",
                f"Minimalist cover design for {genre} book '{title}' - professional and eye-catching",
            ],
            "vintage": [
                f"Vintage-style book cover for '{title}' - classic {genre.lower()} design with retro typography",
                f"Retro book cover design for '{title}' by {author} - nostalgic and timeless",
                f"Classic vintage cover for {genre} novel '{title}' - elegant and traditional",
            ],
            "dramatic": [
                f"Dramatic book cover for '{title}' - bold {genre.lower()} imagery with striking visuals",
                f"High-impact cover design for '{title}' by {author} - compelling and intense",
                f"Bold dramatic cover for {genre} book '{title}' - attention-grabbing design",
            ],
        }

        selected_prompts = base_prompts.get(style, base_prompts["modern"])

        prompts = []
        for i, prompt_text in enumerate(selected_prompts):
            prompts.append(
                {
                    "prompt": prompt_text,
                    "focus": ["cover", "typography", "theme"][i % 3],
                    "style": style,
                }
            )

        return prompts

    def _suggest_colors(self, book_info: Dict[str, Any]) -> List[str]:
        """Suggest color palette based on genre"""
        genre = book_info.get("genre", "").lower()

        color_mapping = {
            "fiction": ["navy", "gold", "cream"],
            "fantasy": ["purple", "silver", "dark blue"],
            "science": ["blue", "white", "gray"],
            "biography": ["brown", "beige", "dark green"],
            "romance": ["pink", "gold", "white"],
            "mystery": ["black", "red", "gray"],
            "thriller": ["red", "black", "white"],
        }

        return color_mapping.get(genre, ["blue", "white", "gray"])


class BookRecommendationGenerator(ContentGenerator):
    """Generator for book recommendations"""

    def generate_recommendation_text(
        self, book_info: Dict[str, Any], user_context: Dict[str, Any]
    ) -> str:
        """Generate book recommendation text"""
        title = book_info.get("title", "Unknown Title")
        author = book_info.get("author", "Unknown Author")
        genre = book_info.get("genre", "Unknown Genre")

        user_type = user_context.get("user_type", "general reader")

        if user_type == "academic":
            return f"For academic readers: {title} by {author} offers deep insights into {genre.lower()} themes with scholarly rigor."
        elif user_type == "casual":
            return f"Perfect for your next read: {title} is an engaging {genre.lower()} book that's hard to put down!"
        else:
            return f"Highly recommended: {title} by {author} - a compelling {genre.lower()} work that readers are loving."


def main():
    """Test the generators"""
    generator = AdCopyGenerator()
    generator.initialize()

    # Test ad copy generation
    result = generator.generate_ad_copy("BOOK_000001", "social_media")
    print(f"Ad copy result: {result}")

    # Test image prompts
    img_generator = ImagePromptGenerator()
    img_result = img_generator.generate_image_prompts("BOOK_000001", "modern")
    print(f"Image prompts result: {img_result}")


if __name__ == "__main__":
    main()
