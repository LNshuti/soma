"""
Recommendation Engine for Soma Content Distribution
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.utils.database import DatabaseManager
from src.utils.helpers import setup_logging

logger = setup_logging(__name__)


class RecommendationEngine(BaseModel):
    """Content recommendation engine using collaborative filtering and content-based methods"""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """Initialize recommendation engine"""
        super().__init__()
        self.db_manager = db_manager or DatabaseManager()
        self.model_name = "recommendation_engine"
        self.is_trained = False
        self.user_item_matrix = None
        self.item_features = None
        self.similarity_matrix = None

    def load_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training data for recommendations"""
        try:
            # Load sales data as user-item interactions
            sales_query = """
            SELECT 
                customer_type as user_id,
                book_id,
                SUM(quantity) as rating,
                COUNT(*) as interaction_count
            FROM fact_sales 
            WHERE sale_date >= CURRENT_DATE - INTERVAL '365 days'
            GROUP BY customer_type, book_id
            HAVING SUM(quantity) > 0
            """

            # Load book features
            features_query = """
            SELECT 
                book_id,
                genre,
                price_category,
                length_category,
                recency_category,
                publisher_type,
                avg_selling_price,
                total_quantity_sold,
                velocity_score
            FROM book_features
            WHERE total_transactions > 0
            """

            interactions = self.db_manager.fetch_dataframe(sales_query)
            features = self.db_manager.fetch_dataframe(features_query)

            logger.info(
                f"Loaded {len(interactions)} interactions and {len(features)} item features"
            )
            return interactions, features

        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            raise

    def prepare_data(self, interactions: pd.DataFrame, features: pd.DataFrame) -> None:
        """Prepare data for training"""
        try:
            # Create user-item matrix
            self.user_item_matrix = interactions.pivot_table(
                index="user_id", columns="book_id", values="rating", fill_value=0
            )

            # Prepare item features
            self.item_features = features.set_index("book_id")

            # One-hot encode categorical features
            categorical_cols = [
                "genre",
                "price_category",
                "length_category",
                "recency_category",
                "publisher_type",
            ]

            for col in categorical_cols:
                if col in self.item_features.columns:
                    dummies = pd.get_dummies(self.item_features[col], prefix=col)
                    self.item_features = pd.concat(
                        [self.item_features, dummies], axis=1
                    )
                    self.item_features.drop(col, axis=1, inplace=True)

            logger.info(f"Prepared user-item matrix: {self.user_item_matrix.shape}")
            logger.info(f"Prepared item features: {self.item_features.shape}")

        except Exception as e:
            logger.error(f"Failed to prepare data: {e}")
            raise

    def compute_item_similarity(self) -> None:
        """Compute item-item similarity matrix"""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            from sklearn.preprocessing import StandardScaler

            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(self.item_features.fillna(0))

            # Compute similarity matrix
            self.similarity_matrix = cosine_similarity(features_scaled)

            # Convert to DataFrame for easier indexing
            self.similarity_matrix = pd.DataFrame(
                self.similarity_matrix,
                index=self.item_features.index,
                columns=self.item_features.index,
            )

            logger.info("Computed item similarity matrix")

        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            raise

    def train(self) -> Dict:
        """Train the recommendation model"""
        try:
            logger.info("🚀 Starting recommendation engine training...")

            # Load data
            interactions, features = self.load_training_data()

            if interactions.empty or features.empty:
                raise ValueError("No training data available")

            # Prepare data
            self.prepare_data(interactions, features)

            # Compute similarities
            self.compute_item_similarity()

            self.is_trained = True
            self.last_trained = datetime.now()

            # Save model artifacts
            self.save_model()

            metrics = {
                "num_users": len(self.user_item_matrix.index),
                "num_items": len(self.user_item_matrix.columns),
                "num_interactions": interactions.shape[0],
                "sparsity": 1
                - (
                    interactions.shape[0]
                    / (
                        len(self.user_item_matrix.index)
                        * len(self.user_item_matrix.columns)
                    )
                ),
                "training_time": datetime.now(),
            }

            logger.info(f"✅ Training completed: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise

    def get_similar_items(
        self, item_id: str, n_items: int = 10
    ) -> List[Tuple[str, float]]:
        """Get similar items to a given item"""
        if not self.is_trained or self.similarity_matrix is None:
            raise ValueError("Model not trained")

        if item_id not in self.similarity_matrix.index:
            logger.warning(f"Item {item_id} not found in similarity matrix")
            return []

        # Get similarity scores for the item
        similarities = self.similarity_matrix.loc[item_id].sort_values(ascending=False)

        # Exclude the item itself and return top n
        similar_items = similarities.iloc[1 : n_items + 1]

        return [(item, score) for item, score in similar_items.items()]

    def get_user_recommendations(
        self, user_id: str, n_recommendations: int = 10
    ) -> List[Tuple[str, float]]:
        """Get recommendations for a user"""
        if not self.is_trained:
            raise ValueError("Model not trained")

        if user_id not in self.user_item_matrix.index:
            logger.warning(f"User {user_id} not found, using popular items")
            return self.get_popular_items(n_recommendations)

        # Get user's ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        user_items = user_ratings[user_ratings > 0].index.tolist()

        if not user_items:
            return self.get_popular_items(n_recommendations)

        # Calculate scores for all items
        item_scores = {}

        for item in self.similarity_matrix.index:
            if item in user_items:
                continue  # Skip items user already has

            score = 0
            weight_sum = 0

            for user_item in user_items:
                if user_item in self.similarity_matrix.index:
                    similarity = self.similarity_matrix.loc[item, user_item]
                    rating = user_ratings[user_item]
                    score += similarity * rating
                    weight_sum += abs(similarity)

            if weight_sum > 0:
                item_scores[item] = score / weight_sum

        # Sort and return top recommendations
        recommendations = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]

    def get_popular_items(self, n_items: int = 10) -> List[Tuple[str, float]]:
        """Get popular items as fallback recommendations"""
        try:
            query = """
            SELECT 
                book_id,
                total_quantity_sold as popularity_score
            FROM book_features 
            WHERE total_transactions > 10
            ORDER BY total_quantity_sold DESC
            LIMIT ?
            """

            popular_items = self.db_manager.fetch_dataframe(query, params=(n_items,))

            if popular_items.empty:
                return []

            return [
                (row["book_id"], row["popularity_score"])
                for _, row in popular_items.iterrows()
            ]

        except Exception as e:
            logger.error(f"Failed to get popular items: {e}")
            return []

    def predict(
        self, user_id: str, item_id: Optional[str] = None, n_recommendations: int = 10
    ) -> Dict:
        """Make predictions"""
        try:
            if item_id:
                # Get similar items
                similar_items = self.get_similar_items(item_id, n_recommendations)
                return {
                    "type": "similar_items",
                    "item_id": item_id,
                    "recommendations": similar_items,
                }
            else:
                # Get user recommendations
                recommendations = self.get_user_recommendations(
                    user_id, n_recommendations
                )
                return {
                    "type": "user_recommendations",
                    "user_id": user_id,
                    "recommendations": recommendations,
                }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Fallback to popular items
            popular = self.get_popular_items(n_recommendations)
            return {
                "type": "fallback_popular",
                "recommendations": popular,
                "error": str(e),
            }

    def save_model(self) -> None:
        """Save model artifacts"""
        try:
            artifacts_dir = Path("artifacts/models")
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            # Save similarity matrix
            if self.similarity_matrix is not None:
                self.similarity_matrix.to_pickle(
                    artifacts_dir / "similarity_matrix.pkl"
                )

            # Save user-item matrix
            if self.user_item_matrix is not None:
                self.user_item_matrix.to_pickle(artifacts_dir / "user_item_matrix.pkl")

            # Save item features
            if self.item_features is not None:
                self.item_features.to_pickle(artifacts_dir / "item_features.pkl")

            # Save metadata
            metadata = {
                "model_name": self.model_name,
                "is_trained": self.is_trained,
                "last_trained": (
                    self.last_trained.isoformat() if self.last_trained else None
                ),
                "num_users": (
                    len(self.user_item_matrix.index)
                    if self.user_item_matrix is not None
                    else 0
                ),
                "num_items": (
                    len(self.similarity_matrix.index)
                    if self.similarity_matrix is not None
                    else 0
                ),
            }

            import json

            with open(artifacts_dir / "recommendation_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Model saved to {artifacts_dir}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_model(self) -> bool:
        """Load model artifacts"""
        try:
            artifacts_dir = Path("artifacts/models")

            # Check if files exist
            required_files = ["similarity_matrix.pkl", "recommendation_metadata.json"]
            if not all((artifacts_dir / f).exists() for f in required_files):
                logger.warning("Model artifacts not found")
                return False

            # Load similarity matrix
            self.similarity_matrix = pd.read_pickle(
                artifacts_dir / "similarity_matrix.pkl"
            )

            # Load user-item matrix if exists
            if (artifacts_dir / "user_item_matrix.pkl").exists():
                self.user_item_matrix = pd.read_pickle(
                    artifacts_dir / "user_item_matrix.pkl"
                )

            # Load item features if exists
            if (artifacts_dir / "item_features.pkl").exists():
                self.item_features = pd.read_pickle(artifacts_dir / "item_features.pkl")

            # Load metadata
            import json

            with open(artifacts_dir / "recommendation_metadata.json", "r") as f:
                metadata = json.load(f)

            self.is_trained = metadata.get("is_trained", False)
            if metadata.get("last_trained"):
                self.last_trained = datetime.fromisoformat(metadata["last_trained"])

            logger.info("Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def get_similar_books(self, book_id: str, n_recommendations: int = 10) -> Dict:
        """Get similar books based on content similarity."""
        try:
            similar_items = self.get_similar_items(book_id, n_recommendations)

            if not similar_items:
                return {
                    "error": f"No similar books found for {book_id}",
                    "book_id": book_id,
                }

            # Get book details for recommendations
            book_ids = [item[0] for item in similar_items]
            placeholders = ",".join(["?" for _ in book_ids])
            query = f"""
            SELECT book_id, title, author, genre, price 
            FROM dim_books 
            WHERE book_id IN ({placeholders})
            """

            books_df = self.db_manager.fetch_dataframe(query, book_ids)

            recommendations = []
            for book_id, similarity_score in similar_items:
                book_info = books_df[books_df["book_id"] == book_id]
                if not book_info.empty:
                    book = book_info.iloc[0]
                    recommendations.append(
                        {
                            "book_id": book_id,
                            "title": book["title"],
                            "author": book["author"],
                            "genre": book["genre"],
                            "price": float(book["price"]),
                            "similarity_score": float(similarity_score),
                            "reason": "Similar content and features",
                        }
                    )

            return {
                "type": "content_based",
                "source_book_id": book_id,
                "recommendations": recommendations,
                "total_found": len(recommendations),
                "method": "similarity_matrix",
                "generated_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Similar books recommendation failed: {e}")
            return {"error": str(e), "book_id": book_id}

    def get_popular_by_user_type(
        self, user_type: str, n_recommendations: int = 10
    ) -> Dict:
        """Get popular books for specific user type."""
        try:
            query = """
            SELECT 
                b.book_id,
                b.title,
                b.author,
                b.genre,
                b.price,
                SUM(s.quantity) as popularity_score
            FROM dim_books b
            JOIN fact_sales s ON b.book_id = s.book_id
            WHERE s.customer_type = ?
            GROUP BY b.book_id, b.title, b.author, b.genre, b.price
            ORDER BY popularity_score DESC
            LIMIT ?
            """

            books_df = self.db_manager.fetch_dataframe(
                query, [user_type, n_recommendations]
            )

            if books_df.empty:
                # Fallback to general popular books
                return self.get_popular_items(n_recommendations)

            recommendations = []
            for _, book in books_df.iterrows():
                recommendations.append(
                    {
                        "book_id": book["book_id"],
                        "title": book["title"],
                        "author": book["author"],
                        "genre": book["genre"],
                        "price": float(book["price"]),
                        "popularity_score": float(book["popularity_score"]),
                        "reason": f"Popular among {user_type} customers",
                    }
                )

            return {
                "type": "popular_by_user_type",
                "user_type": user_type,
                "recommendations": recommendations,
                "total_found": len(recommendations),
                "method": "sales_popularity",
                "generated_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Popular by user type recommendation failed: {e}")
            return {"error": str(e), "user_type": user_type}

    def get_content_based_recommendations(
        self, book_id: str, n_recommendations: int = 10
    ) -> Dict:
        """Alias for get_similar_books for API compatibility."""
        return self.get_similar_books(book_id, n_recommendations)


def main():
    """Main function for training recommendation engine"""
    try:
        logger.info("🚀 Starting recommendation engine training...")

        # Initialize engine
        engine = RecommendationEngine()

        # Train model
        metrics = engine.train()

        logger.info("✅ Recommendation engine training completed!")
        logger.info(f"📊 Metrics: {metrics}")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return False

    return True


if __name__ == "__main__":
    main()
