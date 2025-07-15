"""
Recommendation Engine for Soma Content Distribution
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.base import BaseModel, ModelType
from src.utils.database import DatabaseManager
from src.utils.helpers import setup_logging

logger = setup_logging(__name__)


class RecommendationEngine(BaseModel):
    """Content recommendation engine using collaborative filtering and content-based methods"""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """Initialize recommendation engine"""
        super().__init__(
            model_name="recommendation_engine",
            model_type=ModelType.RECOMMENDATION,
            db_manager=db_manager
        )
        self.is_trained = False
        self.user_item_matrix = None
        self.item_features = None
        self.similarity_matrix = None
        self.popular_items = None

    def load_data(self) -> pd.DataFrame:
        """Load training data for recommendations"""
        try:
            # Load sales data as user-item interactions
            sales_query = """
            SELECT 
                s.customer_type as user_id,
                s.book_id,
                SUM(s.quantity) as rating,
                COUNT(*) as interaction_count,
                AVG(s.unit_price) as avg_price,
                MAX(s.sale_date) as last_interaction
            FROM fact_sales s
            WHERE s.sale_date >= CURRENT_DATE - INTERVAL '365 days'
                AND s.quantity > 0
            GROUP BY s.customer_type, s.book_id
            HAVING SUM(s.quantity) > 0
            ORDER BY user_id, book_id
            """

            interactions = self.db_manager.fetch_dataframe(sales_query)
            logger.info(f"Loaded {len(interactions)} user-item interactions")
            return interactions

        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=["user_id", "book_id", "rating", "interaction_count"])

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features for training"""
        try:
            if df.empty:
                return pd.DataFrame(), pd.Series()

            # Load book features
            features_query = """
            SELECT 
                b.book_id,
                b.genre,
                b.price_category,
                b.length_category,
                b.recency_category,
                b.publisher_type,
                b.price,
                b.page_count,
                b.publication_year,
                COALESCE(bf.total_quantity_sold, 0) as total_sales,
                COALESCE(bf.velocity_score, 0) as velocity_score,
                COALESCE(bf.avg_selling_price, b.price) as avg_selling_price
            FROM dim_books b
            LEFT JOIN book_features bf ON b.book_id = bf.book_id
            WHERE b.book_id IN (SELECT DISTINCT book_id FROM fact_sales)
            """

            try:
                features = self.db_manager.fetch_dataframe(features_query)
            except:
                # Fallback query without book_features
                features_query = """
                SELECT 
                    b.book_id,
                    b.genre,
                    b.price_category,
                    b.price,
                    b.page_count,
                    b.publication_year,
                    p.publisher_type
                FROM dim_books b
                LEFT JOIN dim_publishers p ON b.publisher_id = p.publisher_id
                WHERE b.book_id IN (SELECT DISTINCT book_id FROM fact_sales)
                """
                features = self.db_manager.fetch_dataframe(features_query)

            logger.info(f"Loaded {len(features)} item features")
            
            # Return both interactions and features
            return df, features

        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def train(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None) -> Dict:
        """Train the recommendation model"""
        try:
            logger.info("Starting recommendation engine training...")

            # Load data
            interactions = self.load_data()
            if interactions.empty:
                raise ValueError("No training data available")

            interactions, features = self.prepare_features(interactions)
            
            if interactions.empty:
                raise ValueError("No valid training data after preparation")

            # Store data for recommendations
            self.interactions = interactions
            self.item_features = features.set_index("book_id") if not features.empty else pd.DataFrame()

            # Create user-item matrix
            self.user_item_matrix = interactions.pivot_table(
                index="user_id", 
                columns="book_id", 
                values="rating", 
                fill_value=0
            )

            # Compute item similarity if we have features
            if not self.item_features.empty:
                self._compute_item_similarity()

            # Compute popular items
            self._compute_popular_items()

            self.is_trained = True
            self.last_trained = datetime.now()

            # Save model artifacts
            self.save_model()

            metrics = {
                "num_users": len(self.user_item_matrix.index),
                "num_items": len(self.user_item_matrix.columns),
                "num_interactions": len(interactions),
                "sparsity": 1 - (len(interactions) / (len(self.user_item_matrix.index) * len(self.user_item_matrix.columns))),
                "training_time": datetime.now().isoformat(),
            }

            logger.info(f"Training completed: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def _compute_item_similarity(self) -> None:
        """Compute item-item similarity matrix"""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            from sklearn.preprocessing import StandardScaler

            # Prepare numerical features
            numeric_features = self.item_features.select_dtypes(include=[np.number]).fillna(0)
            
            # Encode categorical features
            categorical_columns = ["genre", "price_category", "publisher_type"]
            for col in categorical_columns:
                if col in self.item_features.columns:
                    encoded = pd.get_dummies(self.item_features[col], prefix=col)
                    numeric_features = pd.concat([numeric_features, encoded], axis=1)

            if numeric_features.empty:
                logger.warning("No features available for similarity computation")
                return

            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(numeric_features)

            # Compute similarity matrix
            similarity_scores = cosine_similarity(features_scaled)

            # Convert to DataFrame
            self.similarity_matrix = pd.DataFrame(
                similarity_scores,
                index=numeric_features.index,
                columns=numeric_features.index,
            )

            logger.info("Computed item similarity matrix")

        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            self.similarity_matrix = None

    def _compute_popular_items(self) -> None:
        """Compute popular items for recommendations"""
        try:
            popular_query = """
            SELECT 
                b.book_id,
                b.title,
                b.author,
                b.genre,
                b.price,
                SUM(s.quantity) as total_sales,
                COUNT(DISTINCT s.customer_type) as unique_customers,
                AVG(s.unit_price) as avg_price
            FROM dim_books b
            JOIN fact_sales s ON b.book_id = s.book_id
            WHERE s.sale_date >= CURRENT_DATE - INTERVAL '180 days'
            GROUP BY b.book_id, b.title, b.author, b.genre, b.price
            ORDER BY total_sales DESC, unique_customers DESC
            LIMIT 100
            """
            
            self.popular_items = self.db_manager.fetch_dataframe(popular_query)
            logger.info(f"Computed {len(self.popular_items)} popular items")

        except Exception as e:
            logger.error(f"Failed to compute popular items: {e}")
            self.popular_items = pd.DataFrame()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions - not applicable for recommendation engine"""
        # This method is required by BaseModel but not used for recommendations
        return np.array([])

    def get_similar_items(self, item_id: str, n_items: int = 10) -> List[Tuple[str, float]]:
        """Get similar items to a given item"""
        if not self.is_trained:
            raise ValueError("Model not trained")

        if self.similarity_matrix is None or item_id not in self.similarity_matrix.index:
            logger.warning(f"Item {item_id} not found in similarity matrix")
            return self.get_popular_items(n_items)

        # Get similarity scores for the item
        similarities = self.similarity_matrix.loc[item_id].sort_values(ascending=False)

        # Exclude the item itself and return top n
        similar_items = similarities.iloc[1 : n_items + 1]

        return [(item, score) for item, score in similar_items.items()]

    def get_user_recommendations(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
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

        for item in self.user_item_matrix.columns:
            if item in user_items:
                continue  # Skip items user already has

            score = 0
            weight_sum = 0

            # Use similarity if available
            if self.similarity_matrix is not None and item in self.similarity_matrix.index:
                for user_item in user_items:
                    if user_item in self.similarity_matrix.index:
                        similarity = self.similarity_matrix.loc[item, user_item]
                        rating = user_ratings[user_item]
                        score += similarity * rating
                        weight_sum += abs(similarity)

                if weight_sum > 0:
                    item_scores[item] = score / weight_sum
            else:
                # Fallback to popularity-based scoring
                if not self.popular_items.empty and item in self.popular_items["book_id"].values:
                    popularity = self.popular_items[self.popular_items["book_id"] == item]["total_sales"].iloc[0]
                    item_scores[item] = popularity

        # Sort and return top recommendations
        recommendations = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]

    def get_popular_items(self, n_items: int = 10) -> List[Tuple[str, float]]:
        """Get popular items as fallback recommendations"""
        try:
            if not self.popular_items.empty:
                popular_subset = self.popular_items.head(n_items)
                return [
                    (row["book_id"], row["total_sales"])
                    for _, row in popular_subset.iterrows()
                ]

            # Fallback query
            query = """
            SELECT 
                b.book_id,
                SUM(s.quantity) as popularity_score
            FROM dim_books b
            JOIN fact_sales s ON b.book_id = s.book_id
            WHERE s.sale_date >= CURRENT_DATE - INTERVAL '90 days'
            GROUP BY b.book_id
            ORDER BY popularity_score DESC
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

    def get_similar_books(self, book_id: str, n_recommendations: int = 10) -> Dict:
        """Get similar books based on content similarity."""
        try:
            if not self.is_trained:
                if not self.load_model():
                    # Train if no saved model
                    self.train()

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
            for book_id_sim, similarity_score in similar_items:
                book_info = books_df[books_df["book_id"] == book_id_sim]
                if not book_info.empty:
                    book = book_info.iloc[0]
                    recommendations.append(
                        {
                            "book_id": book_id_sim,
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

    def get_popular_by_user_type(self, user_type: str, n_recommendations: int = 10) -> Dict:
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
                AND s.sale_date >= CURRENT_DATE - INTERVAL '180 days'
            GROUP BY b.book_id, b.title, b.author, b.genre, b.price
            ORDER BY popularity_score DESC
            LIMIT ?
            """

            books_df = self.db_manager.fetch_dataframe(query, [user_type, n_recommendations])

            if books_df.empty:
                # Fallback to general popular books
                popular_items = self.get_popular_items(n_recommendations)
                if not popular_items:
                    return {"error": f"No recommendations found for user type {user_type}"}
                
                # Convert to expected format
                book_ids = [item[0] for item in popular_items]
                placeholders = ",".join(["?" for _ in book_ids])
                fallback_query = f"""
                SELECT book_id, title, author, genre, price
                FROM dim_books 
                WHERE book_id IN ({placeholders})
                """
                books_df = self.db_manager.fetch_dataframe(fallback_query, book_ids)

            recommendations = []
            for _, book in books_df.iterrows():
                recommendations.append(
                    {
                        "book_id": book["book_id"],
                        "title": book["title"],
                        "author": book["author"],
                        "genre": book["genre"],
                        "price": float(book["price"]),
                        "popularity_score": float(book.get("popularity_score", 0)),
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

    def get_content_based_recommendations(self, book_id: str, n_recommendations: int = 10) -> Dict:
        """Alias for get_similar_books for API compatibility."""
        return self.get_similar_books(book_id, n_recommendations)

    def save_model(self) -> None:
        """Save model artifacts"""
        try:
            artifacts_dir = Path("artifacts/models/recommendation_engine")
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            # Save similarity matrix
            if self.similarity_matrix is not None:
                self.similarity_matrix.to_pickle(artifacts_dir / "similarity_matrix.pkl")

            # Save user-item matrix
            if self.user_item_matrix is not None:
                self.user_item_matrix.to_pickle(artifacts_dir / "user_item_matrix.pkl")

            # Save item features
            if self.item_features is not None:
                self.item_features.to_pickle(artifacts_dir / "item_features.pkl")

            # Save popular items
            if self.popular_items is not None:
                self.popular_items.to_pickle(artifacts_dir / "popular_items.pkl")

            # Save metadata
            metadata = {
                "model_name": self.model_name,
                "is_trained": self.is_trained,
                "last_trained": self.last_trained.isoformat() if self.last_trained else None,
                "num_users": len(self.user_item_matrix.index) if self.user_item_matrix is not None else 0,
                "num_items": len(self.similarity_matrix.index) if self.similarity_matrix is not None else 0,
            }

            import json
            with open(artifacts_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Model saved to {artifacts_dir}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_model(self) -> bool:
        """Load model artifacts"""
        try:
            artifacts_dir = Path("artifacts/models/recommendation_engine")

            # Check if metadata exists
            metadata_path = artifacts_dir / "metadata.json"
            if not metadata_path.exists():
                logger.warning("Model metadata not found")
                return False

            # Load metadata
            import json
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Load similarity matrix
            sim_path = artifacts_dir / "similarity_matrix.pkl"
            if sim_path.exists():
                self.similarity_matrix = pd.read_pickle(sim_path)

            # Load user-item matrix
            ui_path = artifacts_dir / "user_item_matrix.pkl"
            if ui_path.exists():
                self.user_item_matrix = pd.read_pickle(ui_path)

            # Load item features
            feat_path = artifacts_dir / "item_features.pkl"
            if feat_path.exists():
                self.item_features = pd.read_pickle(feat_path)

            # Load popular items
            pop_path = artifacts_dir / "popular_items.pkl"
            if pop_path.exists():
                self.popular_items = pd.read_pickle(pop_path)

            self.is_trained = metadata.get("is_trained", False)
            if metadata.get("last_trained"):
                self.last_trained = datetime.fromisoformat(metadata["last_trained"])

            logger.info("Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


def main():
    """Main function for training recommendation engine"""
    try:
        logger.info("Starting recommendation engine training...")

        # Initialize engine
        engine = RecommendationEngine()

        # Train model
        metrics = engine.train()

        logger.info("Recommendation engine training completed!")
        logger.info(f"Metrics: {metrics}")

        return True

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
