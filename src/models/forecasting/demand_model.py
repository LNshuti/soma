# src/models/forecasting/demand_model.py
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.models.base import BaseModel, ModelType
from src.utils.database import DatabaseManager
from src.utils.helpers import setup_logging

logger = setup_logging(__name__)


class DemandForecastingModel(BaseModel):
    """Demand forecasting model using Random Forest"""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the demand forecasting model"""
        super().__init__(
            model_name="demand_forecasting", model_type=ModelType.FORECASTING
        )

        self.feature_columns = []
        self.target_column = "daily_quantity"
        self.model_path = (
            model_path or "./artifacts/models/demand_forecasting_model.pkl"
        )

        # Ensure model directory exists
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load training data from database"""
        try:
            query = """
            SELECT 
                s.book_id,
                s.sale_date,
                s.daily_quantity,
                b.price,
                b.page_count,
                EXTRACT(dow FROM s.sale_date) as day_of_week,
                EXTRACT(month FROM s.sale_date) as month,
                EXTRACT(quarter FROM s.sale_date) as quarter
            FROM main.sales_performance_daily s
            JOIN main.stg_books b ON s.book_id = b.book_id
            WHERE s.daily_quantity IS NOT NULL
            ORDER BY s.book_id, s.sale_date
            """

            df = self.db_manager.fetch_df(query)
            logger.info(f"Loaded {len(df)} records for training")
            return df

        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(
                columns=[
                    "book_id",
                    "sale_date",
                    "daily_quantity",
                    "price",
                    "page_count",
                ]
            )

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features for training/prediction"""
        try:
            # Convert date columns
            if "sale_date" in df.columns:
                df["sale_date"] = pd.to_datetime(df["sale_date"])
                df["day_of_week"] = df["sale_date"].dt.dayofweek
                df["month"] = df["sale_date"].dt.month
                df["quarter"] = df["sale_date"].dt.quarter
                df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

            # Create lag features
            if "daily_quantity" in df.columns:
                df = df.sort_values(["book_id", "sale_date"])
                df["quantity_lag_1"] = df.groupby("book_id")["daily_quantity"].shift(1)
                df["quantity_lag_7"] = df.groupby("book_id")["daily_quantity"].shift(7)
                df["quantity_rolling_7"] = (
                    df.groupby("book_id")["daily_quantity"]
                    .rolling(window=7)
                    .mean()
                    .values
                )

            # Book-level features
            if "price" in df.columns:
                df["price_category"] = pd.cut(
                    df["price"],
                    bins=5,
                    labels=["very_low", "low", "medium", "high", "very_high"],
                )
                df["price_category"] = df["price_category"].cat.codes

            # Select numeric features
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            self.feature_columns = [
                col for col in numeric_columns if col != self.target_column
            ]

            feature_df = df[
                self.feature_columns
                + ([self.target_column] if self.target_column in df.columns else [])
            ]

            if self.target_column in feature_df.columns:
                X = feature_df[self.feature_columns]
                y = feature_df[self.target_column]
                return X, y
            else:
                return feature_df[self.feature_columns], None

        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            raise

    def train(
        self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Train the demand forecasting model"""
        try:
            logger.info("🚀 Starting demand forecasting model training...")

            if X is None or y is None:
                # Load and prepare data
                df = self.load_data()
                if df.empty:
                    raise ValueError("No training data available")

                X, y = self.prepare_features(df)

            # Remove any remaining NaN values
            if X is not None and y is not None:
                df_clean = pd.concat([X, y], axis=1).dropna()
                if df_clean.empty:
                    raise ValueError("No valid training data after cleaning")

                X = df_clean[self.feature_columns]
                y = df_clean[self.target_column]

            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Train model
            self.model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )

            self.model.fit(X_train, y_train)
            self.last_trained = datetime.now()

            # Evaluate
            y_pred = self.model.predict(X_test)
            metrics = {
                "mae": mean_absolute_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            }

            # Save model
            self.save_model()

            logger.info(
                f"✅ Model training completed. MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}"
            )
            return metrics

        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Ensure X has the same features as training
        X_pred = X[self.feature_columns].fillna(0)
        return self.model.predict(X_pred)

    def predict_for_book(self, book_id: str, days_ahead: int = 7) -> Dict:
        """Predict demand for a specific book"""
        try:
            if self.model is None:
                self.load_model()

            # Get recent data for the book
            query = """
            SELECT 
                s.book_id,
                s.sale_date,
                s.daily_quantity,
                b.price,
                b.page_count
            FROM main.sales_performance_daily s
            JOIN main.stg_books b ON s.book_id = b.book_id
            WHERE s.book_id = ?
            ORDER BY s.sale_date DESC
            LIMIT 30
            """

            df = self.db_manager.fetch_df(query, [book_id])

            if df.empty:
                return {
                    "book_id": book_id,
                    "predictions": [
                        {"day": i + 1, "predicted_demand": 5.0}
                        for i in range(days_ahead)
                    ],
                    "method": "default",
                    "generated_at": datetime.now().isoformat(),
                }

            # Prepare features
            X, _ = self.prepare_features(df)

            if X.empty or not all(col in X.columns for col in self.feature_columns):
                return {
                    "book_id": book_id,
                    "predictions": [
                        {"day": i + 1, "predicted_demand": 5.0}
                        for i in range(days_ahead)
                    ],
                    "method": "fallback",
                    "generated_at": datetime.now().isoformat(),
                }

            # Use latest data point as base for prediction
            latest_features = X[self.feature_columns].iloc[-1:].fillna(0)

            # Make predictions
            predictions = []
            for i in range(days_ahead):
                pred = self.predict(latest_features)[0]
                predictions.append(
                    {"day": i + 1, "predicted_demand": max(0, float(pred))}
                )

            return {
                "book_id": book_id,
                "predictions": predictions,
                "method": "ml_model",
                "model_confidence": "medium",
                "generated_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Prediction failed for book {book_id}: {e}")
            return {
                "book_id": book_id,
                "error": str(e),
                "predictions": [
                    {"day": i + 1, "predicted_demand": 5.0} for i in range(days_ahead)
                ],
                "method": "error_fallback",
                "generated_at": datetime.now().isoformat(),
            }

    def save_model(self):
        """Save the trained model"""
        try:
            model_data = {
                "model": self.model,
                "feature_columns": self.feature_columns,
                "target_column": self.target_column,
                "last_trained": self.last_trained,
            }
            joblib.dump(model_data, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def load_model(self):
        """Load a trained model"""
        try:
            if not Path(self.model_path).exists():
                logger.warning(f"Model file not found: {self.model_path}")
                return False

            model_data = joblib.load(self.model_path)
            self.model = model_data["model"]
            self.feature_columns = model_data["feature_columns"]
            self.target_column = model_data.get("target_column", "daily_quantity")
            self.last_trained = model_data.get("last_trained")

            logger.info(f"Model loaded from {self.model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


def main():
    """Train the demand forecasting model"""
    try:
        model = DemandForecastingModel()
        metrics = model.train()
        print(f"✅ Training completed with metrics: {metrics}")
    except Exception as e:
        print(f"❌ Training failed: {e}")


if __name__ == "__main__":
    main()
