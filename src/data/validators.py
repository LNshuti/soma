# src/data/validators.py
import logging
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data quality and integrity"""

    def __init__(self, db_manager):
        self.db = db_manager

    def validate_all_tables(self) -> Dict[str, Dict[str, Any]]:
        """Validate all tables"""
        results = {}

        tables = ["publishers", "books", "sales", "inventory", "campaign_events"]

        for table in tables:
            try:
                results[table] = self._validate_table(table)
            except Exception as e:
                logger.error(f"Validation failed for {table}: {e}")
                results[table] = {"overall_status": "ERROR", "error": str(e)}

        return results

    def _validate_table(self, table_name: str) -> Dict[str, Any]:
        """Validate a specific table"""
        try:
            # Check if table exists
            if not self.db.table_exists(table_name, "raw"):
                return {"overall_status": "FAIL", "issues": ["Table does not exist"]}

            df = self.db.get_table(table_name, "raw")

            if df is None or df.empty:
                return {"overall_status": "FAIL", "issues": ["Table is empty"]}

            issues = []

            # Basic validations
            null_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns))
            if null_percentage > 0.1:  # More than 10% nulls
                issues.append(f"High null percentage: {null_percentage:.2%}")

            if len(df) == 0:
                issues.append("No records found")

            # Table-specific validations
            if table_name == "sales":
                if "total_amount" in df.columns:
                    invalid_amounts = (df["total_amount"] <= 0).sum()
                    if invalid_amounts > 0:
                        issues.append(f"Invalid sales amounts found: {invalid_amounts}")

                if "quantity" in df.columns:
                    invalid_quantities = (df["quantity"] <= 0).sum()
                    if invalid_quantities > 0:
                        issues.append(f"Invalid quantities found: {invalid_quantities}")

            elif table_name == "books":
                if "price" in df.columns:
                    invalid_prices = (df["price"] <= 0).sum()
                    if invalid_prices > 0:
                        issues.append(f"Invalid prices found: {invalid_prices}")

            elif table_name == "inventory":
                if "stock_quantity" in df.columns:
                    negative_stock = (df["stock_quantity"] < 0).sum()
                    if negative_stock > 0:
                        issues.append(
                            f"Negative stock quantities found: {negative_stock}"
                        )

            status = "PASS" if not issues else "FAIL"

            return {
                "overall_status": status,
                "record_count": len(df),
                "column_count": len(df.columns),
                "null_percentage": f"{null_percentage:.2%}",
                "issues": issues,
            }

        except Exception as e:
            logger.error(f"Validation error for {table_name}: {e}")
            return {"overall_status": "ERROR", "error": str(e)}
