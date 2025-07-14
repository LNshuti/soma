# src/utils/database.py
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import duckdb
import pandas as pd

from src.utils.helpers import setup_logging

logger = setup_logging(__name__)


class DatabaseManager:
    """Thread-safe DuckDB connection manager"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_path: str = "./data/soma.duckdb"):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, db_path: str = "./data/soma.duckdb"):
        if hasattr(self, "_initialized"):
            return

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._initialized = True
        self._setup_database()
        logger.info(f"Database manager initialized: {self.db_path}")

    def _setup_database(self):
        """Setup database schemas"""
        try:
            conn = self.get_connection()

            # Create schemas
            conn.execute("CREATE SCHEMA IF NOT EXISTS raw")
            conn.execute("CREATE SCHEMA IF NOT EXISTS staging")
            conn.execute("CREATE SCHEMA IF NOT EXISTS analytics")

            logger.info("Database schemas created")
        except Exception as e:
            logger.error(f"Database setup failed: {e}")

    def get_connection(self):
        """Get thread-local connection"""
        if not hasattr(self._local, "conn"):
            self._local.conn = duckdb.connect(str(self.db_path))
        return self._local.conn

    @contextmanager
    def get_cursor(self):
        """Get a cursor with automatic cleanup"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    def execute_query(self, query: str, params=None):
        """Execute query and return results"""
        with self.get_cursor() as cursor:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchall()

    def fetch_dataframe(self, query: str, params=None) -> pd.DataFrame:
        """Fetch query results as DataFrame"""
        try:
            conn = self.get_connection()
            if params:
                return conn.execute(query, params).df()
            else:
                return conn.execute(query).df()
        except Exception as e:
            logger.error(f"DataFrame fetch failed: {e}")
            raise

    def fetch_df(self, query: str, params=None) -> pd.DataFrame:
        """Alias for fetch_dataframe"""
        return self.fetch_dataframe(query, params)

    def save_dataframe(self, df: pd.DataFrame, table_name: str, schema: str = "raw"):
        """Save DataFrame to database table"""
        try:
            conn = self.get_connection()
            full_table_name = f"{schema}.{table_name}"

            # Drop table if exists and create new one
            conn.execute(f"DROP TABLE IF EXISTS {full_table_name}")
            conn.execute(f"CREATE TABLE {full_table_name} AS SELECT * FROM df")

            logger.info(f"Saved {len(df)} records to {full_table_name}")

        except Exception as e:
            logger.error(f"Failed to save DataFrame to {schema}.{table_name}: {e}")
            raise

    def get_table(self, table_name: str, schema: str = "raw") -> pd.DataFrame:
        """Get table as DataFrame"""
        try:
            full_table_name = f"{schema}.{table_name}"
            query = f"SELECT * FROM {full_table_name}"
            return self.fetch_dataframe(query)
        except Exception as e:
            logger.error(f"Failed to get table {schema}.{table_name}: {e}")
            return pd.DataFrame()

    def table_exists(self, table_name: str, schema: str = "raw") -> bool:
        """Check if table exists"""
        try:
            query = """
            SELECT COUNT(*) as count
            FROM information_schema.tables 
            WHERE table_schema = ? AND table_name = ?
            """
            result = self.execute_query(query, (schema, table_name))
            return result[0][0] > 0
        except Exception:
            return False

    def close(self):
        """Close thread-local connection"""
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            delattr(self._local, "conn")
