# src/data/generators.py
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataGenerationConfig:
    """Configuration for data generation"""

    n_publishers: int = 1000
    n_books: int = 10000
    n_sales: int = 100000
    n_inventory: int = 50000
    n_campaigns: int = 5000


class ContentDataGenerator:
    """Generates synthetic content distribution data"""

    def __init__(self, config: DataGenerationConfig):
        self.config = config

    def generate_all(self) -> Dict[str, int]:
        """Generate all synthetic data tables"""
        from src.utils.database import DatabaseManager

        # Initialize database
        db = DatabaseManager()

        results = {}

        # Generate each table in dependency order
        try:
            logger.info("Starting data generation...")

            results["publishers"] = self._generate_publishers(db)
            results["books"] = self._generate_books(db)
            results["sales"] = self._generate_sales(db)
            results["inventory"] = self._generate_inventory(db)
            results["campaign_events"] = self._generate_campaigns(db)

            logger.info("Data generation completed successfully")

        except Exception as e:
            logger.error(f"Data generation failed: {e}")
            raise

        return results

    def _generate_publishers(self, db) -> int:
        """Generate publisher data"""
        try:
            import faker

            fake = faker.Faker()

            publishers = []
            for i in range(self.config.n_publishers):
                publishers.append(
                    {
                        "publisher_id": f"PUB_{i:06d}",
                        "publisher_name": fake.company(),
                        "publisher_type": fake.random_element(
                            ["Traditional", "Independent", "Academic", "Digital"]
                        ),
                        "country": fake.country(),
                        "established_year": fake.random_int(1950, 2020),
                        "total_titles": fake.random_int(1, 500),
                        "active_status": fake.random_element(["Active", "Inactive"]),
                        "created_at": fake.date_time_this_year(),
                    }
                )

            df = pd.DataFrame(publishers)
            db.save_dataframe(df, "publishers", schema="raw")
            logger.info(f"Generated {len(df)} publishers")
            return len(df)

        except Exception as e:
            logger.error(f"Failed to generate publishers: {e}")
            raise

    def _generate_books(self, db) -> int:
        """Generate book catalog data"""
        try:
            import faker

            fake = faker.Faker()

            # Get publisher IDs
            if not db.table_exists("publishers", "raw"):
                raise ValueError("Publishers table not found")

            publishers_df = db.get_table("publishers", "raw")
            if publishers_df.empty:
                raise ValueError("No publishers found")

            publisher_ids = publishers_df["publisher_id"].tolist()

            books = []
            for i in range(self.config.n_books):
                books.append(
                    {
                        "book_id": f"BOOK_{i:06d}",
                        "isbn": fake.isbn13(),
                        "title": fake.catch_phrase(),
                        "author": fake.name(),
                        "publisher_id": fake.random_element(publisher_ids),
                        "genre": fake.random_element(
                            [
                                "Fiction",
                                "Non-Fiction",
                                "Science",
                                "Biography",
                                "Fantasy",
                            ]
                        ),
                        "publication_year": fake.random_int(2000, 2024),
                        "page_count": fake.random_int(50, 800),
                        "price": round(fake.random.uniform(9.99, 99.99), 2),
                        "format": fake.random_element(
                            ["Hardcover", "Paperback", "eBook", "Audiobook"]
                        ),
                        "language": fake.random_element(
                            ["English", "Spanish", "French", "German"]
                        ),
                        "created_at": fake.date_time_this_year(),
                    }
                )

            df = pd.DataFrame(books)
            db.save_dataframe(df, "books", schema="raw")
            logger.info(f"Generated {len(df)} books")
            return len(df)

        except Exception as e:
            logger.error(f"Failed to generate books: {e}")
            raise

    def _generate_sales(self, db) -> int:
        """Generate sales transaction data"""
        try:
            import faker

            fake = faker.Faker()

            # Get book IDs
            if not db.table_exists("books", "raw"):
                raise ValueError("Books table not found")

            books_df = db.get_table("books", "raw")
            if books_df.empty:
                raise ValueError("No books found")

            book_ids = books_df["book_id"].tolist()

            sales = []
            for i in range(self.config.n_sales):
                quantity = fake.random_int(1, 5)
                unit_price = round(fake.random.uniform(9.99, 99.99), 2)
                discount = fake.random.uniform(0, 0.3)
                total_amount = round(quantity * unit_price * (1 - discount), 2)

                sales.append(
                    {
                        "transaction_id": f"TXN_{i:08d}",
                        "book_id": fake.random_element(book_ids),
                        "sale_date": fake.date_between(
                            start_date="-2y", end_date="today"
                        ),
                        "quantity": quantity,
                        "unit_price": unit_price,
                        "discount_percent": round(discount * 100, 2),
                        "total_amount": total_amount,
                        "channel": fake.random_element(
                            ["Online", "Retail", "Wholesale", "Direct"]
                        ),
                        "customer_type": fake.random_element(
                            ["Individual", "Business", "Educational"]
                        ),
                        "region": fake.random_element(
                            ["North", "South", "East", "West", "Central"]
                        ),
                        "created_at": fake.date_time_this_year(),
                    }
                )

            df = pd.DataFrame(sales)
            db.save_dataframe(df, "sales", schema="raw")
            logger.info(f"Generated {len(df)} sales transactions")
            return len(df)

        except Exception as e:
            logger.error(f"Failed to generate sales: {e}")
            raise

    def _generate_inventory(self, db) -> int:
        """Generate inventory data"""
        try:
            import faker

            fake = faker.Faker()

            # Get book IDs
            if not db.table_exists("books", "raw"):
                raise ValueError("Books table not found")

            books_df = db.get_table("books", "raw")
            if books_df.empty:
                raise ValueError("No books found")

            book_ids = books_df["book_id"].tolist()

            inventory = []
            for i in range(self.config.n_inventory):
                stock_qty = fake.random_int(0, 1000)
                reorder_point = fake.random_int(10, 100)

                inventory.append(
                    {
                        "inventory_id": f"INV_{i:08d}",
                        "book_id": fake.random_element(book_ids),
                        "warehouse_location": fake.random_element(
                            ["Warehouse_A", "Warehouse_B", "Warehouse_C"]
                        ),
                        "stock_quantity": stock_qty,
                        "reorder_point": reorder_point,
                        "last_restock_date": fake.date_between(
                            start_date="-6m", end_date="today"
                        ),
                        "storage_cost_per_unit": round(
                            fake.random.uniform(0.50, 5.00), 2
                        ),
                        "shelf_life_days": fake.random_int(365, 1825),
                        "created_at": fake.date_time_this_year(),
                    }
                )

            df = pd.DataFrame(inventory)
            db.save_dataframe(df, "inventory", schema="raw")
            logger.info(f"Generated {len(df)} inventory records")
            return len(df)

        except Exception as e:
            logger.error(f"Failed to generate inventory: {e}")
            raise

    def _generate_campaigns(self, db) -> int:
        """Generate campaign events data"""
        try:
            import faker

            fake = faker.Faker()

            campaigns = []
            for i in range(self.config.n_campaigns):
                impressions = fake.random_int(1000, 100000)
                clicks = fake.random_int(10, impressions // 10)
                spend = round(fake.random.uniform(100, 10000), 2)

                campaigns.append(
                    {
                        "campaign_id": f"CAMP_{i:06d}",
                        "experiment_id": (
                            f"EXP_{fake.random_int(1, 50):03d}"
                            if fake.boolean(chance_of_getting_true=30)
                            else None
                        ),
                        "treatment_group": (
                            fake.random_element(["control", "treatment"])
                            if fake.boolean(chance_of_getting_true=30)
                            else None
                        ),
                        "impressions": impressions,
                        "clicks": clicks,
                        "spend": spend,
                        "campaign_date": fake.date_between(
                            start_date="-1y", end_date="today"
                        ),
                        "created_at": fake.date_time_this_year(),
                    }
                )

            df = pd.DataFrame(campaigns)
            db.save_dataframe(df, "campaign_events", schema="raw")
            logger.info(f"Generated {len(df)} campaign events")
            return len(df)

        except Exception as e:
            logger.error(f"Failed to generate campaigns: {e}")
            raise
