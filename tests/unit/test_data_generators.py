from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.generators import ContentDataGenerator, DataGenerationConfig


class TestContentDataGenerator:
    """Test data generation."""

    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager."""
        mock_db = MagicMock()
        mock_db.table_exists.return_value = True
        mock_db.get_table.return_value = pd.DataFrame(
            {
                "publisher_id": ["PUB_001", "PUB_002"],
                "book_id": ["BOOK_001", "BOOK_002"],
            }
        )
        return mock_db

    @pytest.fixture
    def generator_config(self):
        """Create test configuration."""
        return DataGenerationConfig(
            n_publishers=10, n_books=20, n_sales=50, n_inventory=30, n_campaigns=15
        )

    @pytest.fixture
    def data_generator(self, generator_config):
        """Create data generator."""
        return ContentDataGenerator(generator_config)

    @patch("faker.Faker")
    def test_generate_publishers(self, mock_faker, data_generator, mock_db_manager):
        """Test publisher data generation."""
        # Setup faker mock
        fake_instance = MagicMock()
        mock_faker.return_value = fake_instance
        fake_instance.company.return_value = "Test Publisher"
        fake_instance.random_element.return_value = "Traditional"
        fake_instance.country.return_value = "USA"
        fake_instance.random_int.return_value = 2000
        fake_instance.date_time_this_year.return_value = "2023-01-01"

        result = data_generator._generate_publishers(mock_db_manager)

        assert result == 10  # Based on config
        mock_db_manager.save_dataframe.assert_called_once()

    @patch("faker.Faker")
    def test_generate_books(self, mock_faker, data_generator, mock_db_manager):
        """Test book data generation."""
        # Setup faker mock
        fake_instance = MagicMock()
        mock_faker.return_value = fake_instance
        fake_instance.isbn13.return_value = "1234567890123"
        fake_instance.catch_phrase.return_value = "Test Book Title"
        fake_instance.name.return_value = "Test Author"
        fake_instance.random_element.return_value = "Fiction"
        fake_instance.random_int.return_value = 200
        fake_instance.random.uniform.return_value = 19.99
        fake_instance.date_time_this_year.return_value = "2023-01-01"

        result = data_generator._generate_books(mock_db_manager)

        assert result == 20  # Based on config
        mock_db_manager.save_dataframe.assert_called_once()
