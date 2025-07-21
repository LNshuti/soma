from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.models.rag.generators import AdCopyGenerator, ImagePromptGenerator
from src.models.rag.system import RAGSystem


class TestRAGSystem:
    """Test RAG system."""

    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager."""
        mock_db = MagicMock()
        mock_db.fetch_dataframe.return_value = pd.DataFrame(
            {
                "book_id": ["BOOK_001"],
                "title": ["Test Book"],
                "author": ["Test Author"],
                "genre": ["Fiction"],
            }
        )
        return mock_db

    @pytest.fixture
    def rag_system(self, mock_db_manager):
        """Create RAG system."""
        return RAGSystem(mock_db_manager)

    def test_initialization(self, rag_system):
        """Test RAG system initialization."""
        assert rag_system.is_initialized is False
        assert isinstance(rag_system.ad_generator, AdCopyGenerator)
        assert isinstance(rag_system.image_generator, ImagePromptGenerator)

    def test_initialize(self, rag_system):
        """Test RAG system initialization."""
        result = rag_system.initialize()

        assert result is True
        assert rag_system.is_initialized is True

    def test_generate_ad_copy(self, rag_system):
        """Test ad copy generation."""
        rag_system.initialize()

        result = rag_system.generate_ad_copy("BOOK_001", "social_media", "young_adult")

        assert isinstance(result, dict)
        assert "book_id" in result
        assert "ad_type" in result
        assert "ad_copy_variants" in result

    def test_generate_image_prompts(self, rag_system):
        """Test image prompt generation."""
        rag_system.initialize()

        result = rag_system.generate_image_prompts("BOOK_001", "modern")

        assert isinstance(result, dict)
        assert "book_id" in result
        assert "style" in result
        assert "image_prompts" in result

    def test_health_check(self, rag_system):
        """Test health check."""
        health = rag_system.health_check()

        assert isinstance(health, dict)
        assert "status" in health
        assert "components" in health
