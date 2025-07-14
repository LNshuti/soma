"""Models package for Soma ML Platform."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Safe imports with error handling
try:
    from .base import BaseModel

    logger.info("Successfully imported BaseModel")
except ImportError as e:
    logger.warning(f"Failed to import BaseModel: {e}")
    BaseModel = None

try:
    from .forecasting.demand_model import DemandForecastingModel

    logger.info("Successfully imported DemandForecastingModel")
except ImportError as e:
    logger.warning(f"Failed to import DemandForecastingModel: {e}")
    DemandForecastingModel = None

try:
    from .recommendation.engine import RecommendationEngine

    logger.info("Successfully imported RecommendationEngine")
except ImportError as e:
    logger.warning(f"Some model imports failed: {e}")
    RecommendationEngine = None

try:
    from .rag.system import RAGSystem

    logger.info("Successfully imported RAGSystem")
except ImportError as e:
    logger.warning(f"Failed to import RAGSystem: {e}")
    RAGSystem = None

# Export available models
__all__ = []

if BaseModel:
    __all__.append("BaseModel")
if DemandForecastingModel:
    __all__.append("DemandForecastingModel")
if RecommendationEngine:
    __all__.append("RecommendationEngine")
if RAGSystem:
    __all__.append("RAGSystem")


def get_available_models():
    """Get list of available models"""
    available = {}
    if BaseModel:
        available["BaseModel"] = BaseModel
    if DemandForecastingModel:
        available["DemandForecastingModel"] = DemandForecastingModel
    if RecommendationEngine:
        available["RecommendationEngine"] = RecommendationEngine
    if RAGSystem:
        available["RAGSystem"] = RAGSystem
    return available
