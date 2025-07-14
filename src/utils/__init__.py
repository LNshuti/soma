"""Utilities package for Soma ML Platform."""

from .helpers import ensure_directory, safe_division, setup_logging

# Don't import DatabaseManager at module level to avoid singleton issues
# from .database import DatabaseManager

__all__ = ["setup_logging", "ensure_directory", "safe_division"]
