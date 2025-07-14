import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(name: str, level: str = "INFO") -> logging.Logger:
    """Setup logging configuration

    Args:
        name: Logger name (usually __name__)
        level: Logging level

    Returns:
        Configured logger
    """
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "soma.log"),
        ],
    )

    return logging.getLogger(name)


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value"""
    return numerator / denominator if denominator != 0 else default
