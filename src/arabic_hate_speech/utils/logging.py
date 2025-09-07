"""
Logging utilities for Arabic Hate Speech Detection.
"""

import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logging(log_file: str = "training.log", log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Name of the log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Full path to log file
    log_path = log_dir / log_file
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("arabic_hate_speech")
    logger.info(f"Logging initialized. Log file: {log_path}")
    
    return logger
