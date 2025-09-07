"""
Arabic Hate Speech Detection Package

A complete deep learning project for detecting hate speech in Arabic text using AraBERT.
"""

__version__ = "1.0.0"
__author__ = "Arabic Hate Speech Detection Team"

# Core imports
from .core.config import Config
from .core.model import ArabicHateSpeechClassifier, ModelManager

# Data imports
from .data.dataset import DataProcessor, ArabicHateSpeechDataset

# Training imports
from .training.trainer import Trainer

# Evaluation imports
from .evaluation.evaluator import Evaluator

# Utils imports
from .utils.logging import setup_logging
from .utils.helpers import clean_text, get_device

__all__ = [
    # Core
    "Config",
    "ArabicHateSpeechClassifier", 
    "ModelManager",
    
    # Data
    "DataProcessor",
    "ArabicHateSpeechDataset",
    
    # Training
    "Trainer",
    
    # Evaluation
    "Evaluator",
    
    # Utils
    "setup_logging",
    "clean_text",
    "get_device"
]
