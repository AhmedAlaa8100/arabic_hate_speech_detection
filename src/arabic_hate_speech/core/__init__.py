"""
Core components for Arabic Hate Speech Detection.
"""

from .config import Config
from .model import ArabicHateSpeechClassifier, ModelManager
from .losses import get_loss_function

__all__ = ["Config", "ArabicHateSpeechClassifier", "ModelManager", "get_loss_function"]
