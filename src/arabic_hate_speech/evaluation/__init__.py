"""
Evaluation components for Arabic Hate Speech Detection.
"""

from .evaluator import Evaluator
from .metrics import calculate_metrics, plot_confusion_matrix, plot_loss_curve

__all__ = ["Evaluator", "calculate_metrics", "plot_confusion_matrix", "plot_loss_curve"]
