"""
Training callbacks for Arabic Hate Speech Detection.
"""

import os
import torch
import logging
from typing import Dict, Any, Optional
from pathlib import Path


class EarlyStopping:
    """
    Early stopping callback to stop training when validation loss stops improving.
    """
    
    def __init__(self, 
                 patience: int = 5, 
                 min_delta: float = 0.0,
                 restore_best_weights: bool = True,
                 monitor: str = "val_loss",
                 mode: str = "min"):
        """
        Initialize early stopping callback.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore best weights when stopping
            monitor: Metric to monitor
            mode: "min" for loss, "max" for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.monitor = monitor
        self.mode = mode
        
        self.wait = 0
        self.best_score = None
        self.best_weights = None
        self.stopped_epoch = 0
        
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, epoch: int, metrics: Dict[str, float], model: torch.nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            epoch: Current epoch number
            metrics: Current metrics dictionary
            model: Current model
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.monitor not in metrics:
            self.logger.warning(f"Metric {self.monitor} not found in metrics")
            return False
        
        current_score = metrics[self.monitor]
        
        if self.best_score is None:
            self.best_score = current_score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif self._is_improvement(current_score, self.best_score):
            self.best_score = current_score
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                self.logger.info(f"Early stopping at epoch {epoch}")
                return True
        
        return False
    
    def _is_improvement(self, current: float, best: float) -> bool:
        """Check if current score is an improvement."""
        if self.mode == "min":
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta


class ModelCheckpoint:
    """
    Model checkpoint callback to save model at regular intervals.
    """
    
    def __init__(self, 
                 filepath: str,
                 monitor: str = "val_loss",
                 mode: str = "min",
                 save_best_only: bool = True,
                 save_freq: int = 1):
        """
        Initialize model checkpoint callback.
        
        Args:
            filepath: Path to save the model
            monitor: Metric to monitor
            mode: "min" for loss, "max" for accuracy
            save_best_only: Whether to save only the best model
            save_freq: Frequency of saving (every N epochs)
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        
        self.best_score = None
        self.logger = logging.getLogger(__name__)
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    def __call__(self, epoch: int, metrics: Dict[str, float], model: torch.nn.Module, 
                 optimizer: torch.optim.Optimizer, **kwargs) -> bool:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            metrics: Current metrics dictionary
            model: Current model
            optimizer: Current optimizer
            **kwargs: Additional arguments
            
        Returns:
            True if model was saved, False otherwise
        """
        if epoch % self.save_freq != 0:
            return False
        
        if self.monitor not in metrics:
            self.logger.warning(f"Metric {self.monitor} not found in metrics")
            return False
        
        current_score = metrics[self.monitor]
        
        if self.best_score is None or self._is_improvement(current_score, self.best_score):
            self.best_score = current_score
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                **kwargs
            }
            
            torch.save(checkpoint, self.filepath)
            self.logger.info(f"Model saved to {self.filepath} (epoch {epoch}, {self.monitor}: {current_score:.4f})")
            return True
        
        return False
    
    def _is_improvement(self, current: float, best: float) -> bool:
        """Check if current score is an improvement."""
        if self.mode == "min":
            return current < best
        else:
            return current > best
