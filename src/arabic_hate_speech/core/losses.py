"""
Loss functions for Arabic Hate Speech Detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Reference: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [batch_size, num_classes] (raw logits)
        targets: [batch_size] (class indices)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # probability of the true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss for handling class imbalance.
    """
    def __init__(self, class_weights: Optional[torch.Tensor] = None, reduction="mean"):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [batch_size, num_classes] (raw logits)
        targets: [batch_size] (class indices)
        """
        if self.class_weights is not None:
            # Move weights to the same device as inputs
            weights = self.class_weights.to(inputs.device)
            return F.cross_entropy(inputs, targets, weight=weights, reduction=self.reduction)
        else:
            return F.cross_entropy(inputs, targets, reduction=self.reduction)

def get_loss_function(loss_type: str, class_weights: Optional[torch.Tensor] = None, **kwargs):
    """
    Get the appropriate loss function based on the configuration.
    
    Args:
        loss_type: Type of loss function ('ce', 'weighted', 'focal')
        class_weights: Class weights for weighted loss
        **kwargs: Additional arguments for loss functions
        
    Returns:
        Loss function instance
    """
    if loss_type == "ce":
        return nn.CrossEntropyLoss()
    elif loss_type == "weighted":
        return WeightedCrossEntropyLoss(class_weights=class_weights)
    elif loss_type == "focal":
        alpha = kwargs.get('alpha', 1.0)
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    else:
        raise ValueError(f"Unknown loss function type: {loss_type}")

def compute_class_weights(labels, num_classes):
    """
    Compute class weights for handling class imbalance.
    
    Args:
        labels: List or tensor of class labels
        num_classes: Number of classes
        
    Returns:
        Class weights tensor
    """
    if isinstance(labels, list):
        labels = torch.tensor(labels)
    
    # Count occurrences of each class
    class_counts = torch.bincount(labels, minlength=num_classes).float()
    
    # Avoid division by zero
    class_counts = torch.clamp(class_counts, min=1.0)
    
    # Compute inverse frequency weights
    total_samples = len(labels)
    class_weights = total_samples / (num_classes * class_counts)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * num_classes
    
    logger.info(f"Class weights computed: {class_weights.tolist()}")
    logger.info(f"Class distribution: {class_counts.tolist()}")
    
    return class_weights
