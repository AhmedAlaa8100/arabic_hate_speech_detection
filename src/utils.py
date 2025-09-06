"""
Utility functions for Arabic text preprocessing and general utilities.
"""

import re
import logging
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def setup_logging(log_file: str, level: int = logging.INFO) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalize_arabic_text(text: str) -> str:
    """
    Normalize Arabic text for better model performance.
    
    Args:
        text: Input Arabic text
        
    Returns:
        Normalized Arabic text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Normalize Arabic diacritics (optional - can be removed if needed)
    # text = re.sub(r'[\u064B-\u0652\u0670\u0640]', '', text)
    
    # Normalize Arabic letters
    text = re.sub(r'[أإآا]', 'ا', text)  # Normalize alef variations
    text = re.sub(r'[يى]', 'ي', text)    # Normalize yeh variations
    text = re.sub(r'[ة]', 'ه', text)     # Normalize teh marbuta
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove mentions and hashtags (optional)
    # text = re.sub(r'@\w+', '', text)
    # text = re.sub(r'#\w+', '', text)
    
    # Remove extra punctuation
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def clean_text(text: str) -> str:
    """
    Clean and preprocess text for training.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Basic cleaning
    text = text.strip()
    text = normalize_arabic_text(text)
    
    # Remove empty strings
    if not text or len(text.strip()) == 0:
        return ""
    
    return text

def save_metrics(metrics: Dict[str, Any], filepath: str) -> None:
    """Save metrics to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

def load_metrics(filepath: str) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def plot_training_curves(train_losses: List[float], 
                        eval_losses: List[float], 
                        eval_accuracies: List[float],
                        save_path: str) -> None:
    """
    Plot training curves and save to file.
    
    Args:
        train_losses: List of training losses
        eval_losses: List of evaluation losses
        eval_accuracies: List of evaluation accuracies
        save_path: Path to save the plot
    """
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Training Loss', color='blue', alpha=0.7)
    ax1.plot(eval_losses, label='Validation Loss', color='red', alpha=0.7)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(eval_accuracies, label='Validation Accuracy', color='green', alpha=0.7)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true: List[int], 
                         y_pred: List[int], 
                         labels: List[str],
                         save_path: str) -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def print_classification_report(y_true: List[int], 
                               y_pred: List[int], 
                               labels: List[str]) -> str:
    """
    Print and return classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        
    Returns:
        Classification report as string
    """
    report = classification_report(
        y_true, y_pred, 
        target_names=labels, 
        digits=4
    )
    print(report)
    return report

def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.2f}m"
    else:
        return f"{seconds/3600:.2f}h"
