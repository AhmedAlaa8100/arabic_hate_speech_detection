"""
Configuration file for Arabic Hate Speech Detection project.
Contains all hyperparameters, paths, and settings.
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Configuration class for the Arabic Hate Speech Detection project."""
    
    # Model Configuration
    model_name: str = "aubmindlab/bert-base-arabertv02"
    num_labels: int = 2  # Binary classification: hate speech or not
    max_length: int = 128
    
    # Training Configuration
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # Data Configuration
    dataset_name: str = "manueltonneau/arabic-hate-speech-superset"
    train_split: str = "train"
    test_split: str = "test"
    validation_split: float = 0.1
    
    # Paths
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = os.path.join(project_root, "data")
    models_dir: str = os.path.join(project_root, "models")
    results_dir: str = os.path.join(project_root, "results")
    
    # Model saving
    model_save_path: str = os.path.join(models_dir, "arabert_hate_speech_model")
    best_model_path: str = os.path.join(models_dir, "best_arabert_model")
    
    # Logging
    log_file: str = os.path.join(results_dir, "training.log")
    loss_plot_path: str = os.path.join(results_dir, "loss_curve.png")
    metrics_file: str = os.path.join(results_dir, "metrics.json")
    
    # Training settings
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    early_stopping_patience: int = 3
    
    # Device
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # Random seed for reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """Create necessary directories after initialization."""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

# Global config instance
config = Config()
