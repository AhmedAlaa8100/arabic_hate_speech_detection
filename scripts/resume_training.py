#!/usr/bin/env python3
"""
Resume training from the last saved checkpoint.
This script helps you continue training from where it left off.
"""

import sys
import os
import torch
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import Config
from src.dataset import DataProcessor
from src.model import ModelManager
from src.train import Trainer
from src.utils import setup_logging

def resume_training():
    """Resume training from the last checkpoint."""
    
    print("🔄 Resuming training from last checkpoint...")
    
    # Load configuration
    config = Config.from_json()
    logger = setup_logging(config.log_file)
    
    try:
        # Load and prepare data
        print("📊 Loading dataset...")
        data_processor = DataProcessor(config)
        train_dataset, val_dataset, test_dataset = data_processor.load_dataset()
        
        # Create data loaders
        train_loader, val_loader, test_loader = data_processor.create_dataloaders(
            train_dataset, val_dataset, test_dataset
        )
        
        # Initialize model manager
        model_manager = ModelManager(config)
        
        # Create model
        print("🤖 Creating model...")
        model = model_manager.create_model()
        
        # Try to load the best model first
        try:
            print("📥 Loading best model...")
            loaded_info = model_manager.load_best_model(model)
            start_epoch = loaded_info['epoch'] + 1
            print(f"✅ Loaded best model from epoch {loaded_info['epoch']} with accuracy {loaded_info['accuracy']:.4f}")
        except:
            # If best model fails, try epoch 2
            try:
                print("📥 Loading model from epoch 2...")
                loaded_info = model_manager.load_model(model, "models/arabert_hate_speech_model_epoch_2.pt")
                start_epoch = loaded_info['epoch'] + 1
                print(f"✅ Loaded model from epoch {loaded_info['epoch']} with accuracy {loaded_info['accuracy']:.4f}")
            except:
                # If all fails, start from scratch
                print("⚠️  Could not load checkpoint, starting from scratch...")
                start_epoch = 1
        
        # Create trainer
        trainer = Trainer(config, model_manager)
        
        # Resume training
        print(f"🚀 Resuming training from epoch {start_epoch}...")
        training_results = trainer.train(
            train_loader, val_loader, test_loader,
            start_epoch=start_epoch
        )
        
        print("✅ Training completed successfully!")
        print(f"📊 Final Results:")
        print(f"   - Best Accuracy: {training_results['best_accuracy']:.4f}")
        print(f"   - Final Loss: {training_results['final_loss']:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        print(f"❌ Training failed: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume Arabic Hate Speech Detection Training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of additional epochs to train")
    
    args = parser.parse_args()
    
    # Update config for additional epochs
    config = Config.from_json()
    config.num_epochs = args.epochs
    
    success = resume_training()
    if success:
        print("🎉 Training resumed and completed successfully!")
    else:
        print("💥 Training failed. Check the logs for details.")
