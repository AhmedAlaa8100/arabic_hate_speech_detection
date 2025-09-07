#!/usr/bin/env python3
"""
Simple script to test the trained model on the test set.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import Config
from src.evaluate import Evaluator
from src.dataset import DataProcessor

def test_model():
    """Test the trained model on the test set."""
    
    print("🧪 Testing trained model on test set...")
    
    try:
        # Load configuration
        config = Config.from_json()
        
        # Load and prepare data
        data_processor = DataProcessor(config)
        train_dataset, val_dataset, test_dataset = data_processor.load_dataset()
        
        # Create data loaders
        train_loader, val_loader, test_loader = data_processor.create_dataloaders(
            train_dataset, val_dataset, test_dataset
        )
        
        # Initialize evaluator
        evaluator = Evaluator(config)
        
        # Evaluate model
        print("🔍 Testing model...")
        evaluation_results = evaluator.evaluate_and_save_results(test_loader)
        
        print("\n" + "="*50)
        print("📈 TEST RESULTS")
        print("="*50)
        print(f"Accuracy: {evaluation_results['metrics']['accuracy']:.4f} ({evaluation_results['metrics']['accuracy']*100:.2f}%)")
        print(f"Precision: {evaluation_results['metrics']['precision']:.4f}")
        print(f"Recall: {evaluation_results['metrics']['recall']:.4f}")
        print(f"F1-Score: {evaluation_results['metrics']['f1_score']:.4f}")
        print(f"Total samples: {evaluation_results['metrics']['support']}")
        
        print("\n✅ Model testing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing model: {str(e)}")
        return

if __name__ == "__main__":
    test_model()
