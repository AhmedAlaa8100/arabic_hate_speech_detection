"""
Dataset handling for Arabic Hate Speech Detection.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from datasets import load_dataset
import logging

from .config import Config
from .utils import clean_text, setup_logging

logger = setup_logging("dataset.log")

class ArabicHateSpeechDataset(Dataset):
    """
    Custom Dataset class for Arabic Hate Speech Detection.
    """
    
    def __init__(self, 
                 texts: List[str], 
                 labels: List[int], 
                 tokenizer: AutoTokenizer,
                 max_length: int = 128):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text samples
            labels: List of corresponding labels
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Validate inputs
        assert len(self.texts) == len(self.labels), \
            "Number of texts and labels must be equal"
        
        logger.info(f"Dataset initialized with {len(self.texts)} samples")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing input_ids, attention_mask, and labels
        """
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Clean the text
        text = clean_text(text)
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class DataProcessor:
    """
    Data processor for loading and preparing the Arabic Hate Speech dataset.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the data processor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Tokenizer loaded: {config.model_name}")
    
    def load_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Load and preprocess the dataset.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info(f"Loading dataset: {self.config.dataset_name}")
        
        try:
            # Load dataset from Hugging Face
            dataset = load_dataset(self.config.dataset_name)
            
            # Get train and test splits
            train_data = dataset[self.config.train_split]
            test_data = dataset[self.config.test_split]
            
            logger.info(f"Train samples: {len(train_data)}")
            logger.info(f"Test samples: {len(test_data)}")
            
            # Extract texts and labels
            train_texts = train_data['text']
            train_labels = train_data['label']
            
            test_texts = test_data['text']
            test_labels = test_data['label']
            
            # Create validation split from training data
            val_size = int(len(train_texts) * self.config.validation_split)
            val_texts = train_texts[:val_size]
            val_labels = train_labels[:val_size]
            
            train_texts = train_texts[val_size:]
            train_labels = train_labels[val_size:]
            
            logger.info(f"After split - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
            
            # Create datasets
            train_dataset = ArabicHateSpeechDataset(
                train_texts, train_labels, self.tokenizer, self.config.max_length
            )
            
            val_dataset = ArabicHateSpeechDataset(
                val_texts, val_labels, self.tokenizer, self.config.max_length
            )
            
            test_dataset = ArabicHateSpeechDataset(
                test_texts, test_labels, self.tokenizer, self.config.max_length
            )
            
            return train_dataset, val_dataset, test_dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def create_dataloaders(self, 
                          train_dataset: Dataset, 
                          val_dataset: Dataset, 
                          test_dataset: Dataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create DataLoaders for training, validation, and testing.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        logger.info(f"DataLoaders created - Batch size: {self.config.batch_size}")
        
        return train_loader, val_loader, test_loader
    
    def get_label_distribution(self, labels: List[int]) -> Dict[int, int]:
        """
        Get the distribution of labels in the dataset.
        
        Args:
            labels: List of labels
            
        Returns:
            Dictionary with label counts
        """
        from collections import Counter
        return dict(Counter(labels))
    
    def print_dataset_info(self, 
                          train_dataset: Dataset, 
                          val_dataset: Dataset, 
                          test_dataset: Dataset) -> None:
        """
        Print information about the datasets.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
        """
        logger.info("=" * 50)
        logger.info("DATASET INFORMATION")
        logger.info("=" * 50)
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")
        logger.info(f"Total samples: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
        logger.info(f"Max sequence length: {self.config.max_length}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info("=" * 50)
