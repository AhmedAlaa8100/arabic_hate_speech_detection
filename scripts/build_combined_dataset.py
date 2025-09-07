"""
Script to build a combined Arabic hate speech dataset from multiple sources.

This script:
1. Loads two Arabic hate speech datasets from Hugging Face
2. Cleans texts by removing usernames (@USER), URLs, and other noise
3. Normalizes labels into binary classification (hate/offensive vs non-hate)
4. Balances the superset by taking all hate samples and equal random non-hate samples
5. Merges with the Egyptian dataset
6. Splits into train/val/test sets (70/15/15) with stratification
7. Saves as CSV files in data/combined/

Memory Optimization:
- Uses pandas DataFrames instead of numpy arrays for text handling
- Avoids numpy._core._exceptions._ArrayMemoryError for large text datasets
- Keeps texts as Python lists throughout the process

Text Cleaning:
- Removes @USER mentions and other usernames
- Removes URLs and links
- Filters out empty texts after cleaning
- Preserves hashtags and other meaningful content
"""

import os
import sys
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from collections import Counter
import logging
from typing import Tuple, List, Dict, Any
import argparse
import re

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_logging() -> logging.Logger:
    """Setup logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('dataset_build.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Clean text by removing usernames, URLs, and other noise.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return str(text)
    
    # Remove @USER mentions (usernames)
    text = re.sub(r'@USER\b', '', text)
    
    # Remove other common username patterns
    text = re.sub(r'@\w+\b', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove hashtags (optional - uncomment if needed)
    # text = re.sub(r'#\w+\b', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def load_superset_dataset() -> Tuple[List[str], List[int]]:
    """
    Load and process the manueltonneau/arabic-hate-speech-superset dataset.
    
    Returns:
        Tuple of (texts, labels) with binary labels
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading manueltonneau/arabic-hate-speech-superset dataset...")
    
    try:
        dataset = load_dataset("manueltonneau/arabic-hate-speech-superset")
        
        # Get the main split (usually 'train')
        if "train" in dataset:
            data = dataset["train"]
        else:
            data = dataset[list(dataset.keys())[0]]
        
        logger.info(f"Loaded {len(data)} samples from superset")
        logger.info(f"Columns: {data.column_names}")
        
        # Extract texts and labels and ensure they are Python lists
        texts = list(data['text'])
        labels = list(data['labels'])
        
        # Clean texts by removing usernames and other noise
        logger.info("Cleaning texts (removing @USER mentions, URLs, etc.)...")
        texts = [clean_text(text) for text in texts]
        
        # Filter out empty texts after cleaning
        original_count = len(texts)
        texts_labels_pairs = [(text, label) for text, label in zip(texts, labels) if text.strip()]
        texts = [pair[0] for pair in texts_labels_pairs]
        labels = [pair[1] for pair in texts_labels_pairs]
        filtered_count = len(texts)
        
        if original_count != filtered_count:
            logger.info(f"Filtered out {original_count - filtered_count} empty texts after cleaning")
        
        # Normalize labels to binary (1 = hate/offensive, 0 = non-hate)
        # The superset uses: 0 = non-hate, 1 = hate
        binary_labels = [1 if label == 1 else 0 for label in labels]
        
        # Log original distribution
        original_dist = Counter(labels)
        binary_dist = Counter(binary_labels)
        logger.info(f"Original label distribution: {dict(original_dist)}")
        logger.info(f"Binary label distribution: {dict(binary_dist)}")
        
        return texts, binary_labels
        
    except Exception as e:
        logger.error(f"Error loading superset dataset: {e}")
        raise


def load_egyptian_dataset() -> Tuple[List[str], List[int]]:
    """
    Load and process the IbrahimAmin/egyptian-arabic-hate-speech dataset.
    
    Returns:
        Tuple of (texts, labels) with binary labels
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading IbrahimAmin/egyptian-arabic-hate-speech dataset...")
    
    try:
        dataset = load_dataset("IbrahimAmin/egyptian-arabic-hate-speech")
        
        # Get the main split
        if "train" in dataset:
            data = dataset["train"]
        else:
            data = dataset[list(dataset.keys())[0]]
        
        logger.info(f"Loaded {len(data)} samples from Egyptian dataset")
        logger.info(f"Columns: {data.column_names}")
        
        # Extract texts and labels and ensure they are Python lists
        texts = list(data['text'])
        labels = list(data['label'])
        
        # Clean texts by removing usernames and other noise
        logger.info("Cleaning texts (removing @USER mentions, URLs, etc.)...")
        texts = [clean_text(text) for text in texts]
        
        # Filter out empty texts after cleaning
        original_count = len(texts)
        texts_labels_pairs = [(text, label) for text, label in zip(texts, labels) if text.strip()]
        texts = [pair[0] for pair in texts_labels_pairs]
        labels = [pair[1] for pair in texts_labels_pairs]
        filtered_count = len(texts)
        
        if original_count != filtered_count:
            logger.info(f"Filtered out {original_count - filtered_count} empty texts after cleaning")
        
        # Normalize labels to binary
        # This dataset uses: 0 = non-hate, 1 = hate
        binary_labels = [1 if label == 1 else 0 for label in labels]
        
        # Log distribution
        original_dist = Counter(labels)
        binary_dist = Counter(binary_labels)
        logger.info(f"Original label distribution: {dict(original_dist)}")
        logger.info(f"Binary label distribution: {dict(binary_dist)}")
        
        return texts, binary_labels
        
    except Exception as e:
        logger.error(f"Error loading Egyptian dataset: {e}")
        raise

def balance_superset(texts: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
    """
    Balance the superset dataset by:
    1. Taking ALL hate/offensive rows
    2. Randomly sampling an equal number of non-hate rows
    3. Concatenating them and shuffling
    
    Args:
        texts: List of texts
        labels: List of binary labels
        
    Returns:
        Tuple of balanced (texts, labels)
    """
    logger = logging.getLogger(__name__)
    
    # Ensure inputs are proper Python lists
    texts = list(texts)
    labels = list(labels)
    
    # Validate inputs
    if len(texts) != len(labels):
        raise ValueError(f"Texts and labels length mismatch: {len(texts)} vs {len(labels)}")
    
    if not texts:
        raise ValueError("Empty dataset provided")
    
    # Create DataFrame for easier manipulation (avoids numpy memory issues with strings)
    df = pd.DataFrame({'text': texts, 'label': labels})
    
    # Get hate and non-hate samples
    hate_df = df[df['label'] == 1]
    non_hate_df = df[df['label'] == 0]
    
    n_hate = len(hate_df)
    n_non_hate = len(non_hate_df)
    
    logger.info(f"Original distribution - Hate: {n_hate}, Non-hate: {n_non_hate}")
    logger.info(f"Taking ALL {n_hate} hate samples")
    
    # Sample equal number of non-hate samples
    if n_non_hate > n_hate:
        # Randomly sample non-hate samples to match hate count
        sampled_non_hate_df = non_hate_df.sample(n=n_hate, random_state=42)
        logger.info(f"Randomly sampled {n_hate} non-hate samples from {n_non_hate} available")
    else:
        # Use all non-hate samples if there are fewer than hate samples
        sampled_non_hate_df = non_hate_df
        logger.info(f"Using ALL {n_non_hate} non-hate samples (fewer than hate samples)")
    
    # Combine balanced samples and shuffle
    balanced_df = pd.concat([hate_df, sampled_non_hate_df]).sample(frac=1, random_state=42)
    
    # Verify final balanced distribution
    final_hate_count = (balanced_df['label'] == 1).sum()
    final_non_hate_count = (balanced_df['label'] == 0).sum()
    logger.info(f"Balanced distribution - Hate: {final_hate_count}, Non-hate: {final_non_hate_count}")
    logger.info(f"Total balanced samples: {len(balanced_df)}")
    
    # Verify perfect balance
    if final_hate_count == final_non_hate_count:
        logger.info("Perfect balance achieved!")
    else:
        logger.warning(f"Imbalance detected: {abs(final_hate_count - final_non_hate_count)} difference")
    
    # Return as lists (avoiding numpy arrays for text)
    return balanced_df['text'].tolist(), balanced_df['label'].tolist()

def merge_datasets(superset_texts: List[str], superset_labels: List[int],
                  egyptian_texts: List[str], egyptian_labels: List[int]) -> Tuple[List[str], List[int]]:
    """
    Merge the two datasets into one.
    
    Args:
        superset_texts: Superset texts
        superset_labels: Superset labels
        egyptian_texts: Egyptian dataset texts
        egyptian_labels: Egyptian dataset labels
        
    Returns:
        Tuple of merged (texts, labels)
    """
    logger = logging.getLogger(__name__)
    
    # Ensure all inputs are proper Python lists
    superset_texts = list(superset_texts)
    superset_labels = list(superset_labels)
    egyptian_texts = list(egyptian_texts)
    egyptian_labels = list(egyptian_labels)
    
    # Validate inputs
    if len(superset_texts) != len(superset_labels):
        raise ValueError(f"Superset texts and labels length mismatch: {len(superset_texts)} vs {len(superset_labels)}")
    
    if len(egyptian_texts) != len(egyptian_labels):
        raise ValueError(f"Egyptian texts and labels length mismatch: {len(egyptian_texts)} vs {len(egyptian_labels)}")
    
    # Combine all texts and labels
    all_texts = superset_texts + egyptian_texts
    all_labels = superset_labels + egyptian_labels
    
    logger.info(f"Total merged samples: {len(all_texts)}")
    logger.info(f"Final distribution: {dict(Counter(all_labels))}")
    
    # Create DataFrame for shuffling (avoids numpy memory issues)
    df = pd.DataFrame({'text': all_texts, 'label': all_labels})
    
    # Shuffle the merged dataset
    shuffled_df = df.sample(frac=1, random_state=42)
    
    return shuffled_df['text'].tolist(), shuffled_df['label'].tolist()

def split_dataset(texts: List[str], labels: List[int], 
                 train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/val/test sets with stratification.
    
    Args:
        texts: List of texts
        labels: List of labels
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger = logging.getLogger(__name__)
    
    # First split: separate train from temp (val + test)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, 
        test_size=(1 - train_ratio), 
        random_state=42, 
        stratify=labels
    )
    
    # Second split: separate val from test
    val_ratio_adjusted = val_ratio / (1 - train_ratio)  # Adjust ratio for remaining data
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels,
        test_size=(1 - val_ratio_adjusted),
        random_state=42,
        stratify=temp_labels
    )
    
    # Create DataFrames
    train_df = pd.DataFrame({'text': train_texts, 'label': train_labels})
    val_df = pd.DataFrame({'text': val_texts, 'label': val_labels})
    test_df = pd.DataFrame({'text': test_texts, 'label': test_labels})
    
    # Log split information
    logger.info(f"Train set: {len(train_df)} samples")
    logger.info(f"Train distribution: {dict(Counter(train_labels))}")
    logger.info(f"Validation set: {len(val_df)} samples")
    logger.info(f"Validation distribution: {dict(Counter(val_labels))}")
    logger.info(f"Test set: {len(test_df)} samples")
    logger.info(f"Test distribution: {dict(Counter(test_labels))}")
    
    return train_df, val_df, test_df

def save_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
                 output_dir: str) -> None:
    """
    Save datasets as CSV files.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        output_dir: Output directory path
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False, encoding='utf-8')
    val_df.to_csv(val_path, index=False, encoding='utf-8')
    test_df.to_csv(test_path, index=False, encoding='utf-8')
    
    logger.info(f"Datasets saved to {output_dir}")
    logger.info(f"Train: {train_path}")
    logger.info(f"Validation: {val_path}")
    logger.info(f"Test: {test_path}")

def main():
    """Main function to build the combined dataset."""
    parser = argparse.ArgumentParser(description='Build combined Arabic hate speech dataset')
    parser.add_argument('--output-dir', type=str, default='data/combined',
                       help='Output directory for combined datasets')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting combined dataset building process...")
    
    try:
        # Load both datasets
        logger.info("=" * 50)
        logger.info("LOADING DATASETS")
        logger.info("=" * 50)
        
        logger.info("Loading superset dataset...")
        superset_texts, superset_labels = load_superset_dataset()
        logger.info(f"Successfully loaded superset: {len(superset_texts)} samples")
        
        logger.info("Loading Egyptian dataset...")
        egyptian_texts, egyptian_labels = load_egyptian_dataset()
        logger.info(f"Successfully loaded Egyptian: {len(egyptian_texts)} samples")
        
        # Balance the superset dataset
        logger.info("=" * 50)
        logger.info("BALANCING SUPERSET DATASET")
        logger.info("=" * 50)
        
        logger.info("Balancing superset dataset...")
        balanced_texts, balanced_labels = balance_superset(superset_texts, superset_labels)
        logger.info(f"Successfully balanced superset: {len(balanced_texts)} samples")
        
        # Merge datasets
        logger.info("=" * 50)
        logger.info("MERGING DATASETS")
        logger.info("=" * 50)
        
        logger.info("Merging balanced superset with Egyptian dataset...")
        merged_texts, merged_labels = merge_datasets(
            balanced_texts, balanced_labels,
            egyptian_texts, egyptian_labels
        )
        logger.info(f"Successfully merged datasets: {len(merged_texts)} samples")
        
        # Split into train/val/test
        logger.info("=" * 50)
        logger.info("SPLITTING DATASET")
        logger.info("=" * 50)
        
        logger.info(f"Splitting dataset with ratios: train={args.train_ratio}, val={args.val_ratio}, test={1-args.train_ratio-args.val_ratio}")
        train_df, val_df, test_df = split_dataset(
            merged_texts, merged_labels,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio
        )
        logger.info("Successfully split dataset into train/val/test sets")
        
        # Save datasets
        logger.info("=" * 50)
        logger.info("SAVING DATASETS")
        logger.info("=" * 50)
        
        logger.info(f"Saving datasets to {args.output_dir}")
        save_datasets(train_df, val_df, test_df, args.output_dir)
        logger.info("Successfully saved all datasets")
        
        logger.info("=" * 50)
        logger.info("DATASET BUILDING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        
        # Print final statistics
        total_samples = len(train_df) + len(val_df) + len(test_df)
        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Train: {len(train_df)} ({len(train_df)/total_samples*100:.1f}%)")
        logger.info(f"Validation: {len(val_df)} ({len(val_df)/total_samples*100:.1f}%)")
        logger.info(f"Test: {len(test_df)} ({len(test_df)/total_samples*100:.1f}%)")
        
    except Exception as e:
        logger.error(f"Dataset building failed: {e}")
        raise

if __name__ == "__main__":
    main()
