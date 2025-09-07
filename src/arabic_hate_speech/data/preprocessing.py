"""
Text preprocessing utilities for Arabic Hate Speech Detection.
"""

import re
import string
from typing import List, Optional


class TextPreprocessor:
    """
    Text preprocessing class for Arabic text.
    """
    
    def __init__(self, 
                 remove_diacritics: bool = True,
                 remove_punctuation: bool = False,
                 normalize_whitespace: bool = True,
                 remove_urls: bool = True,
                 remove_mentions: bool = True,
                 remove_hashtags: bool = False):
        """
        Initialize the text preprocessor.
        
        Args:
            remove_diacritics: Whether to remove Arabic diacritics
            remove_punctuation: Whether to remove punctuation
            normalize_whitespace: Whether to normalize whitespace
            remove_urls: Whether to remove URLs
            remove_mentions: Whether to remove @mentions
            remove_hashtags: Whether to remove #hashtags
        """
        self.remove_diacritics = remove_diacritics
        self.remove_punctuation = remove_punctuation
        self.normalize_whitespace = normalize_whitespace
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        
        # Arabic diacritics pattern
        self.diacritics_pattern = re.compile(r'[\u064B-\u0652\u0670\u0640]')
        
        # URL pattern
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        
        # Mention pattern
        self.mention_pattern = re.compile(r'@\w+')
        
        # Hashtag pattern
        self.hashtag_pattern = re.compile(r'#\w+')
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess Arabic text.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub('', text)
        
        # Remove mentions
        if self.remove_mentions:
            text = self.mention_pattern.sub('', text)
        
        # Remove hashtags
        if self.remove_hashtags:
            text = self.hashtag_pattern.sub('', text)
        
        # Remove diacritics
        if self.remove_diacritics:
            text = self.diacritics_pattern.sub('', text)
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Normalize whitespace
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of texts to preprocess
            
        Returns:
            List of preprocessed texts
        """
        return [self.clean_text(text) for text in texts]
    
    def get_stats(self, texts: List[str]) -> dict:
        """
        Get statistics about the text preprocessing.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary containing preprocessing statistics
        """
        stats = {
            "total_texts": len(texts),
            "avg_length_before": sum(len(text) for text in texts) / len(texts) if texts else 0,
            "avg_length_after": 0,
            "empty_texts": 0,
        }
        
        if texts:
            processed_texts = self.preprocess_batch(texts)
            stats["avg_length_after"] = sum(len(text) for text in processed_texts) / len(processed_texts)
            stats["empty_texts"] = sum(1 for text in processed_texts if not text.strip())
        
        return stats


def clean_text(text: str) -> str:
    """
    Simple text cleaning function for backward compatibility.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    preprocessor = TextPreprocessor()
    return preprocessor.clean_text(text)
