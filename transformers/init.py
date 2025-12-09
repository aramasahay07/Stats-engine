"""
Data transformation engine for AI Data Lab

This module provides 60+ intelligent transforms for data preprocessing,
feature engineering, and analysis preparation.

Usage:
    from transformers.registry import registry
    
    # Get a transformer
    transformer = registry.create('month', {'format': 'name'})
    
    # Apply transform
    result = transformer.transform(date_series, 'date_column')
    
    # Get suggestions
    suggestions = registry.suggest_transforms(series, 'column_name')
"""

from .registry import registry
from .base import BaseTransformer, TransformError, TransformPipeline

__version__ = "2.0.0"

__all__ = [
    'registry',
    'BaseTransformer',
    'TransformError',
    'TransformPipeline'
]
