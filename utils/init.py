"""
Utility functions for data processing

This module provides intelligent type inference, date parsing,
and pattern analysis capabilities.

Usage:
    from utils import infer_column_role, smart_parse_datetime
    
    # Detect column type
    role = infer_column_role(series)  # Returns: 'numeric', 'datetime', 'categorical', 'text'
    
    # Parse dates intelligently
    dates = smart_parse_datetime(series)
    
    # Get transform suggestions
    suggestions = suggest_column_transforms(series)
"""

from .type_inference import (
    infer_column_role,
    smart_parse_datetime,
    analyze_column_patterns,
    suggest_column_transforms,
    detect_date_format
)

__version__ = "2.0.0"

__all__ = [
    'infer_column_role',
    'smart_parse_datetime',
    'analyze_column_patterns',
    'suggest_column_transforms',
    'detect_date_format'
]
