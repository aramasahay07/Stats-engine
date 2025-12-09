"""
Base transformer class and utilities
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np


class TransformError(Exception):
    """Custom exception for transform errors"""
    def __init__(self, message: str, column: str = None, suggestion: str = None):
        self.message = message
        self.column = column
        self.suggestion = suggestion
        super().__init__(self.message)


class BaseTransformer(ABC):
    """Base class for all transformers"""
    
    # Subclasses must define these
    TRANSFORM_TYPE: str = None
    SUPPORTED_INPUT_TYPES: List[str] = []
    OUTPUT_TYPE: str = None
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self.validate_params()
    
    @abstractmethod
    def validate_params(self):
        """Validate transformation parameters"""
        pass
    
    @abstractmethod
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        """Apply transformation to a pandas Series"""
        pass
    
    def can_transform(self, series: pd.Series) -> bool:
        """Check if this transformer can handle the series dtype"""
        dtype_str = str(series.dtype)
        
        # Check numeric types
        if "numeric" in self.SUPPORTED_INPUT_TYPES:
            if pd.api.types.is_numeric_dtype(series):
                return True
        
        # Check datetime types
        if "datetime" in self.SUPPORTED_INPUT_TYPES:
            if pd.api.types.is_datetime64_any_dtype(series):
                return True
        
        # Check string/text types
        if "string" in self.SUPPORTED_INPUT_TYPES or "text" in self.SUPPORTED_INPUT_TYPES:
            if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
                return True
        
        # Check categorical
        if "categorical" in self.SUPPORTED_INPUT_TYPES:
            if pd.api.types.is_categorical_dtype(series):
                return True
        
        # Any type
        if "any" in self.SUPPORTED_INPUT_TYPES:
            return True
        
        return False
    
    def handle_nulls(self, series: pd.Series, null_strategy: str = "preserve") -> pd.Series:
        """Handle null values based on strategy"""
        if null_strategy == "ignore":
            return series.dropna()
        elif null_strategy == "preserve":
            return series
        elif null_strategy == "default":
            default_value = self.params.get("null_default")
            return series.fillna(default_value)
        elif null_strategy == "error":
            if series.isna().any():
                raise TransformError(
                    f"Null values found in column, but transform does not allow nulls",
                    suggestion="Use null_fill transform first or set null_strategy to 'ignore'"
                )
        return series
    
    def get_output_column_name(self, original_name: str) -> str:
        """Generate output column name with transform suffix"""
        suffix = self.params.get("suffix", self.TRANSFORM_TYPE)
        return f"{original_name}_{suffix}"
    
    def get_description(self) -> str:
        """Return human-readable description of this transform"""
        return f"{self.TRANSFORM_TYPE} transformation"
    
    def get_metadata(self, input_series: pd.Series, output_series: pd.Series) -> Dict[str, Any]:
        """Generate metadata about the transformation"""
        return {
            "source_column": input_series.name,
            "transform_type": self.TRANSFORM_TYPE,
            "null_count": int(output_series.isna().sum()),
            "unique_values": int(output_series.nunique()),
            "sample_output": output_series.dropna().head(5).tolist() if len(output_series) > 0 else []
        }


class TransformPipeline:
    """Execute a chain of transforms"""
    
    def __init__(self, transforms: List[BaseTransformer]):
        self.transforms = transforms
    
    def execute(self, series: pd.Series, column_name: str) -> pd.Series:
        """Execute transforms in sequence"""
        result = series.copy()
        
        for i, transformer in enumerate(self.transforms):
            if not transformer.can_transform(result):
                raise TransformError(
                    f"Transform {i+1} ({transformer.TRANSFORM_TYPE}) cannot handle "
                    f"output type from previous transform",
                    column=column_name
                )
            
            try:
                result = transformer.transform(result, column_name)
            except Exception as e:
                raise TransformError(
                    f"Transform {i+1} ({transformer.TRANSFORM_TYPE}) failed: {str(e)}",
                    column=column_name
                )
        
        return result
    
    def validate(self) -> List[str]:
        """Validate the entire pipeline and return warnings"""
        warnings = []
        
        if len(self.transforms) == 0:
            warnings.append("Empty transform pipeline")
        
        # Check for type compatibility between chained transforms
        for i in range(len(self.transforms) - 1):
            current = self.transforms[i]
            next_transform = self.transforms[i + 1]
            
            if current.OUTPUT_TYPE not in next_transform.SUPPORTED_INPUT_TYPES:
                if "any" not in next_transform.SUPPORTED_INPUT_TYPES:
                    warnings.append(
                        f"Potential type mismatch: {current.TRANSFORM_TYPE} outputs "
                        f"{current.OUTPUT_TYPE}, but {next_transform.TRANSFORM_TYPE} "
                        f"expects {next_transform.SUPPORTED_INPUT_TYPES}"
                    )
        
        return warnings


def create_transform_key(transform_type: str, params: Dict[str, Any]) -> str:
    """Create a cache key for a transform with params"""
    # Sort params for consistent hashing
    param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
    return f"{transform_type}:{param_str}" if param_str else transform_type
