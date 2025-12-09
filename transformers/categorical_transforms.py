"""
Categorical transformations
"""
import pandas as pd
import numpy as np
from .base import BaseTransformer, TransformError


class RemapTransformer(BaseTransformer):
    TRANSFORM_TYPE = "remap"
    SUPPORTED_INPUT_TYPES = ["any"]
    OUTPUT_TYPE = "any"
    
    def validate_params(self):
        if "mapping" not in self.params:
            raise TransformError("'mapping' parameter is required")
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        mapping = self.params.get("mapping")
        default = self.params.get("default")
        
        if default is not None:
            return series.map(mapping).fillna(default)
        else:
            return series.map(mapping)


class TopNTransformer(BaseTransformer):
    TRANSFORM_TYPE = "top_n"
    SUPPORTED_INPUT_TYPES = ["categorical", "string", "text"]
    OUTPUT_TYPE = "categorical"
    
    def validate_params(self):
        pass
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        n = self.params.get("n", 10)
        other_label = self.params.get("other_label", "Other")
        
        # Get top N values by frequency
        top_values = series.value_counts().head(n).index.tolist()
        
        return series.apply(lambda x: x if x in top_values else other_label)


class BinaryTransformer(BaseTransformer):
    TRANSFORM_TYPE = "binary"
    SUPPORTED_INPUT_TYPES = ["any"]
    OUTPUT_TYPE = "categorical"
    
    def validate_params(self):
        if "condition" not in self.params:
            raise TransformError("'condition' parameter is required")
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        condition = self.params.get("condition")
        true_label = self.params.get("true_label", "Yes")
        false_label = self.params.get("false_label", "No")
        
        # Evaluate condition (simple comparison)
        # Example: "> 50", "== 'Active'", "in ['A', 'B']"
        try:
            # This is simplified - in production, use safer evaluation
            result = series.apply(lambda x: eval(f"x {condition}"))
            return result.map({True: true_label, False: false_label})
        except Exception as e:
            raise TransformError(f"Failed to evaluate condition: {str(e)}", column=column_name)


class NullFillTransformer(BaseTransformer):
    TRANSFORM_TYPE = "null_fill"
    SUPPORTED_INPUT_TYPES = ["any"]
    OUTPUT_TYPE = "any"
    
    def validate_params(self):
        if "value" not in self.params:
            raise TransformError("'value' parameter is required")
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        value = self.params.get("value")
        return series.fillna(value)


class NullFillSmartTransformer(BaseTransformer):
    """Intelligent null filling"""
    TRANSFORM_TYPE = "null_fill_smart"
    SUPPORTED_INPUT_TYPES = ["any"]
    OUTPUT_TYPE = "any"
    
    def validate_params(self):
        method = self.params.get("method", "mean")
        valid_methods = ["mean", "median", "mode", "forward_fill", "backward_fill", "interpolate"]
        if method not in valid_methods:
            raise TransformError(f"method must be one of: {valid_methods}")
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        method = self.params.get("method", "mean")
        
        if method == "mean":
            if pd.api.types.is_numeric_dtype(series):
                return series.fillna(series.mean())
            else:
                raise TransformError("mean method only works for numeric columns", column=column_name)
        
        elif method == "median":
            if pd.api.types.is_numeric_dtype(series):
                return series.fillna(series.median())
            else:
                raise TransformError("median method only works for numeric columns", column=column_name)
        
        elif method == "mode":
            mode_value = series.mode()
            if len(mode_value) > 0:
                return series.fillna(mode_value[0])
            return series
        
        elif method == "forward_fill":
            return series.fillna(method='ffill')
        
        elif method == "backward_fill":
            return series.fillna(method='bfill')
        
        elif method == "interpolate":
            if pd.api.types.is_numeric_dtype(series):
                return series.interpolate()
            else:
                raise TransformError("interpolate method only works for numeric columns", column=column_name)
        
        return series


class CoalesceTransformer(BaseTransformer):
    """Return first non-null value from multiple columns"""
    TRANSFORM_TYPE = "coalesce"
    SUPPORTED_INPUT_TYPES = ["any"]
    OUTPUT_TYPE = "any"
    
    def validate_params(self):
        if "columns" not in self.params:
            raise TransformError("'columns' parameter is required")
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        # Note: This transformer needs access to the full dataframe
        # This is handled specially in the pipeline
        raise NotImplementedError("Coalesce requires full dataframe access")


class EncodeTransformer(BaseTransformer):
    """Encode categorical variables"""
    TRANSFORM_TYPE = "encode"
    SUPPORTED_INPUT_TYPES = ["categorical", "string", "text"]
    OUTPUT_TYPE = "numeric"
    
    def validate_params(self):
        method = self.params.get("method", "label")
        if method not in ["label", "ordinal", "frequency", "target"]:
            raise TransformError("method must be 'label', 'ordinal', 'frequency', or 'target'")
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        method = self.params.get("method", "label")
        
        if method == "label":
            # Simple label encoding
            categories = series.unique()
            mapping = {cat: i for i, cat in enumerate(categories)}
            return series.map(mapping)
        
        elif method == "frequency":
            # Encode by frequency
            freq = series.value_counts()
            return series.map(freq)
        
        elif method == "ordinal":
            # Use provided ordering
            order = self.params.get("order")
            if order is None:
                raise TransformError("ordinal encoding requires 'order' parameter")
            mapping = {cat: i for i, cat in enumerate(order)}
            return series.map(mapping)
        
        return series


class OneHotTransformer(BaseTransformer):
    """One-hot encoding (returns multiple columns)"""
    TRANSFORM_TYPE = "onehot"
    SUPPORTED_INPUT_TYPES = ["categorical", "string", "text"]
    OUTPUT_TYPE = "numeric"
    
    def validate_params(self):
        pass
    
    def transform(self, series: pd.Series, column_name: str) -> pd.DataFrame:
        prefix = self.params.get("prefix", column_name)
        max_categories = self.params.get("max_categories", 10)
        
        # Limit to top N categories
        top_cats = series.value_counts().head(max_categories).index
        series_limited = series.apply(lambda x: x if x in top_cats else "Other")
        
        return pd.get_dummies(series_limited, prefix=prefix)


class GroupTransformer(BaseTransformer):
    """Group rare categories"""
    TRANSFORM_TYPE = "group_rare"
    SUPPORTED_INPUT_TYPES = ["categorical", "string", "text"]
    OUTPUT_TYPE = "categorical"
    
    def validate_params(self):
        pass
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        threshold = self.params.get("threshold", 0.01)  # 1% threshold
        other_label = self.params.get("other_label", "Other")
        
        # Calculate frequency
        freq = series.value_counts(normalize=True)
        rare_categories = freq[freq < threshold].index
        
        return series.apply(lambda x: other_label if x in rare_categories else x)


class ConditionalTransformer(BaseTransformer):
    """SQL-like CASE WHEN"""
    TRANSFORM_TYPE = "conditional"
    SUPPORTED_INPUT_TYPES = ["any"]
    OUTPUT_TYPE = "any"
    
    def validate_params(self):
        if "conditions" not in self.params:
            raise TransformError("'conditions' parameter is required")
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        conditions = self.params.get("conditions", [])
        else_value = self.params.get("else", None)
        
        result = pd.Series([else_value] * len(series), index=series.index)
        
        for condition in conditions:
            when = condition.get("when")
            then = condition.get("then")
            
            if when and then is not None:
                try:
                    # Simplified condition evaluation
                    mask = series.apply(lambda x: eval(when.replace(column_name, "x")))
                    result[mask] = then
                except Exception as e:
                    raise TransformError(f"Failed to evaluate condition: {str(e)}", column=column_name)
        
        return result
