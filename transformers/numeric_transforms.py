"""
Numeric transformations
"""
import pandas as pd
import numpy as np
from typing import List
from .base import BaseTransformer, TransformError


class BucketTransformer(BaseTransformer):
    TRANSFORM_TYPE = "bucket"
    SUPPORTED_INPUT_TYPES = ["numeric"]
    OUTPUT_TYPE = "categorical"
    
    def validate_params(self):
        bins = self.params.get("bins")
        bin_count = self.params.get("bin_count")
        if bins is None and bin_count is None:
            raise TransformError("Either 'bins' or 'bin_count' must be provided")
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        bins = self.params.get("bins")
        labels = self.params.get("labels")
        
        if bins:
            result = pd.cut(series, bins=bins, labels=labels, include_lowest=True)
        else:
            bin_count = self.params.get("bin_count", 5)
            result = pd.cut(series, bins=bin_count, labels=labels)
        
        # Convert intervals to strings if no custom labels
        if labels is None:
            result = result.astype(str)
        
        return result


class PercentileBucketTransformer(BaseTransformer):
    TRANSFORM_TYPE = "percentile_bucket"
    SUPPORTED_INPUT_TYPES = ["numeric"]
    OUTPUT_TYPE = "categorical"
    
    def validate_params(self):
        pass
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        quantiles = self.params.get("quantiles", [0.25, 0.5, 0.75, 1.0])
        labels = self.params.get("labels")
        
        if labels is None:
            # Generate default labels
            labels = []
            prev = 0
            for q in quantiles:
                labels.append(f"{int(prev*100)}-{int(q*100)}%")
                prev = q
        
        result = pd.qcut(series, q=quantiles, labels=labels, duplicates='drop')
        return result


class RoundTransformer(BaseTransformer):
    TRANSFORM_TYPE = "round"
    SUPPORTED_INPUT_TYPES = ["numeric"]
    OUTPUT_TYPE = "numeric"
    
    def validate_params(self):
        pass
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        precision = self.params.get("precision", 0)
        return series.round(precision)


class NormalizeTransformer(BaseTransformer):
    TRANSFORM_TYPE = "normalize"
    SUPPORTED_INPUT_TYPES = ["numeric"]
    OUTPUT_TYPE = "numeric"
    
    def validate_params(self):
        method = self.params.get("method", "minmax")
        if method not in ["minmax", "zscore"]:
            raise TransformError("method must be 'minmax' or 'zscore'")
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        method = self.params.get("method", "minmax")
        
        if method == "minmax":
            min_val = series.min()
            max_val = series.max()
            if max_val == min_val:
                return pd.Series(0.5, index=series.index)
            return (series - min_val) / (max_val - min_val)
        else:  # zscore
            return (series - series.mean()) / series.std()


class LogTransformer(BaseTransformer):
    TRANSFORM_TYPE = "log"
    SUPPORTED_INPUT_TYPES = ["numeric"]
    OUTPUT_TYPE = "numeric"
    
    def validate_params(self):
        pass
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        base = self.params.get("base", "natural")
        
        # Handle negative values
        if (series <= 0).any():
            raise TransformError(
                f"Log transformation requires positive values",
                column=column_name,
                suggestion="Use 'absolute' transform first or add a constant"
            )
        
        if base == "natural":
            return np.log(series)
        elif base == 10:
            return np.log10(series)
        elif base == 2:
            return np.log2(series)
        else:
            return np.log(series) / np.log(base)


class AbsoluteTransformer(BaseTransformer):
    TRANSFORM_TYPE = "absolute"
    SUPPORTED_INPUT_TYPES = ["numeric"]
    OUTPUT_TYPE = "numeric"
    
    def validate_params(self):
        pass
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        return series.abs()


class SignTransformer(BaseTransformer):
    TRANSFORM_TYPE = "sign"
    SUPPORTED_INPUT_TYPES = ["numeric"]
    OUTPUT_TYPE = "categorical"
    
    def validate_params(self):
        pass
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        return pd.cut(series, bins=[-np.inf, 0, np.inf], 
                     labels=["negative", "positive"], include_lowest=True)


class BucketDistributionTransformer(BaseTransformer):
    """Advanced distribution-aware binning"""
    TRANSFORM_TYPE = "bucket_distribution"
    SUPPORTED_INPUT_TYPES = ["numeric"]
    OUTPUT_TYPE = "categorical"
    
    def validate_params(self):
        method = self.params.get("method", "equal_frequency")
        if method not in ["equal_frequency", "equal_width", "exponential"]:
            raise TransformError("method must be 'equal_frequency', 'equal_width', or 'exponential'")
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        method = self.params.get("method", "equal_frequency")
        n_bins = self.params.get("n_bins", 5)
        labels = self.params.get("labels")
        
        if method == "equal_frequency":
            result = pd.qcut(series, q=n_bins, labels=labels, duplicates='drop')
        elif method == "equal_width":
            result = pd.cut(series, bins=n_bins, labels=labels)
        elif method == "exponential":
            # Exponential bins for skewed data
            min_val = series.min()
            max_val = series.max()
            bins = np.logspace(np.log10(min_val + 1), np.log10(max_val + 1), n_bins + 1) - 1
            result = pd.cut(series, bins=bins, labels=labels, include_lowest=True)
        
        if labels is None:
            result = result.astype(str)
        
        return result


class RollingTransformer(BaseTransformer):
    """Rolling window calculations"""
    TRANSFORM_TYPE = "rolling"
    SUPPORTED_INPUT_TYPES = ["numeric"]
    OUTPUT_TYPE = "numeric"
    
    def validate_params(self):
        window = self.params.get("window")
        if window is None or window < 1:
            raise TransformError("window must be a positive integer")
        
        function = self.params.get("function", "mean")
        if function not in ["mean", "sum", "std", "min", "max", "median"]:
            raise TransformError("function must be one of: mean, sum, std, min, max, median")
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        window = self.params.get("window", 7)
        function = self.params.get("function", "mean")
        
        rolling = series.rolling(window=window, min_periods=1)
        
        if function == "mean":
            return rolling.mean()
        elif function == "sum":
            return rolling.sum()
        elif function == "std":
            return rolling.std()
        elif function == "min":
            return rolling.min()
        elif function == "max":
            return rolling.max()
        elif function == "median":
            return rolling.median()


class DifferenceTransformer(BaseTransformer):
    """Calculate difference with lag"""
    TRANSFORM_TYPE = "difference"
    SUPPORTED_INPUT_TYPES = ["numeric"]
    OUTPUT_TYPE = "numeric"
    
    def validate_params(self):
        pass
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        lag = self.params.get("lag", 1)
        return series.diff(periods=lag)


class PercentChangeTransformer(BaseTransformer):
    """Calculate percent change"""
    TRANSFORM_TYPE = "percent_change"
    SUPPORTED_INPUT_TYPES = ["numeric"]
    OUTPUT_TYPE = "numeric"
    
    def validate_params(self):
        pass
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        periods = self.params.get("periods", 1)
        return series.pct_change(periods=periods) * 100
