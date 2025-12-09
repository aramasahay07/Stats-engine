"""
Intelligent and ML-enhanced transformations
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from .base import BaseTransformer, TransformError


class OutlierDetectionTransformer(BaseTransformer):
    """Detect and flag outliers"""
    TRANSFORM_TYPE = "detect_outliers"
    SUPPORTED_INPUT_TYPES = ["numeric"]
    OUTPUT_TYPE = "categorical"
    
    def validate_params(self):
        method = self.params.get("method", "iqr")
        if method not in ["iqr", "zscore", "isolation_forest"]:
            raise TransformError("method must be 'iqr', 'zscore', or 'isolation_forest'")
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        method = self.params.get("method", "iqr")
        
        if method == "iqr":
            threshold = self.params.get("threshold", 1.5)
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            return ((series < lower) | (series > upper)).map({True: "Outlier", False: "Normal"})
        
        elif method == "zscore":
            threshold = self.params.get("threshold", 3)
            z_scores = np.abs((series - series.mean()) / series.std())
            return (z_scores > threshold).map({True: "Outlier", False: "Normal"})
        
        elif method == "isolation_forest":
            contamination = self.params.get("contamination", 0.1)
            try:
                model = IsolationForest(contamination=contamination, random_state=42)
                predictions = model.fit_predict(series.values.reshape(-1, 1))
                return pd.Series(predictions, index=series.index).map({
                    1: "Normal", -1: "Outlier"
                })
            except Exception as e:
                raise TransformError(f"Isolation Forest failed: {str(e)}", column=column_name)
        
        return series


class BucketSmartTransformer(BaseTransformer):
    """Intelligent bucketing using statistical methods"""
    TRANSFORM_TYPE = "bucket_smart"
    SUPPORTED_INPUT_TYPES = ["numeric"]
    OUTPUT_TYPE = "categorical"
    
    def validate_params(self):
        method = self.params.get("method", "outlier_aware")
        if method not in ["outlier_aware", "quantile", "kmeans"]:
            raise TransformError("method must be 'outlier_aware', 'quantile', or 'kmeans'")
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        method = self.params.get("method", "outlier_aware")
        n_bins = self.params.get("n_bins", 5)
        
        if method == "outlier_aware":
            # Remove outliers for bin calculation
            threshold = self.params.get("outlier_threshold", 1.5)
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            
            # Create bins based on non-outlier data
            non_outlier = series[(series >= lower) & (series <= upper)]
            bins = pd.qcut(non_outlier, q=n_bins, retbins=True, duplicates='drop')[1]
            
            # Apply bins to all data
            return pd.cut(series, bins=bins, include_lowest=True).astype(str)
        
        elif method == "quantile":
            return pd.qcut(series, q=n_bins, duplicates='drop').astype(str)
        
        elif method == "kmeans":
            # Use k-means for natural clustering
            try:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_bins, random_state=42)
                labels = kmeans.fit_predict(series.values.reshape(-1, 1))
                
                # Sort cluster labels by cluster center
                centers = kmeans.cluster_centers_.flatten()
                sorted_labels = np.argsort(centers)
                label_map = {old: new for new, old in enumerate(sorted_labels)}
                
                result = pd.Series(labels, index=series.index).map(label_map)
                return result.map(lambda x: f"Group {x+1}")
            except Exception as e:
                raise TransformError(f"K-means clustering failed: {str(e)}", column=column_name)
        
        return series


class CastSmartTransformer(BaseTransformer):
    """Smart type coercion with validation"""
    TRANSFORM_TYPE = "cast_smart"
    SUPPORTED_INPUT_TYPES = ["any"]
    OUTPUT_TYPE = "any"
    
    def validate_params(self):
        target_type = self.params.get("target_type")
        if target_type not in ["numeric", "datetime", "boolean", "string"]:
            raise TransformError("target_type must be 'numeric', 'datetime', 'boolean', or 'string'")
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        target_type = self.params.get("target_type")
        errors = self.params.get("errors", "coerce")
        
        try:
            if target_type == "numeric":
                result = pd.to_numeric(series, errors=errors)
                
                # Validate range if specified
                validation = self.params.get("validation", {})
                if "range" in validation:
                    min_val, max_val = validation["range"]
                    mask = (result >= min_val) & (result <= max_val)
                    if not mask.all():
                        if errors == "raise":
                            raise TransformError(f"Values outside range [{min_val}, {max_val}]")
                        result[~mask] = np.nan
                
                return result
            
            elif target_type == "datetime":
                format_str = self.params.get("format")
                return pd.to_datetime(series, format=format_str, errors=errors)
            
            elif target_type == "boolean":
                true_values = self.params.get("true_values", ["true", "yes", "1", "t", "y"])
                false_values = self.params.get("false_values", ["false", "no", "0", "f", "n"])
                
                result = series.astype(str).str.lower().str.strip()
                bool_series = pd.Series(np.nan, index=series.index)
                bool_series[result.isin(true_values)] = True
                bool_series[result.isin(false_values)] = False
                
                if errors == "raise" and bool_series.isna().any():
                    raise TransformError("Unable to convert all values to boolean")
                
                return bool_series
            
            elif target_type == "string":
                return series.astype(str)
        
        except Exception as e:
            if errors == "raise":
                raise TransformError(f"Type conversion failed: {str(e)}", column=column_name)
            return series


class FuzzyMatchTransformer(BaseTransformer):
    """Standardize using fuzzy string matching"""
    TRANSFORM_TYPE = "fuzzy_match"
    SUPPORTED_INPUT_TYPES = ["string", "text"]
    OUTPUT_TYPE = "string"
    
    def validate_params(self):
        if "target_values" not in self.params:
            raise TransformError("'target_values' parameter is required")
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        try:
            from fuzzywuzzy import process
        except ImportError:
            raise TransformError("fuzzywuzzy library required for fuzzy matching")
        
        target_values = self.params.get("target_values")
        threshold = self.params.get("similarity_threshold", 80)
        
        def find_best_match(value):
            if pd.isna(value):
                return value
            match, score = process.extractOne(str(value), target_values)
            return match if score >= threshold else value
        
        return series.apply(find_best_match)


class SeasonalityTransformer(BaseTransformer):
    """Extract seasonal patterns"""
    TRANSFORM_TYPE = "seasonality"
    SUPPORTED_INPUT_TYPES = ["datetime"]
    OUTPUT_TYPE = "categorical"
    
    def validate_params(self):
        pass
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        feature = self.params.get("feature", "season")
        
        if feature == "season":
            # Meteorological seasons
            month = series.dt.month
            conditions = [
                month.isin([12, 1, 2]),
                month.isin([3, 4, 5]),
                month.isin([6, 7, 8]),
                month.isin([9, 10, 11])
            ]
            choices = ["Winter", "Spring", "Summer", "Fall"]
            return pd.Series(np.select(conditions, choices), index=series.index)
        
        elif feature == "is_weekend":
            return series.dt.dayofweek >= 5
        
        elif feature == "is_business_day":
            return series.dt.dayofweek < 5
        
        elif feature == "time_of_day":
            hour = series.dt.hour
            conditions = [
                hour < 6,
                (hour >= 6) & (hour < 12),
                (hour >= 12) & (hour < 18),
                (hour >= 18) & (hour < 22),
                hour >= 22
            ]
            choices = ["Night", "Morning", "Afternoon", "Evening", "Late Night"]
            return pd.Series(np.select(conditions, choices), index=series.index)
        
        return series


class InteractionTransformer(BaseTransformer):
    """Create interaction features between columns"""
    TRANSFORM_TYPE = "interaction"
    SUPPORTED_INPUT_TYPES = ["numeric"]
    OUTPUT_TYPE = "numeric"
    
    def validate_params(self):
        if "columns" not in self.params or len(self.params["columns"]) < 2:
            raise TransformError("'columns' parameter must contain at least 2 column names")
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        # This requires dataframe access - handled specially
        raise NotImplementedError("Interaction requires full dataframe access")


class BinningAdaptiveTransformer(BaseTransformer):
    """Adaptive binning that optimizes for target variable"""
    TRANSFORM_TYPE = "binning_adaptive"
    SUPPORTED_INPUT_TYPES = ["numeric"]
    OUTPUT_TYPE = "categorical"
    
    def validate_params(self):
        pass
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        n_bins = self.params.get("n_bins", 5)
        
        # Use decision tree to find optimal splits
        try:
            from sklearn.tree import DecisionTreeClassifier
            
            # This would need a target variable - for now, use quantiles
            return pd.qcut(series, q=n_bins, duplicates='drop').astype(str)
        
        except Exception:
            # Fallback to standard quantiles
            return pd.qcut(series, q=n_bins, duplicates='drop').astype(str)


class WindowAggregationTransformer(BaseTransformer):
    """Rolling window with grouping support"""
    TRANSFORM_TYPE = "window_aggregation"
    SUPPORTED_INPUT_TYPES = ["numeric"]
    OUTPUT_TYPE = "numeric"
    
    def validate_params(self):
        if "window" not in self.params:
            raise TransformError("'window' parameter is required")
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        window = self.params.get("window")
        function = self.params.get("function", "mean")
        min_periods = self.params.get("min_periods", 1)
        
        rolling = series.rolling(window=window, min_periods=min_periods)
        
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
        
        return series
