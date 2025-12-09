"""
Date and time transformations
"""
import pandas as pd
import numpy as np
from typing import Any, Dict
from .base import BaseTransformer, TransformError


class MonthTransformer(BaseTransformer):
    TRANSFORM_TYPE = "month"
    SUPPORTED_INPUT_TYPES = ["datetime"]
    OUTPUT_TYPE = "categorical"
    
    def validate_params(self):
        format_type = self.params.get("format", "name")
        if format_type not in ["name", "number", "short"]:
            raise TransformError("format must be 'name', 'number', or 'short'")
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        format_type = self.params.get("format", "name")
        
        if format_type == "name":
            return series.dt.month_name()
        elif format_type == "short":
            return series.dt.strftime("%b")
        else:  # number
            return series.dt.month


class MonthYearTransformer(BaseTransformer):
    TRANSFORM_TYPE = "month_year"
    SUPPORTED_INPUT_TYPES = ["datetime"]
    OUTPUT_TYPE = "categorical"
    
    def validate_params(self):
        format_type = self.params.get("format", "short")
        if format_type not in ["short", "long"]:
            raise TransformError("format must be 'short' or 'long'")
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        format_type = self.params.get("format", "short")
        
        if format_type == "short":
            return series.dt.strftime("%b %Y")
        else:
            return series.dt.strftime("%B %Y")


class YearTransformer(BaseTransformer):
    TRANSFORM_TYPE = "year"
    SUPPORTED_INPUT_TYPES = ["datetime"]
    OUTPUT_TYPE = "numeric"
    
    def validate_params(self):
        pass
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        return series.dt.year


class QuarterTransformer(BaseTransformer):
    TRANSFORM_TYPE = "quarter"
    SUPPORTED_INPUT_TYPES = ["datetime"]
    OUTPUT_TYPE = "categorical"
    
    def validate_params(self):
        pass
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        include_year = self.params.get("include_year", False)
        
        if include_year:
            return "Q" + series.dt.quarter.astype(str) + " " + series.dt.year.astype(str)
        else:
            return "Q" + series.dt.quarter.astype(str)


class WeekdayTransformer(BaseTransformer):
    TRANSFORM_TYPE = "weekday"
    SUPPORTED_INPUT_TYPES = ["datetime"]
    OUTPUT_TYPE = "categorical"
    
    def validate_params(self):
        format_type = self.params.get("format", "name")
        if format_type not in ["name", "number", "short"]:
            raise TransformError("format must be 'name', 'number', or 'short'")
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        format_type = self.params.get("format", "name")
        
        if format_type == "name":
            return series.dt.day_name()
        elif format_type == "short":
            return series.dt.strftime("%a")
        else:  # number
            return series.dt.dayofweek


class WeekTransformer(BaseTransformer):
    TRANSFORM_TYPE = "week"
    SUPPORTED_INPUT_TYPES = ["datetime"]
    OUTPUT_TYPE = "numeric"
    
    def validate_params(self):
        pass
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        iso = self.params.get("iso", False)
        
        if iso:
            return series.dt.isocalendar().week
        else:
            return series.dt.strftime("%U").astype(int)


class DayTransformer(BaseTransformer):
    TRANSFORM_TYPE = "day"
    SUPPORTED_INPUT_TYPES = ["datetime"]
    OUTPUT_TYPE = "numeric"
    
    def validate_params(self):
        pass
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        return series.dt.day


class HourTransformer(BaseTransformer):
    TRANSFORM_TYPE = "hour"
    SUPPORTED_INPUT_TYPES = ["datetime"]
    OUTPUT_TYPE = "numeric"
    
    def validate_params(self):
        pass
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        return series.dt.hour


class DateOnlyTransformer(BaseTransformer):
    TRANSFORM_TYPE = "date_only"
    SUPPORTED_INPUT_TYPES = ["datetime"]
    OUTPUT_TYPE = "datetime"
    
    def validate_params(self):
        pass
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        format_str = self.params.get("format", "%Y-%m-%d")
        return series.dt.strftime(format_str)


class FiscalQuarterTransformer(BaseTransformer):
    TRANSFORM_TYPE = "fiscal_quarter"
    SUPPORTED_INPUT_TYPES = ["datetime"]
    OUTPUT_TYPE = "categorical"
    
    def validate_params(self):
        fiscal_start = self.params.get("fiscal_start_month", 1)
        if not 1 <= fiscal_start <= 12:
            raise TransformError("fiscal_start_month must be between 1 and 12")
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        fiscal_start = self.params.get("fiscal_start_month", 1)
        
        # Adjust month to fiscal year
        fiscal_month = (series.dt.month - fiscal_start) % 12 + 1
        fiscal_quarter = ((fiscal_month - 1) // 3) + 1
        
        include_year = self.params.get("include_year", False)
        if include_year:
            # Calculate fiscal year
            fiscal_year = series.dt.year
            fiscal_year = np.where(series.dt.month < fiscal_start, fiscal_year - 1, fiscal_year)
            return "FQ" + fiscal_quarter.astype(str) + " FY" + fiscal_year.astype(str)
        else:
            return "FQ" + fiscal_quarter.astype(str)


class TimeFeatureTransformer(BaseTransformer):
    """Extract multiple time-based features"""
    TRANSFORM_TYPE = "time_features"
    SUPPORTED_INPUT_TYPES = ["datetime"]
    OUTPUT_TYPE = "categorical"
    
    def validate_params(self):
        features = self.params.get("features", [])
        valid_features = ["is_weekend", "is_month_end", "is_month_start", 
                          "is_quarter_end", "is_year_end", "season"]
        for f in features:
            if f not in valid_features:
                raise TransformError(f"Invalid feature: {f}. Valid: {valid_features}")
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        features = self.params.get("features", ["is_weekend"])
        feature = features[0]  # For simplicity, handle one at a time
        
        if feature == "is_weekend":
            return series.dt.dayofweek >= 5
        elif feature == "is_month_end":
            return series.dt.is_month_end
        elif feature == "is_month_start":
            return series.dt.is_month_start
        elif feature == "is_quarter_end":
            return series.dt.is_quarter_end
        elif feature == "is_year_end":
            return series.dt.is_year_end
        elif feature == "season":
            month = series.dt.month
            return pd.cut(month, bins=[0, 3, 6, 9, 12], 
                         labels=["Winter", "Spring", "Summer", "Fall"])
        
        return series


class AgeFromDateTransformer(BaseTransformer):
    """Calculate age from birth date"""
    TRANSFORM_TYPE = "age_from_date"
    SUPPORTED_INPUT_TYPES = ["datetime"]
    OUTPUT_TYPE = "numeric"
    
    def validate_params(self):
        pass
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        reference_date = self.params.get("reference_date", "today")
        
        if reference_date == "today":
            ref = pd.Timestamp.now()
        else:
            ref = pd.to_datetime(reference_date)
        
        age = (ref - series).dt.days / 365.25
        return age.round().astype(int)
