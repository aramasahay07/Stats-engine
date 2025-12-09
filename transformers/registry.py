"""
Transform registry and factory
"""
from typing import Dict, Type, List, Optional
from .base import BaseTransformer, TransformError
from .datetime_transforms import *
from .numeric_transforms import *
from .text_transforms import *
from .categorical_transforms import *
from .smart_transforms import *


class TransformRegistry:
    """Central registry for all transformers"""
    
    def __init__(self):
        self._transformers: Dict[str, Type[BaseTransformer]] = {}
        self._register_all()
    
    def _register_all(self):
        """Register all available transformers"""
        # DateTime transformers
        self.register(MonthTransformer)
        self.register(MonthYearTransformer)
        self.register(YearTransformer)
        self.register(QuarterTransformer)
        self.register(WeekdayTransformer)
        self.register(WeekTransformer)
        self.register(DayTransformer)
        self.register(HourTransformer)
        self.register(DateOnlyTransformer)
        self.register(FiscalQuarterTransformer)
        self.register(TimeFeatureTransformer)
        self.register(AgeFromDateTransformer)
        
        # Numeric transformers
        self.register(BucketTransformer)
        self.register(PercentileBucketTransformer)
        self.register(RoundTransformer)
        self.register(NormalizeTransformer)
        self.register(LogTransformer)
        self.register(AbsoluteTransformer)
        self.register(SignTransformer)
        self.register(BucketDistributionTransformer)
        self.register(RollingTransformer)
        self.register(DifferenceTransformer)
        self.register(PercentChangeTransformer)
        
        # Text transformers
        self.register(LowercaseTransformer)
        self.register(UppercaseTransformer)
        self.register(TrimTransformer)
        self.register(LengthTransformer)
        self.register(ExtractTransformer)
        self.register(ReplaceTransformer)
        self.register(FirstWordTransformer)
        self.register(WordCountTransformer)
        self.register(ContainsTransformer)
        self.register(TitleCaseTransformer)
        self.register(ExtractEntityTransformer)
        self.register(StandardizeTransformer)
        self.register(SlugifyTransformer)
        
        # Categorical transformers
        self.register(RemapTransformer)
        self.register(TopNTransformer)
        self.register(BinaryTransformer)
        self.register(NullFillTransformer)
        self.register(NullFillSmartTransformer)
        self.register(EncodeTransformer)
        self.register(OneHotTransformer)
        self.register(GroupTransformer)
        self.register(ConditionalTransformer)
        
        # Smart transformers
        self.register(OutlierDetectionTransformer)
        self.register(BucketSmartTransformer)
        self.register(CastSmartTransformer)
        self.register(SeasonalityTransformer)
        self.register(WindowAggregationTransformer)
        self.register(BinningAdaptiveTransformer)
    
    def register(self, transformer_class: Type[BaseTransformer]):
        """Register a transformer class"""
        if not issubclass(transformer_class, BaseTransformer):
            raise ValueError(f"{transformer_class} must be a subclass of BaseTransformer")
        
        transform_type = transformer_class.TRANSFORM_TYPE
        if transform_type is None:
            raise ValueError(f"{transformer_class} must define TRANSFORM_TYPE")
        
        self._transformers[transform_type] = transformer_class
    
    def get(self, transform_type: str) -> Optional[Type[BaseTransformer]]:
        """Get transformer class by type"""
        return self._transformers.get(transform_type)
    
    def create(self, transform_type: str, params: Dict = None) -> BaseTransformer:
        """Create transformer instance"""
        transformer_class = self.get(transform_type)
        if transformer_class is None:
            raise TransformError(
                f"Unknown transform type: {transform_type}",
                suggestion=f"Available transforms: {', '.join(self.list())}"
            )
        
        return transformer_class(params or {})
    
    def list(self) -> List[str]:
        """List all registered transform types"""
        return sorted(self._transformers.keys())
    
    def get_definition(self, transform_type: str) -> Optional[Dict]:
        """Get transformer definition for API discovery"""
        transformer_class = self.get(transform_type)
        if transformer_class is None:
            return None
        
        return {
            "input_types": transformer_class.SUPPORTED_INPUT_TYPES,
            "output_type": transformer_class.OUTPUT_TYPE,
            "description": transformer_class.__doc__ or "No description available"
        }
    
    def get_all_definitions(self) -> Dict[str, Dict]:
        """Get all transformer definitions"""
        return {
            transform_type: self.get_definition(transform_type)
            for transform_type in self.list()
        }
    
    def suggest_transforms(self, series: pd.Series, column_name: str) -> List[Dict]:
        """Suggest applicable transforms for a series"""
        suggestions = []
        
        for transform_type, transformer_class in self._transformers.items():
            # Create instance to check compatibility
            try:
                transformer = transformer_class()
                if transformer.can_transform(series):
                    # Calculate usefulness score based on data characteristics
                    score = self._calculate_usefulness(series, transform_type)
                    
                    if score > 0.3:  # Only suggest if reasonably useful
                        suggestions.append({
                            "transform": transform_type,
                            "usefulness_score": score,
                            "reason": self._get_suggestion_reason(series, transform_type),
                            "preview": self._generate_preview(series, transformer)
                        })
            except Exception:
                continue
        
        # Sort by usefulness score
        suggestions.sort(key=lambda x: x["usefulness_score"], reverse=True)
        return suggestions[:5]  # Top 5 suggestions
    
    def _calculate_usefulness(self, series: pd.Series, transform_type: str) -> float:
        """Calculate how useful a transform would be for this series"""
        score = 0.5  # Base score
        
        # DateTime transforms are highly useful for date columns
        if transform_type in ["month", "year", "quarter", "weekday"]:
            if pd.api.types.is_datetime64_any_dtype(series):
                score = 0.9
                # Higher score for high cardinality dates
                if series.nunique() > 100:
                    score = 0.95
        
        # Bucketing is useful for high cardinality numeric columns
        elif transform_type in ["bucket", "percentile_bucket"]:
            if pd.api.types.is_numeric_dtype(series):
                unique_ratio = series.nunique() / len(series)
                if unique_ratio > 0.5:
                    score = 0.85
        
        # Text cleanup is useful for string columns with whitespace
        elif transform_type in ["trim", "lowercase"]:
            if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
                # Check if there's whitespace or mixed case
                sample = series.dropna().head(100).astype(str)
                if any(s != s.strip() for s in sample) or any(s != s.lower() for s in sample):
                    score = 0.75
        
        # Top N is useful for high cardinality categorical
        elif transform_type == "top_n":
            unique_count = series.nunique()
            if 10 < unique_count < 100:
                score = 0.8
        
        return score
    
    def _get_suggestion_reason(self, series: pd.Series, transform_type: str) -> str:
        """Generate reason for suggestion"""
        if transform_type in ["month", "year", "quarter"]:
            return "High cardinality date - time-based grouping recommended"
        elif transform_type in ["bucket", "percentile_bucket"]:
            return "Many unique values - bucketing will improve analysis"
        elif transform_type == "top_n":
            return f"Column has {series.nunique()} unique values - consolidate to top categories"
        elif transform_type in ["trim", "lowercase"]:
            return "Clean text data for consistent analysis"
        else:
            return "Applicable transform for this data type"
    
    def _generate_preview(self, series: pd.Series, transformer: BaseTransformer) -> List:
        """Generate preview of transform output"""
        try:
            sample = series.dropna().head(5)
            result = transformer.transform(sample, series.name or "column")
            return result.tolist()
        except Exception:
            return []


# Global registry instance
registry = TransformRegistry()
