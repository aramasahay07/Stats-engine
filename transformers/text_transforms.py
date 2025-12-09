"""
Text transformations
"""
import pandas as pd
import re
from .base import BaseTransformer, TransformError


class LowercaseTransformer(BaseTransformer):
    TRANSFORM_TYPE = "lowercase"
    SUPPORTED_INPUT_TYPES = ["string", "text"]
    OUTPUT_TYPE = "string"
    
    def validate_params(self):
        pass
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        return series.astype(str).str.lower()


class UppercaseTransformer(BaseTransformer):
    TRANSFORM_TYPE = "uppercase"
    SUPPORTED_INPUT_TYPES = ["string", "text"]
    OUTPUT_TYPE = "string"
    
    def validate_params(self):
        pass
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        return series.astype(str).str.upper()


class TrimTransformer(BaseTransformer):
    TRANSFORM_TYPE = "trim"
    SUPPORTED_INPUT_TYPES = ["string", "text"]
    OUTPUT_TYPE = "string"
    
    def validate_params(self):
        pass
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        return series.astype(str).str.strip()


class LengthTransformer(BaseTransformer):
    TRANSFORM_TYPE = "length"
    SUPPORTED_INPUT_TYPES = ["string", "text"]
    OUTPUT_TYPE = "numeric"
    
    def validate_params(self):
        pass
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        return series.astype(str).str.len()


class ExtractTransformer(BaseTransformer):
    TRANSFORM_TYPE = "extract"
    SUPPORTED_INPUT_TYPES = ["string", "text"]
    OUTPUT_TYPE = "string"
    
    def validate_params(self):
        if "pattern" not in self.params:
            raise TransformError("'pattern' parameter is required for extract transform")
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        pattern = self.params.get("pattern")
        group = self.params.get("group", 0)
        
        try:
            result = series.astype(str).str.extract(pattern, expand=False)
            if isinstance(result, pd.DataFrame):
                result = result.iloc[:, group]
            return result
        except Exception as e:
            raise TransformError(f"Regex extraction failed: {str(e)}", column=column_name)


class ReplaceTransformer(BaseTransformer):
    TRANSFORM_TYPE = "replace"
    SUPPORTED_INPUT_TYPES = ["string", "text"]
    OUTPUT_TYPE = "string"
    
    def validate_params(self):
        if "find" not in self.params or "replace" not in self.params:
            raise TransformError("'find' and 'replace' parameters are required")
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        find = self.params.get("find")
        replace = self.params.get("replace")
        regex = self.params.get("regex", False)
        
        return series.astype(str).str.replace(find, replace, regex=regex)


class FirstWordTransformer(BaseTransformer):
    TRANSFORM_TYPE = "first_word"
    SUPPORTED_INPUT_TYPES = ["string", "text"]
    OUTPUT_TYPE = "string"
    
    def validate_params(self):
        pass
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        return series.astype(str).str.split().str[0]


class WordCountTransformer(BaseTransformer):
    TRANSFORM_TYPE = "word_count"
    SUPPORTED_INPUT_TYPES = ["string", "text"]
    OUTPUT_TYPE = "numeric"
    
    def validate_params(self):
        pass
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        return series.astype(str).str.split().str.len()


class ContainsTransformer(BaseTransformer):
    TRANSFORM_TYPE = "contains"
    SUPPORTED_INPUT_TYPES = ["string", "text"]
    OUTPUT_TYPE = "categorical"
    
    def validate_params(self):
        if "substring" not in self.params:
            raise TransformError("'substring' parameter is required")
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        substring = self.params.get("substring")
        case_sensitive = self.params.get("case_sensitive", False)
        
        return series.astype(str).str.contains(substring, case=case_sensitive, na=False)


class TitleCaseTransformer(BaseTransformer):
    TRANSFORM_TYPE = "title_case"
    SUPPORTED_INPUT_TYPES = ["string", "text"]
    OUTPUT_TYPE = "string"
    
    def validate_params(self):
        pass
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        return series.astype(str).str.title()


class ExtractEntityTransformer(BaseTransformer):
    """Extract common entities like email, phone, URL"""
    TRANSFORM_TYPE = "extract_entity"
    SUPPORTED_INPUT_TYPES = ["string", "text"]
    OUTPUT_TYPE = "string"
    
    PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "url": r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        "zip": r'\b\d{5}(?:-\d{4})?\b',
        "currency": r'\$\s?\d+(?:,\d{3})*(?:\.\d{2})?'
    }
    
    def validate_params(self):
        entity_type = self.params.get("entity_type")
        if entity_type not in self.PATTERNS:
            raise TransformError(
                f"entity_type must be one of: {list(self.PATTERNS.keys())}"
            )
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        entity_type = self.params.get("entity_type")
        pattern = self.PATTERNS[entity_type]
        
        return series.astype(str).str.extract(f'({pattern})', expand=False)


class StandardizeTransformer(BaseTransformer):
    """Standardize messy categorical values"""
    TRANSFORM_TYPE = "standardize"
    SUPPORTED_INPUT_TYPES = ["string", "text"]
    OUTPUT_TYPE = "string"
    
    # Common standardizations
    BOOLEAN_MAPS = {
        'yes': ['yes', 'y', 'true', 't', '1', 'ok'],
        'no': ['no', 'n', 'false', 'f', '0']
    }
    
    def validate_params(self):
        pass
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        method = self.params.get("method", "boolean")
        
        if method == "boolean":
            # Standardize boolean-like values
            result = series.astype(str).str.lower().str.strip()
            
            for standard, variants in self.BOOLEAN_MAPS.items():
                result = result.replace(variants, standard)
            
            return result
        
        elif method == "trim_lower":
            # Simple cleanup
            return series.astype(str).str.strip().str.lower()
        
        return series


class SlugifyTransformer(BaseTransformer):
    """Convert text to URL-friendly slug"""
    TRANSFORM_TYPE = "slugify"
    SUPPORTED_INPUT_TYPES = ["string", "text"]
    OUTPUT_TYPE = "string"
    
    def validate_params(self):
        pass
    
    def transform(self, series: pd.Series, column_name: str) -> pd.Series:
        result = series.astype(str).str.lower()
        result = result.str.replace(r'[^\w\s-]', '', regex=True)
        result = result.str.replace(r'[-\s]+', '-', regex=True)
        return result.str.strip('-')
