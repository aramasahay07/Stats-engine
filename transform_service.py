"""
Transform service - orchestrates data transformations
"""
from typing import List, Dict, Optional, Any, Tuple
import pandas as pd
import numpy as np
from transformers.registry import registry
from utils.type_inference import infer_column_role, suggest_column_transforms
from models import (
    TransformRequest, TransformSpec, TransformSuggestion, 
    SuggestTransformsResponse, TransformMetadata
)


class TransformService:
    """Handles all data transformation operations"""
    
    def __init__(self):
        self.registry = registry
    
    def get_available_transforms(self, column_type: str) -> List[str]:
        """Get all transforms available for a column type"""
        return self.registry.get_transforms_for_type(column_type)
    
    def get_all_transforms(self) -> Dict[str, Any]:
        """Get comprehensive transform catalog"""
        return self.registry.get_all_transforms()
    
    def suggest_transforms(self, df: pd.DataFrame, column: str, limit: int = 5) -> SuggestTransformsResponse:
        """Suggest useful transforms for a column"""
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        series = df[column]
        detected_type = infer_column_role(series)
        
        # Get suggestions from utility
        suggestions = suggest_column_transforms(series, column)
        
        # Convert to response format
        transform_suggestions = []
        for sugg in suggestions[:limit]:
            # Generate preview
            try:
                preview = self._generate_preview(series, sugg['transform'], sugg.get('params', {}))
                transform_suggestions.append(
                    TransformSuggestion(
                        transform=sugg['transform'],
                        usefulness_score=sugg['score'],
                        reason=sugg['reason'],
                        preview=preview,
                        params=sugg.get('params', {})
                    )
                )
            except Exception:
                continue
        
        return SuggestTransformsResponse(
            column=column,
            detected_type=detected_type,
            suggested_transforms=transform_suggestions
        )
    
    def apply_transform(
        self, 
        df: pd.DataFrame, 
        column: str, 
        transform_type: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> Tuple[pd.Series, TransformMetadata]:
        """Apply a single transform to a column"""
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        params = params or {}
        series = df[column]
        
        # Get transformer
        transformer = self.registry.get_transformer(transform_type, params)
        
        # Apply transform
        result = transformer.transform(series, column)
        
        # Generate metadata
        metadata = TransformMetadata(
            source_column=column,
            transform_type=transform_type,
            null_count=int(result.isna().sum()),
            unique_values=int(result.nunique()),
            sample_output=result.dropna().head(5).tolist() if len(result) > 0 else []
        )
        
        return result, metadata
    
    def apply_transform_chain(
        self, 
        df: pd.DataFrame, 
        request: TransformRequest
    ) -> Tuple[pd.Series, List[TransformMetadata]]:
        """Apply a chain of transforms to a column"""
        if request.column not in df.columns:
            raise ValueError(f"Column '{request.column}' not found")
        
        series = df[request.column]
        metadata_list = []
        
        for spec in request.transforms:
            transformer = self.registry.get_transformer(spec.type, spec.params or {})
            series = transformer.transform(series, request.column)
            
            metadata = TransformMetadata(
                source_column=request.column,
                transform_type=spec.type,
                null_count=int(series.isna().sum()),
                unique_values=int(series.nunique()),
                sample_output=series.dropna().head(5).tolist() if len(series) > 0 else []
            )
            metadata_list.append(metadata)
        
        return series, metadata_list
    
    def preview_transform(
        self,
        df: pd.DataFrame,
        column: str,
        transform_type: str,
        params: Optional[Dict[str, Any]] = None,
        n_rows: int = 100
    ) -> Dict[str, Any]:
        """Preview transform results without applying"""
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        params = params or {}
        series = df[column].head(n_rows)
        
        transformer = self.registry.get_transformer(transform_type, params)
        result = transformer.transform(series, column)
        
        return {
            "original": series.tolist(),
            "transformed": result.tolist(),
            "null_count": int(result.isna().sum()),
            "unique_values": int(result.nunique()),
            "sample_values": result.dropna().unique()[:10].tolist()
        }
    
    def _generate_preview(self, series: pd.Series, transform_type: str, params: Dict) -> List[Any]:
        """Generate preview for suggestion"""
        try:
            transformer = self.registry.get_transformer(transform_type, params)
            result = transformer.transform(series.head(5), series.name)
            return result.tolist()
        except Exception:
            return []
    
    def validate_transform(self, column_type: str, transform_type: str) -> bool:
        """Check if transform is valid for column type"""
        available = self.registry.get_transforms_for_type(column_type)
        return transform_type in available
    
    def group_by(
        self,
        df: pd.DataFrame,
        group_by_cols: List[str],
        agg_specs: Dict[str, str]
    ) -> pd.DataFrame:
        """Perform group by aggregation
        
        Args:
            df: Source dataframe
            group_by_cols: Columns to group by
            agg_specs: {"new_col_name": "source_col:agg_function"}
                      e.g., {"total_sales": "amount:sum", "avg_price": "price:mean"}
        """
        # Parse aggregation specs
        agg_dict = {}
        col_renames = {}
        
        for new_name, spec in agg_specs.items():
            if ':' in spec:
                col, func = spec.split(':', 1)
                if col not in agg_dict:
                    agg_dict[col] = []
                agg_dict[col].append(func)
                col_renames[f"{col}_{func}"] = new_name
            else:
                # Default to first column with specified function
                col = df.columns[0]
                agg_dict[col] = [spec]
                col_renames[f"{col}_{spec}"] = new_name
        
        # Perform groupby
        result = df.groupby(group_by_cols).agg(agg_dict)
        
        # Flatten column names
        result.columns = ['_'.join(col).strip() for col in result.columns.values]
        
        # Rename to user-specified names
        result = result.rename(columns=col_renames)
        
        return result.reset_index()
    
    def pivot_table(
        self,
        df: pd.DataFrame,
        index: List[str],
        columns: str,
        values: str,
        aggfunc: str = 'sum'
    ) -> pd.DataFrame:
        """Create pivot table"""
        result = pd.pivot_table(
            df,
            index=index,
            columns=columns,
            values=values,
            aggfunc=aggfunc,
            fill_value=0
        )
        return result.reset_index()
    
    def unpivot(
        self,
        df: pd.DataFrame,
        id_vars: List[str],
        value_vars: Optional[List[str]] = None,
        var_name: str = 'variable',
        value_name: str = 'value'
    ) -> pd.DataFrame:
        """Unpivot/melt dataframe"""
        return pd.melt(
            df,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name=var_name,
            value_name=value_name
        )
    
    def merge_tables(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        on: Optional[List[str]] = None,
        left_on: Optional[List[str]] = None,
        right_on: Optional[List[str]] = None,
        how: str = 'inner'
    ) -> pd.DataFrame:
        """Merge two dataframes"""
        return pd.merge(
            left_df,
            right_df,
            on=on,
            left_on=left_on,
            right_on=right_on,
            how=how
        )
    
    def remove_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = 'first'
    ) -> pd.DataFrame:
        """Remove duplicate rows"""
        return df.drop_duplicates(subset=subset, keep=keep)
    
    def fill_missing(
        self,
        df: pd.DataFrame,
        column: str,
        method: str = 'ffill',
        value: Optional[Any] = None
    ) -> pd.Series:
        """Fill missing values in a column"""
        series = df[column]
        
        if value is not None:
            return series.fillna(value)
        elif method == 'ffill':
            return series.fillna(method='ffill')
        elif method == 'bfill':
            return series.fillna(method='bfill')
        elif method == 'mean':
            return series.fillna(series.mean())
        elif method == 'median':
            return series.fillna(series.median())
        elif method == 'mode':
            return series.fillna(series.mode()[0] if len(series.mode()) > 0 else None)
        else:
            return series
    
    def filter_rows(
        self,
        df: pd.DataFrame,
        column: str,
        operator: str,
        value: Any
    ) -> pd.DataFrame:
        """Filter dataframe rows"""
        series = df[column]
        
        if operator == 'eq':
            mask = series == value
        elif operator == 'ne':
            mask = series != value
        elif operator == 'gt':
            mask = series > value
        elif operator == 'lt':
            mask = series < value
        elif operator == 'gte':
            mask = series >= value
        elif operator == 'lte':
            mask = series <= value
        elif operator == 'in':
            mask = series.isin(value if isinstance(value, list) else [value])
        elif operator == 'not_in':
            mask = ~series.isin(value if isinstance(value, list) else [value])
        elif operator == 'contains':
            mask = series.astype(str).str.contains(str(value), na=False)
        elif operator == 'starts_with':
            mask = series.astype(str).str.startswith(str(value), na=False)
        else:
            raise ValueError(f"Unknown operator: {operator}")
        
        return df[mask]
    
    def create_virtual_column(
        self,
        df: pd.DataFrame,
        name: str,
        expression: str,
        column_type: str = "numeric"
    ) -> pd.Series:
        """Create a computed column using expression
        
        Args:
            df: Source dataframe
            name: New column name
            expression: Python expression (can reference column names)
            column_type: Expected output type
        
        Example:
            expression = "df['price'] * df['quantity']"
        """
        # Safety: Only allow DataFrame column access
        local_vars = {'df': df, 'pd': pd, 'np': np}
        
        try:
            result = eval(expression, {"__builtins__": {}}, local_vars)
            if isinstance(result, pd.Series):
                return result
            else:
                return pd.Series(result, index=df.index)
        except Exception as e:
            raise ValueError(f"Invalid expression: {e}")
