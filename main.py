"""
AI Data Lab Stats Engine with Advanced Data Transformation
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List
from uuid import uuid4
from io import BytesIO
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Import our modules
from models import *
from session_store import store
from transformers.registry import registry
from transformers.base import TransformPipeline, TransformError, create_transform_key
from utils.type_inference import (
    infer_column_role,
    smart_parse_datetime,
    analyze_column_patterns,
    suggest_column_transforms
)

# ---------------------------------------------------
# FastAPI app setup
# ---------------------------------------------------
app = FastAPI(
    title="AI Data Lab Stats Engine",
    version="2.0.0",
    description="Advanced statistical analysis with intelligent data transformations"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------
# Utility functions
# ---------------------------------------------------
def _load_dataframe(file: UploadFile) -> pd.DataFrame:
    """Read CSV or Excel file into a pandas DataFrame."""
    content = file.file.read()
    file.file.close()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    buffer = BytesIO(content)
    filename = file.filename.lower()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(buffer)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(buffer)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Please upload CSV or Excel.",
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="File contained no rows.")
    
    # Smart datetime parsing for columns that look like dates
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to parse as datetime
            try:
                parsed = smart_parse_datetime(df[col])
                if parsed.notna().sum() / len(parsed) > 0.5:  # 50% success rate
                    df[col] = parsed
            except Exception:
                pass
    
    return df


def _build_profile(df: pd.DataFrame, session_id: str) -> ProfileResponse:
    """Create profile summary for the UI."""
    cols: List[ColumnInfo] = []
    descriptives: List[DescriptiveStats] = []

    for col in df.columns:
        s = df[col]
        role = infer_column_role(s)
        missing_pct = float(s.isna().mean() * 100)
        
        cols.append(
            ColumnInfo(
                name=col,
                dtype=str(s.dtype),
                role=role,
                missing_pct=round(missing_pct, 2),
                unique_count=int(s.nunique()),
                sample_values=s.dropna().head(3).tolist() if len(s) > 0 else []
            )
        )

        if role == "numeric":
            desc = s.describe(percentiles=[0.25, 0.5, 0.75])
            descriptives.append(
                DescriptiveStats(
                    column=col,
                    count=int(desc.get("count", 0)),
                    mean=float(desc.get("mean", np.nan))
                    if not np.isnan(desc.get("mean", np.nan))
                    else None,
                    std=float(desc.get("std", np.nan))
                    if not np.isnan(desc.get("std", np.nan))
                    else None,
                    min=float(desc.get("min", np.nan))
                    if not np.isnan(desc.get("min", np.nan))
                    else None,
                    q25=float(desc.get("25%", np.nan))
                    if not np.isnan(desc.get("25%", np.nan))
                    else None,
                    median=float(desc.get("50%", np.nan))
                    if not np.isnan(desc.get("50%", np.nan))
                    else None,
                    q75=float(desc.get("75%", np.nan))
                    if not np.isnan(desc.get("75%", np.nan))
                    else None,
                    max=float(desc.get("max", np.nan))
                    if not np.isnan(desc.get("max", np.nan))
                    else None,
                )
            )

    return ProfileResponse(
        session_id=session_id,
        n_rows=int(len(df)),
        n_cols=int(df.shape[1]),
        columns=cols,
        schema=cols,
        descriptives=descriptives,
    )


def _build_correlation(df: pd.DataFrame) -> Optional[CorrelationResponse]:
    """Build correlation matrix for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=0, how="any")
    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr()
    corr_dict: Dict[str, Dict[str, float]] = {}
    for col in corr.columns:
        corr_dict[col] = {idx: float(val) for idx, val in corr[col].items()}
    return CorrelationResponse(matrix=corr_dict)


def _auto_tests(df: pd.DataFrame, profile: ProfileResponse) -> List[TestResult]:
    """Run simple t-test or ANOVA if data supports it."""
    tests: List[TestResult] = []

    numeric_cols = [c.name for c in profile.columns if c.role == "numeric"]
    cat_cols = [c.name for c in profile.columns if c.role == "categorical"]

    if not numeric_cols or not cat_cols:
        return tests

    target = numeric_cols[0]
    group_col = cat_cols[0]
    s_target = df[target]
    s_group = df[group_col].astype("category")

    data = pd.DataFrame({"target": s_target, "group": s_group}).dropna()
    if data["group"].nunique() < 2:
        return tests

    groups = [g["target"].values for _, g in data.groupby("group")]

    try:
        if len(groups) == 2:
            stat, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
            interpretation = (
                "Difference between the two groups is statistically significant."
                if p < 0.05
                else "No statistically significant difference between the two groups."
            )
            tests.append(
                TestResult(
                    test_type="t-test",
                    target=target,
                    group_col=group_col,
                    p_value=float(p),
                    statistic=float(stat),
                    df=None,
                    interpretation=interpretation,
                )
            )
        elif len(groups) > 2:
            stat, p = stats.f_oneway(*groups)
            interpretation = (
                "At least one group mean differs significantly from the others."
                if p < 0.05
                else "No statistically significant difference among group means."
            )
            tests.append(
                TestResult(
                    test_type="anova",
                    target=target,
                    group_col=group_col,
                    p_value=float(p),
                    statistic=float(stat),
                    df=None,
                    interpretation=interpretation,
                )
            )
    except Exception:
        return tests

    return tests


def _auto_regression(df: pd.DataFrame, profile: ProfileResponse) -> Optional[RegressionResult]:
    """Fit a simple OLS regression: first numeric as y, others as X."""
    numeric_cols = [c.name for c in profile.columns if c.role == "numeric"]
    if len(numeric_cols) < 2:
        return None

    target = numeric_cols[0]
    predictors = numeric_cols[1:]

    data = df[numeric_cols].dropna()
    if len(data) < 10:
        return None

    y = data[target]
    X = data[predictors]
    X = sm.add_constant(X)

    try:
        model = sm.OLS(y, X).fit()
    except Exception:
        return None

    coeffs = {name: float(val) for name, val in model.params.items()}
    return RegressionResult(
        target=target,
        predictors=predictors,
        r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj),
        coefficients=coeffs,
    )


def _apply_transforms(df: pd.DataFrame, session_id: str, 
                     transform_specs: Dict[str, TransformSpec]) -> tuple[pd.DataFrame, Dict]:
    """Apply transformations to dataframe"""
    result_df = df.copy()
    metadata = {}
    
    for col_name, spec in transform_specs.items():
        if col_name not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{col_name}' not found in dataset"
            )
        
        try:
            # Create transformer
            transformer = registry.create(spec.type, spec.params)
            
            # Check cache
            cache_key = create_transform_key(spec.type, spec.params)
            cached = store.get_cached_transform(session_id, col_name, cache_key)
            
            if cached is not None:
                transformed = cached
            else:
                # Apply transform
                transformed = transformer.transform(df[col_name], col_name)
                
                # Cache result
                store.cache_transform(session_id, col_name, cache_key, transformed)
            
            # Add to result dataframe
            output_col_name = transformer.get_output_column_name(col_name)
            result_df[output_col_name] = transformed
            
            # Collect metadata
            metadata[output_col_name] = transformer.get_metadata(df[col_name], transformed)
            
        except TransformError as e:
            raise HTTPException(status_code=400, detail={
                "error": str(e),
                "column": col_name,
                "suggestion": e.suggestion
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transform failed: {str(e)}")
    
    return result_df, metadata


# ---------------------------------------------------
# API endpoints
# ---------------------------------------------------
@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "version": "2.0.0"}


@app.post("/upload", response_model=ProfileResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a CSV/Excel file, store a cleaned DataFrame in memory,
    and return a profile summary + session_id.
    """
    df = _load_dataframe(file)
    df = df.dropna(axis=1, how="all")

    session_id = str(uuid4())
    store.set(session_id, df)

    profile = _build_profile(df, session_id)
    return profile


@app.get("/analysis/{session_id}", response_model=AnalysisResponse)
async def run_analysis(session_id: str):
    """
    Run correlation, simple group tests, and linear regression
    for the given session_id.
    """
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")

    profile = _build_profile(df, session_id)
    correlation = _build_correlation(df)
    tests = _auto_tests(df, profile)
    regression = _auto_regression(df, profile)

    return AnalysisResponse(
        session_id=session_id,
        correlation=correlation,
        tests=tests,
        regression=regression,
    )


@app.post("/query/{session_id}", response_model=QueryResponse)
async def run_query(session_id: str, query: QueryRequest):
    """
    Execute a query with transformations, aggregations, and filtering
    """
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Apply transforms if specified
        if query.transforms:
            df, transform_metadata = _apply_transforms(df, session_id, query.transforms)
        else:
            transform_metadata = {}
        
        # Apply filters
        if query.filters:
            for f in query.filters:
                if f.column not in df.columns:
                    raise HTTPException(400, f"Column {f.column} not found")
                
                col = df[f.column]
                if f.operator == "eq":
                    df = df[col == f.value]
                elif f.operator == "ne":
                    df = df[col != f.value]
                elif f.operator == "gt":
                    df = df[col > f.value]
                elif f.operator == "lt":
                    df = df[col < f.value]
                elif f.operator == "gte":
                    df = df[col >= f.value]
                elif f.operator == "lte":
                    df = df[col <= f.value]
                elif f.operator == "in":
                    df = df[col.isin(f.value)]
                elif f.operator == "not_in":
                    df = df[~col.isin(f.value)]
                elif f.operator == "contains":
                    df = df[col.astype(str).str.contains(str(f.value), na=False)]
        
        # Execute operation
        if query.operation == "aggregate":
            if query.group_by:
                grouped = df.groupby(query.group_by)
                result = {}
                
                if query.aggregations:
                    for alias, spec in query.aggregations.items():
                        col, func = spec.split(":")
                        if col == "*":
                            result[alias] = grouped.size()
                        else:
                            result[alias] = getattr(grouped[col], func)()
                
                result_df = pd.DataFrame(result).reset_index()
            else:
                result_df = df
        
        elif query.operation == "describe":
            result_df = df.describe()
        
        elif query.operation == "distinct":
            if query.group_by:
                result_df = df[query.group_by].drop_duplicates()
            else:
                result_df = df
        
        else:
            result_df = df
        
        # Apply sorting
        if query.sort:
            if query.sort.column in result_df.columns:
                if query.sort.order == "chronological":
                    # For date-derived columns, sort chronologically
                    result_df = result_df.sort_values(query.sort.column)
                elif query.sort.order == "alphabetical":
                    result_df = result_df.sort_values(query.sort.column)
                elif query.sort.order == "value_desc":
                    result_df = result_df.sort_values(query.sort.column, ascending=False)
                elif query.sort.order == "value_asc":
                    result_df = result_df.sort_values(query.sort.column, ascending=True)
        
        # Apply limit
        if query.limit:
            result_df = result_df.head(query.limit)
        
        # Convert to dict
        data = result_df.to_dict(orient='records')
        
        return QueryResponse(
            success=True,
            result={
                "data": data,
                "row_count": len(data),
                "columns": list(result_df.columns)
            },
            transforms_applied=transform_metadata
        )
    
    except Exception as e:
        return QueryResponse(
            success=False,
            error={
                "code": "QUERY_ERROR",
                "message": str(e)
            }
        )


@app.get("/transforms", response_model=TransformDiscoveryResponse)
async def get_transforms():
    """
    Get all available transforms and their specifications
    """
    definitions = registry.get_all_definitions()
    
    # Convert to proper response format
    transforms = {}
    for name, defn in definitions.items():
        transforms[name] = TransformDefinition(
            input_types=defn["input_types"],
            output_type=defn["output_type"],
            params={},  # Would need to be populated from transformer classes
            description=defn["description"]
        )
    
    return TransformDiscoveryResponse(transforms=transforms)


@app.get("/suggest-transforms/{session_id}", response_model=SuggestTransformsResponse)
async def suggest_transforms(session_id: str, column: str):
    """
    Suggest appropriate transforms for a specific column
    """
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if column not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found")
    
    series = df[column]
    role = infer_column_role(series)
    suggestions = registry.suggest_transforms(series, column)
    
    return SuggestTransformsResponse(
        column=column,
        detected_type=role,
        suggested_transforms=[
            TransformSuggestion(**s) for s in suggestions
        ]
    )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its cached transforms"""
    if not store.exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    store.delete(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.get("/sessions/stats")
async def get_session_stats():
    """Get statistics about active sessions"""
    stats = store.get_stats()
    return stats


@app.post("/sessions/cleanup")
async def cleanup_sessions():
    """Cleanup expired sessions"""
    count = store.cleanup_expired()
    return {"cleaned_up": count}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
