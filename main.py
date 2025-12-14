from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional, Any
from uuid import uuid4
from io import BytesIO
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

from models import *
from session_store import SessionStore
from transform_service import TransformService
from utils.type_inference import infer_column_role

# ---------------------------------------------------
# FastAPI app setup
# ---------------------------------------------------
app = FastAPI(
    title="AI Data Lab Stats & Transform Engine",
    version="2.0.0",
    description="Advanced data transformation engine with 60+ intelligent transforms + statistical analysis"
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
# Services
# ---------------------------------------------------
session_store = SessionStore()
transform_service = TransformService()


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
                sample_values=s.dropna().head(3).tolist() if len(s.dropna()) > 0 else []
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

    # Generate sample rows for frontend data preview (first 100 rows)
    sample_rows = df.head(100).replace({np.nan: None}).to_dict(orient='records')

    return ProfileResponse(
        session_id=session_id,
        n_rows=int(len(df)),
        n_cols=int(df.shape[1]),
        columns=cols,
        schema=cols,
        descriptives=descriptives,
        sample_rows=sample_rows,
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


# ---------------------------------------------------
# Core API endpoints
# ---------------------------------------------------
@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {
        "status": "ok",
        "version": "2.0.0",
        "features": ["transforms", "stats", "query", "table_ops"]
    }


@app.get("/stats")
def get_stats():
    """Get storage and system statistics"""
    return session_store.get_stats()


@app.post("/upload", response_model=ProfileResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a CSV/Excel file, store as STAGING data,
    and return a profile summary + session_id.
    """
    df = _load_dataframe(file)

    # Basic cleaning: drop columns that are all missing
    df = df.dropna(axis=1, how="all")

    session_id = str(uuid4())
    session_store.set(session_id, df, metadata={"filename": file.filename})

    profile = _build_profile(df, session_id)
    return profile


@app.get("/session/{session_id}/profile", response_model=ProfileResponse)
def get_session_profile(session_id: str):
    """Get current profile of session data"""
    df = session_store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return _build_profile(df, session_id)


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Delete a session and free memory"""
    if not session_store.exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_store.delete(session_id)
    return {"message": "Session deleted successfully"}


# ---------------------------------------------------
# Transform Discovery Endpoints
# ---------------------------------------------------
@app.get("/transforms", response_model=TransformDiscoveryResponse)
def get_all_transforms():
    """Get complete catalog of available transforms"""
    all_transforms = transform_service.get_all_transforms()
    
    # Convert to API format
    transforms_dict = {}
    for name, info in all_transforms.items():
        transforms_dict[name] = TransformDefinition(
            input_types=info.get("input_types", []),
            output_type=info.get("output_type", "unknown"),
            params=info.get("params", {}),
            description=info.get("description", ""),
            examples=info.get("examples", [])
        )
    
    return TransformDiscoveryResponse(transforms=transforms_dict)


@app.get("/transforms/for/{column_type}")
def get_transforms_for_type(column_type: str):
    """Get available transforms for a specific column type"""
    transforms = transform_service.get_available_transforms(column_type)
    return {"column_type": column_type, "transforms": transforms}


@app.post("/session/{session_id}/suggest/{column}", response_model=SuggestTransformsResponse)
def suggest_transforms(session_id: str, column: str, limit: int = 5):
    """Get AI-powered transform suggestions for a column"""
    df = session_store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        suggestions = transform_service.suggest_transforms(df, column, limit)
        return suggestions
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------
# Transform Application Endpoints
# ---------------------------------------------------
@app.post("/session/{session_id}/transform/preview")
def preview_transform(
    session_id: str,
    column: str,
    transform_type: str,
    params: Optional[Dict[str, Any]] = None,
    n_rows: int = 100
):
    """Preview transform without applying it"""
    df = session_store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        preview = transform_service.preview_transform(df, column, transform_type, params, n_rows)
        return preview
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/session/{session_id}/transform/apply")
def apply_transform(
    session_id: str,
    request: TransformRequest,
    new_column_name: Optional[str] = None,
    replace_original: bool = False
):
    """Apply transform(s) to create a new column or replace existing"""
    df = session_store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Apply transform chain
        result_series, metadata_list = transform_service.apply_transform_chain(df, request)
        
        # Determine column name
        if replace_original:
            col_name = request.column
        elif new_column_name:
            col_name = new_column_name
        else:
            # Auto-generate name
            transform_names = "_".join([spec.type for spec in request.transforms])
            col_name = f"{request.column}_{transform_names}"
        
        # Add to dataframe
        df[col_name] = result_series
        session_store.set(session_id, df)
        
        return {
            "success": True,
            "column_created": col_name,
            "metadata": metadata_list,
            "n_rows": len(df)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/session/{session_id}/transform/batch")
def apply_batch_transforms(
    session_id: str,
    transforms: Dict[str, TransformRequest]
):
    """Apply multiple transforms at once
    
    transforms: {"new_col_name": TransformRequest, ...}
    """
    df = session_store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    results = {}
    errors = {}
    
    for new_col_name, request in transforms.items():
        try:
            result_series, metadata_list = transform_service.apply_transform_chain(df, request)
            df[new_col_name] = result_series
            results[new_col_name] = {"success": True, "metadata": metadata_list}
        except Exception as e:
            errors[new_col_name] = str(e)
    
    if results:
        session_store.set(session_id, df)
    
    return {
        "success": len(errors) == 0,
        "results": results,
        "errors": errors if errors else None
    }


# ---------------------------------------------------
# Table Operations (Power Query-level)
# ---------------------------------------------------
@app.post("/session/{session_id}/group_by")
def group_by_aggregate(
    session_id: str,
    group_by: List[str],
    aggregations: Dict[str, str],
    create_new_session: bool = True
):
    """
    Group by and aggregate
    
    aggregations: {"new_col": "source_col:function"}
    Example: {"total_sales": "amount:sum", "avg_price": "price:mean"}
    """
    df = session_store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        result = transform_service.group_by(df, group_by, aggregations)
        
        if create_new_session:
            new_session_id = str(uuid4())
            session_store.set(new_session_id, result, metadata={"parent_session": session_id})
            return {
                "success": True,
                "session_id": new_session_id,
                "n_rows": len(result),
                "n_cols": len(result.columns)
            }
        else:
            session_store.set(session_id, result)
            return {
                "success": True,
                "session_id": session_id,
                "n_rows": len(result),
                "n_cols": len(result.columns)
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/session/{session_id}/pivot")
def create_pivot_table(
    session_id: str,
    index: List[str],
    columns: str,
    values: str,
    aggfunc: str = "sum"
):
    """Create pivot table"""
    df = session_store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        result = transform_service.pivot_table(df, index, columns, values, aggfunc)
        
        new_session_id = str(uuid4())
        session_store.set(new_session_id, result, metadata={"parent_session": session_id})
        
        return {
            "success": True,
            "session_id": new_session_id,
            "n_rows": len(result),
            "n_cols": len(result.columns)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/session/{session_id}/unpivot")
def unpivot_table(
    session_id: str,
    id_vars: List[str],
    value_vars: Optional[List[str]] = None,
    var_name: str = "variable",
    value_name: str = "value"
):
    """Unpivot/melt table"""
    df = session_store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        result = transform_service.unpivot(df, id_vars, value_vars, var_name, value_name)
        
        new_session_id = str(uuid4())
        session_store.set(new_session_id, result, metadata={"parent_session": session_id})
        
        return {
            "success": True,
            "session_id": new_session_id,
            "n_rows": len(result),
            "n_cols": len(result.columns)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/session/{session_id}/merge/{other_session_id}")
def merge_sessions(
    session_id: str,
    other_session_id: str,
    on: Optional[List[str]] = None,
    left_on: Optional[List[str]] = None,
    right_on: Optional[List[str]] = None,
    how: str = "inner"
):
    """Merge two datasets (like SQL JOIN)"""
    left_df = session_store.get(session_id)
    right_df = session_store.get(other_session_id)
    
    if left_df is None or right_df is None:
        raise HTTPException(status_code=404, detail="One or both sessions not found")
    
    try:
        result = transform_service.merge_tables(left_df, right_df, on, left_on, right_on, how)
        
        new_session_id = str(uuid4())
        session_store.set(new_session_id, result, metadata={
            "left_session": session_id,
            "right_session": other_session_id,
            "join_type": how
        })
        
        return {
            "success": True,
            "session_id": new_session_id,
            "n_rows": len(result),
            "n_cols": len(result.columns)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/session/{session_id}/remove_duplicates")
def remove_duplicates(
    session_id: str,
    subset: Optional[List[str]] = None,
    keep: str = "first"
):
    """Remove duplicate rows"""
    df = session_store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    original_count = len(df)
    result = transform_service.remove_duplicates(df, subset, keep)
    session_store.set(session_id, result)
    
    return {
        "success": True,
        "rows_removed": original_count - len(result),
        "n_rows": len(result)
    }


@app.post("/session/{session_id}/fill_missing")
def fill_missing_values(
    session_id: str,
    column: str,
    method: str = "ffill",
    value: Optional[Any] = None
):
    """Fill missing values in a column"""
    df = session_store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        result_series = transform_service.fill_missing(df, column, method, value)
        df[column] = result_series
        session_store.set(session_id, df)
        
        return {
            "success": True,
            "null_count": int(result_series.isna().sum())
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/session/{session_id}/filter")
def filter_rows(
    session_id: str,
    filters: List[FilterSpec],
    create_new_session: bool = False
):
    """Filter rows based on conditions"""
    df = session_store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        result = df
        for filter_spec in filters:
            result = transform_service.filter_rows(
                result,
                filter_spec.column,
                filter_spec.operator,
                filter_spec.value
            )
        
        if create_new_session:
            new_session_id = str(uuid4())
            session_store.set(new_session_id, result, metadata={"parent_session": session_id})
            return {
                "success": True,
                "session_id": new_session_id,
                "n_rows": len(result)
            }
        else:
            session_store.set(session_id, result)
            return {
                "success": True,
                "session_id": session_id,
                "n_rows": len(result)
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------
# Statistical Analysis Endpoints
# ---------------------------------------------------
@app.get("/session/{session_id}/analysis", response_model=AnalysisResponse)
async def run_analysis(session_id: str):
    """
    Run correlation, simple group tests, and linear regression
    for the given session_id.
    """
    df = session_store.get(session_id)
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


@app.get("/session/{session_id}/correlation", response_model=CorrelationResponse)
def get_correlation(session_id: str):
    """Get correlation matrix for numeric columns"""
    df = session_store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    correlation = _build_correlation(df)
    if correlation is None:
        raise HTTPException(status_code=400, detail="Not enough numeric columns for correlation")
    
    return correlation


# ---------------------------------------------------
# Advanced Query Endpoint
# ---------------------------------------------------
@app.post("/session/{session_id}/query", response_model=QueryResponse)
def execute_query(session_id: str, request: QueryRequest):
    """
    Execute complex query with transforms, aggregations, filters
    
    Supports:
    - Virtual columns (computed fields)
    - Inline transforms
    - Grouping and aggregation
    - Filtering
    - Sorting
    """
    df = session_store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        result = df.copy()
        transforms_applied = {}
        
        # 1. Apply virtual columns
        if request.virtual_columns:
            for col_name, spec in request.virtual_columns.items():
                # This is a simplified version - expand based on your needs
                if spec.type == "expression":
                    expression = spec.params.get("expression", "")
                    result[col_name] = transform_service.create_virtual_column(
                        result, col_name, expression
                    )
        
        # 2. Apply inline transforms
        if request.transforms:
            for col_name, transform_spec in request.transforms.items():
                series, metadata = transform_service.apply_transform(
                    result, col_name, transform_spec.type, transform_spec.params
                )
                result[col_name + "_transformed"] = series
                transforms_applied[col_name] = metadata
        
        # 3. Apply filters
        if request.filters:
            for filter_spec in request.filters:
                result = transform_service.filter_rows(
                    result,
                    filter_spec.column,
                    filter_spec.operator,
                    filter_spec.value
                )
        
        # 4. Group by and aggregate
        if request.operation == "aggregate" and request.group_by and request.aggregations:
            result = transform_service.group_by(result, request.group_by, request.aggregations)
        
        # 5. Apply limit
        if request.limit:
            result = result.head(request.limit)
        
        # Convert result to dict
        result_dict = result.to_dict(orient='records')
        
        return QueryResponse(
            success=True,
            result={
                "data": result_dict,
                "n_rows": len(result),
                "columns": list(result.columns)
            },
            transforms_applied=transforms_applied if transforms_applied else None
        )
    
    except Exception as e:
        return QueryResponse(
            success=False,
            error={"message": str(e), "type": type(e).__name__}
        )


# ---------------------------------------------------
# Data Export
# ---------------------------------------------------
@app.get("/session/{session_id}/export")
def export_data(session_id: str, format: str = "csv"):
    """Export current session data"""
    from fastapi.responses import StreamingResponse
    import io
    
    df = session_store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if format == "csv":
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        stream.seek(0)
        return StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=export_{session_id}.csv"}
        )
    elif format == "json":
        stream = io.StringIO()
        df.to_json(stream, orient="records", indent=2)
        stream.seek(0)
        return StreamingResponse(
            iter([stream.getvalue()]),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=export_{session_id}.json"}
        )
    else:
        raise HTTPException(status_code=400, detail="Unsupported format. Use 'csv' or 'json'")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
