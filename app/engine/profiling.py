"""
Dataset Profiling Module

Analyzes datasets and creates comprehensive profiles including schema information,
data quality metrics, and sample data.

Adaptive Sampling Strategy:
- Datasets ≤ 100K rows: 100% scan (perfect accuracy)
- Datasets 100K-1M rows: 5% sample with 10K minimum (±1% accuracy)
- Datasets > 1M rows: 50K sample cap (±0.4% accuracy)
"""

from __future__ import annotations
from typing import Any, Dict, List


def infer_role(dtype: str) -> str:
    """
    Infer the semantic role of a column based on its data type.
    
    Args:
        dtype: Data type string (e.g., 'INTEGER', 'VARCHAR', 'DATE')
    
    Returns:
        Role classification: 'numeric', 'datetime', or 'categorical'
    
    Examples:
        'INTEGER' → 'numeric'
        'DOUBLE' → 'numeric'
        'VARCHAR' → 'categorical'
        'DATE' → 'datetime'
        'TIMESTAMP' → 'datetime'
    """
    dtype_lower = dtype.lower()
    
    # Check for numeric types
    if any(keyword in dtype_lower for keyword in ["int", "float", "double", "decimal", "numeric"]):
        return "numeric"
    
    # Check for datetime types
    if any(keyword in dtype_lower for keyword in ["date", "time", "timestamp"]):
        return "datetime"
    
    # Default to categorical (strings, booleans, etc.)
    return "categorical"


def _determine_sample_strategy(n_rows: int) -> tuple[int, bool]:
    """
    Determine sampling strategy based on dataset size.
    
    Strategy:
    - Small datasets (≤100K): No sampling - scan all rows for perfect accuracy
    - Medium datasets (100K-1M): Sample 5% with 10K minimum
    - Large datasets (>1M): Cap at 50K sample for consistent performance
    
    Args:
        n_rows: Total number of rows in dataset
    
    Returns:
        Tuple of (sample_size, use_sampling)
        - sample_size: Number of rows to sample
        - use_sampling: Whether to use sampling (False = scan all)
    
    Examples:
        50,000 rows → (50000, False) - scan all, perfect accuracy
        100,000 rows → (10000, True) - 10% sample
        500,000 rows → (25000, True) - 5% sample
        5,000,000 rows → (50000, True) - 1% sample (capped)
    """
    # Small datasets: No sampling for perfect accuracy
    if n_rows <= 100_000:
        return n_rows, False
    
    # Medium datasets: Sample 5% with minimum of 10K rows
    elif n_rows <= 1_000_000:
        sample_size = max(10_000, int(n_rows * 0.05))
        return sample_size, True
    
    # Large datasets: Cap at 50K for consistent performance
    else:
        return 50_000, True


def build_profile_from_duckdb(con, view: str, sample_limit: int = 50) -> Dict[str, Any]:
    """
    Build a comprehensive profile of a dataset using DuckDB.
    
    Analyzes:
    - Schema (column names, types, roles)
    - Data quality (missing value percentages)
    - Dataset dimensions (row/column counts)
    - Sample data (preview rows)
    
    Uses intelligent adaptive sampling:
    - Datasets under 100K rows: Full scan (100% accurate)
    - Datasets over 100K rows: Smart sampling (fast with good accuracy)
    
    Args:
        con: DuckDB connection object
        view: Name of the DuckDB view to profile
        sample_limit: Number of rows to include in sample preview (default: 50)
    
    Returns:
        Dictionary containing:
        - n_rows: Total number of rows
        - n_cols: Total number of columns
        - schema: List of column metadata (name, dtype, role, missing_pct)
        - sample_rows: Preview data as list of dictionaries
    
    Example:
        {
            "n_rows": 100000,
            "n_cols": 4,
            "schema": [
                {"name": "product", "dtype": "VARCHAR", "role": "categorical", "missing_pct": 0.0},
                {"name": "sales", "dtype": "DOUBLE", "role": "numeric", "missing_pct": 2.3}
            ],
            "sample_rows": [
                {"product": "Laptop", "sales": 1299.99},
                ...
            ]
        }
    """
    # =========================================================================
    # STEP 1: Discover Schema
    # =========================================================================
    # Get column information: names and data types
    schema_rows = con.execute(f"DESCRIBE SELECT * FROM {view}").fetchall()
    
    schema = []
    for name, dtype, *_ in schema_rows:
        schema.append({
            "name": name,
            "dtype": str(dtype),
            "role": infer_role(str(dtype)),
            "missing_pct": 0.0  # Will be calculated later
        })
    
    # =========================================================================
    # STEP 2: Count Rows and Columns
    # =========================================================================
    n_rows = int(con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0])
    n_cols = len(schema)
    
    # =========================================================================
    # STEP 3: Adaptive Sampling Strategy
    # =========================================================================
    sample_size, use_sampling = _determine_sample_strategy(n_rows)
    
    # Build the appropriate view for missing value analysis
    if use_sampling:
        # Use sampling for large datasets (fast but approximate)
        sample_view = f"(SELECT * FROM {view} USING SAMPLE {sample_size} ROWS)"
    else:
        # Use full dataset for small datasets (perfect accuracy)
        sample_view = view
    
    # =========================================================================
    # STEP 4: Calculate Missing Value Percentages
    # =========================================================================
    missing = {}
    
    if n_rows > 0:
        for col_info in schema:
            col_name = col_info["name"]
            
            # Calculate percentage of NULL values
            # AVG of 1/0 gives us the proportion of missing values
            query = f"""
                SELECT AVG(CASE WHEN "{col_name}" IS NULL THEN 1 ELSE 0 END)::DOUBLE 
                FROM {sample_view}
            """
            
            try:
                result = con.execute(query).fetchone()[0]
                missing[col_name] = float(result or 0.0)
            except Exception:
                # If calculation fails, assume no missing values
                missing[col_name] = 0.0
    
    # Update schema with missing percentages (rounded to 4 decimal places)
    for col_info in schema:
        col_info["missing_pct"] = round(missing.get(col_info["name"], 0.0), 4)
    
    # =========================================================================
    # STEP 5: Extract Sample Rows for Preview
    # =========================================================================
    # Always get first N rows for preview (regardless of sampling strategy)
    rows_df = con.execute(f"SELECT * FROM {view} LIMIT {sample_limit}").fetchdf()
    sample_rows = rows_df.to_dict(orient="records")
    
    # =========================================================================
    # STEP 6: Return Complete Profile
    # =========================================================================
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "schema": schema,
        "sample_rows": sample_rows
    }
