from __future__ import annotations

from pathlib import Path
import pandas as pd

def build_parquet(input_path: Path, output_parquet_path: Path) -> None:
    """Convert CSV/XLSX/TXT to parquet.

    NOTE: This is a starter implementation. For very large CSVs,
    replace with chunked conversion using pyarrow.dataset or pandas chunks.
    """
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(input_path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(input_path)
    elif suffix == ".txt":
        df = pd.read_csv(input_path, sep=None, engine="python")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    df.to_parquet(output_parquet_path, index=False)
