from __future__ import annotations
from pathlib import Path
import hashlib
import uuid
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def csv_to_parquet_streaming(csv_path: Path, parquet_path: Path) -> tuple[int, int]:
    """Stream CSV -> Parquet without loading entire file into memory."""

    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    read_options = pacsv.ReadOptions(autogenerate_column_names=False)
    parse_options = pacsv.ParseOptions(delimiter=",")
    convert_options = pacsv.ConvertOptions(strings_can_be_null=True)

    reader = pacsv.open_csv(csv_path.as_posix(), read_options=read_options, parse_options=parse_options, convert_options=convert_options)
    writer = None
    total_rows = 0
    n_cols = 0

    for batch in reader:
        table = pa.Table.from_batches([batch])
        if writer is None:
            n_cols = table.num_columns
            writer = pq.ParquetWriter(parquet_path.as_posix(), table.schema, compression="zstd")
        writer.write_table(table)
        total_rows += table.num_rows

    if writer:
        writer.close()

    return total_rows, n_cols

def xlsx_to_parquet(xlsx_path: Path, parquet_path: Path, sheet_name: str | int | None = 0) -> tuple[int, int]:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    # ------------------------------------------------------------
    # Normalize risky Excel-derived types BEFORE writing Parquet
    # (prevents DuckDB issues with TIME/TZ-like parquet types)
    # ------------------------------------------------------------
    for col in df.columns:
        s = df[col]

        # tz-aware datetime columns -> string
        if pd.api.types.is_datetime64tz_dtype(s):
            df[col] = s.astype(str)
            continue

        # object columns can contain datetime.time, tz-aware datetime, mixed objects
        if s.dtype == "object":
            sample = s.dropna().head(20)
            if sample.empty:
                continue

            needs_cast = False
            for v in sample:
                # Excel time cells often become datetime.time objects
                # (time has hour/minute/second but no "date" attribute)
                if hasattr(v, "hour") and hasattr(v, "minute") and not hasattr(v, "date"):
                    needs_cast = True
                    break

                # tz-aware datetime objects
                if hasattr(v, "tzinfo") and v.tzinfo is not None:
                    needs_cast = True
                    break

            if needs_cast:
                df[col] = s.astype(str)

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, parquet_path.as_posix(), compression="zstd")
    return int(table.num_rows), int(table.num_columns)


def parquet_copy(parquet_in: Path, parquet_out: Path) -> tuple[int, int]:
    parquet_out.parent.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(parquet_in.as_posix())
    pq.write_table(table, parquet_out.as_posix(), compression="zstd")
    return int(table.num_rows), int(table.num_columns)

def new_dataset_id() -> str:
    return str(uuid.uuid4())
