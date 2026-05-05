from __future__ import annotations

from typing import Optional, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, ConfigDict

from app.db import registry
from app.engine.duckdb_engine import DuckDBUnsupportedTypeError
from app.services.spc_service import run_spc

router = APIRouter()


class SpcRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    chart_type: Literal["i-mr", "xbar-r", "xbar-s", "p", "np", "c", "u", "ewma", "cusum"]
    value_column: Optional[str] = None
    subgroup_column: Optional[str] = None
    subgroup_size: Optional[int] = Field(default=None, ge=2, le=25)
    time_column: Optional[str] = None
    limit: int = Field(default=10000, ge=2, le=200000)
    where: Optional[str] = None
    pipeline_id: Optional[str] = None
    defectives_column: Optional[str] = None
    sample_size_column: Optional[str] = None
    sample_size: Optional[int] = Field(default=None, ge=1)
    defects_column: Optional[str] = None
    area_column: Optional[str] = None
    lambda_param: float = Field(default=0.2, alias="lambda", gt=0, le=1)
    target: Optional[float] = None
    sigma: Optional[float] = Field(default=None, gt=0)
    k: float = Field(default=0.5, gt=0)
    h: float = Field(default=5.0, gt=0)


async def validate_dataset_ready(dataset_id: str, user_id: str) -> dict:
    row_any = await registry.fetchrow(
        """
        SELECT dataset_id, user_id, parquet_ref, state, version, error_message
        FROM datasets
        WHERE dataset_id = $1::uuid
        """,
        dataset_id,
    )

    if not row_any:
        raise HTTPException(status_code=404, detail="Dataset not found")

    row_user_id = row_any.get("user_id") if hasattr(row_any, "get") else row_any["user_id"]
    if row_user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    state = (row_any.get("state") if hasattr(row_any, "get") else row_any["state"]) or "ready"
    if state in ("processing", "reprocessing"):
        raise HTTPException(
            status_code=409,
            detail={"code": "DATASET_PROCESSING", "message": "Dataset is still processing."},
        )
    if state == "failed":
        err = row_any.get("error_message") if hasattr(row_any, "get") else row_any["error_message"]
        raise HTTPException(
            status_code=422,
            detail={"code": "DATASET_FAILED", "message": err or "Dataset processing failed"},
        )

    parquet_ref = row_any.get("parquet_ref") if hasattr(row_any, "get") else row_any["parquet_ref"]
    if not parquet_ref:
        raise HTTPException(
            status_code=409,
            detail={"code": "DATASET_PROCESSING", "message": "Dataset parquet is not ready yet."},
        )

    return dict(row_any)


@router.post("/{dataset_id}/spc")
async def spc_dataset(dataset_id: str, body: SpcRequest, user_id: str = Query(...)):
    try:
        await validate_dataset_ready(dataset_id, user_id)
        result = await run_spc(
            user_id=user_id,
            dataset_id=dataset_id,
            chart_type=body.chart_type,
            value_column=body.value_column,
            subgroup_column=body.subgroup_column,
            subgroup_size=body.subgroup_size,
            time_column=body.time_column,
            limit=body.limit,
            where=body.where,
            pipeline_id=body.pipeline_id,
            defectives_column=body.defectives_column,
            sample_size_column=body.sample_size_column,
            sample_size=body.sample_size,
            defects_column=body.defects_column,
            area_column=body.area_column,
            lambda_param=body.lambda_param,
            target=body.target,
            sigma=body.sigma,
            k=body.k,
            h=body.h,
        )
        return {"ok": True, "spc": result}

    except HTTPException:
        raise

    except DuckDBUnsupportedTypeError as e:
        raise HTTPException(
            status_code=422,
            detail={"code": "UNSUPPORTED_TYPE", "message": str(e)},
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
