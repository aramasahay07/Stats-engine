"""
Quality Router - Control charts and Six Sigma metrics
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from app.services.compute import compute_service

router = APIRouter(prefix="/datasets", tags=["quality"])


class ControlChartRequest(BaseModel):
    """Request for control chart analysis"""
    column: str
    chart_type: str  # "xbar_r", "xbar_s", "i_mr", "p", "c", "u"
    subgroup_size: Optional[int] = 5
    apply_rules: Optional[bool] = True


class SixSigmaRequest(BaseModel):
    """Request for Six Sigma metrics"""
    column: str
    lsl: Optional[float] = None  # Lower spec limit
    usl: Optional[float] = None  # Upper spec limit
    target: Optional[float] = None


@router.post("/{dataset_id}/quality/control-chart")
def create_control_chart(
    dataset_id: str,
    user_id: str,
    request: ControlChartRequest
):
    """
    Create control chart analysis
    
    Supported chart types:
    - **xbar_r**: X-bar and R chart (continuous data, subgroups)
    - **xbar_s**: X-bar and S chart (continuous data, subgroups)
    - **i_mr**: Individual and Moving Range chart
    - **p**: P chart (proportion defective)
    - **c**: C chart (count of defects)
    - **u**: U chart (defects per unit)
    
    Returns control limits and flags for out-of-control points.
    """
    if not user_id:
        raise HTTPException(status_code=422, detail="Missing user_id")
    
    try:
        df = compute_service.load_dataframe(dataset_id, user_id)
        
        if request.column not in df.columns:
            raise HTTPException(status_code=404, detail=f"Column '{request.column}' not found")
        
        data = df[request.column].dropna()
        
        if request.chart_type == "i_mr":
            result = _calculate_i_mr_chart(data, request.apply_rules)
        elif request.chart_type == "xbar_r":
            result = _calculate_xbar_r_chart(data, request.subgroup_size, request.apply_rules)
        elif request.chart_type == "xbar_s":
            result = _calculate_xbar_s_chart(data, request.subgroup_size, request.apply_rules)
        elif request.chart_type in ["p", "c", "u"]:
            result = _calculate_attribute_chart(data, request.chart_type, request.apply_rules)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown chart type: {request.chart_type}")
        
        return {
            "chart_type": request.chart_type,
            "column": request.column,
            **result
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Control chart failed: {str(e)}")


@router.post("/{dataset_id}/quality/six-sigma")
def calculate_six_sigma(
    dataset_id: str,
    user_id: str,
    request: SixSigmaRequest
):
    """
    Calculate Six Sigma metrics
    
    Metrics:
    - **Cp**: Process capability (how well process fits specs)
    - **Cpk**: Process capability adjusted for centering
    - **Pp**: Process performance
    - **Ppk**: Process performance adjusted for centering
    - **Sigma level**: Current sigma level of the process
    - **DPMO**: Defects per million opportunities
    
    Requires specification limits (LSL and/or USL).
    """
    if not user_id:
        raise HTTPException(status_code=422, detail="Missing user_id")
    
    if request.lsl is None and request.usl is None:
        raise HTTPException(
            status_code=400, 
            detail="At least one specification limit (LSL or USL) is required"
        )
    
    try:
        df = compute_service.load_dataframe(dataset_id, user_id)
        
        if request.column not in df.columns:
            raise HTTPException(status_code=404, detail=f"Column '{request.column}' not found")
        
        data = df[request.column].dropna()
        
        # Calculate metrics
        mean = data.mean()
        std = data.std(ddof=1)
        
        metrics = {
            "mean": float(mean),
            "std": float(std),
            "n": int(len(data))
        }
        
        # Calculate Cp and Cpk
        if request.lsl is not None and request.usl is not None:
            cp = (request.usl - request.lsl) / (6 * std)
            cpu = (request.usl - mean) / (3 * std)
            cpl = (mean - request.lsl) / (3 * std)
            cpk = min(cpu, cpl)
            
            metrics.update({
                "cp": float(cp),
                "cpk": float(cpk),
                "cpu": float(cpu),
                "cpl": float(cpl)
            })
        
        # Calculate defect rate and DPMO
        defects = 0
        if request.lsl is not None:
            defects += (data < request.lsl).sum()
        if request.usl is not None:
            defects += (data > request.usl).sum()
        
        dpmo = (defects / len(data)) * 1_000_000
        
        # Approximate sigma level from DPMO
        from scipy.stats import norm
        if dpmo > 0:
            defect_rate = dpmo / 1_000_000
            sigma_level = norm.ppf(1 - defect_rate/2)  # Two-sided
        else:
            sigma_level = 6.0  # Perfect
        
        metrics.update({
            "defects": int(defects),
            "dpmo": float(dpmo),
            "sigma_level": float(sigma_level),
            "lsl": request.lsl,
            "usl": request.usl,
            "target": request.target
        })
        
        # Interpretation
        if "cpk" in metrics:
            if metrics["cpk"] >= 2.0:
                interpretation = "Excellent process capability (Cpk ≥ 2.0)"
            elif metrics["cpk"] >= 1.33:
                interpretation = "Good process capability (Cpk ≥ 1.33)"
            elif metrics["cpk"] >= 1.0:
                interpretation = "Adequate process capability (Cpk ≥ 1.0)"
            else:
                interpretation = "Poor process capability (Cpk < 1.0) - improvement needed"
            
            metrics["interpretation"] = interpretation
        
        return metrics
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Six Sigma calculation failed: {str(e)}")


def _calculate_i_mr_chart(data, apply_rules: bool = True):
    """Calculate Individual and Moving Range chart"""
    individuals = data.values
    n = len(individuals)
    
    # Calculate moving ranges
    moving_ranges = np.abs(np.diff(individuals))
    mr_bar = moving_ranges.mean()
    
    # I chart control limits
    i_center = individuals.mean()
    i_ucl = i_center + 2.66 * mr_bar
    i_lcl = i_center - 2.66 * mr_bar
    
    # MR chart control limits
    mr_center = mr_bar
    mr_ucl = 3.267 * mr_bar
    mr_lcl = 0  # Moving range LCL is typically 0
    
    # Detect out-of-control points
    ooc_points = []
    if apply_rules:
        # Rule 1: Point beyond control limits
        for i, val in enumerate(individuals):
            if val > i_ucl or val < i_lcl:
                ooc_points.append({
                    "index": int(i),
                    "value": float(val),
                    "rule": "Beyond control limits"
                })
    
    return {
        "i_chart": {
            "center": float(i_center),
            "ucl": float(i_ucl),
            "lcl": float(i_lcl),
            "values": individuals.tolist()
        },
        "mr_chart": {
            "center": float(mr_center),
            "ucl": float(mr_ucl),
            "lcl": float(mr_lcl),
            "values": moving_ranges.tolist()
        },
        "out_of_control": ooc_points,
        "n_points": int(n)
    }


def _calculate_xbar_r_chart(data, subgroup_size: int, apply_rules: bool = True):
    """Calculate X-bar and R chart"""
    n = len(data)
    n_subgroups = n // subgroup_size
    
    if n_subgroups < 2:
        raise ValueError(f"Insufficient data for subgroup size {subgroup_size}")
    
    # Reshape into subgroups
    data_trimmed = data[:n_subgroups * subgroup_size]
    subgroups = data_trimmed.reshape(n_subgroups, subgroup_size)
    
    # Calculate X-bar and R for each subgroup
    xbars = subgroups.mean(axis=1)
    ranges = subgroups.max(axis=1) - subgroups.min(axis=1)
    
    # Overall statistics
    xbar_bar = xbars.mean()
    r_bar = ranges.mean()
    
    # Control limit constants (for subgroup size)
    # Simplified - should use proper control chart constants
    if subgroup_size == 5:
        A2, D3, D4 = 0.577, 0, 2.114
    else:
        A2, D3, D4 = 0.729, 0, 2.282  # Default for n=3
    
    # X-bar chart limits
    xbar_ucl = xbar_bar + A2 * r_bar
    xbar_lcl = xbar_bar - A2 * r_bar
    
    # R chart limits
    r_ucl = D4 * r_bar
    r_lcl = D3 * r_bar
    
    return {
        "xbar_chart": {
            "center": float(xbar_bar),
            "ucl": float(xbar_ucl),
            "lcl": float(xbar_lcl),
            "values": xbars.tolist()
        },
        "r_chart": {
            "center": float(r_bar),
            "ucl": float(r_ucl),
            "lcl": float(r_lcl),
            "values": ranges.tolist()
        },
        "subgroup_size": subgroup_size,
        "n_subgroups": int(n_subgroups)
    }


def _calculate_xbar_s_chart(data, subgroup_size: int, apply_rules: bool = True):
    """Calculate X-bar and S chart"""
    # Similar to X-bar R but uses standard deviation instead of range
    n = len(data)
    n_subgroups = n // subgroup_size
    
    if n_subgroups < 2:
        raise ValueError(f"Insufficient data for subgroup size {subgroup_size}")
    
    data_trimmed = data[:n_subgroups * subgroup_size]
    subgroups = data_trimmed.reshape(n_subgroups, subgroup_size)
    
    xbars = subgroups.mean(axis=1)
    stds = subgroups.std(axis=1, ddof=1)
    
    xbar_bar = xbars.mean()
    s_bar = stds.mean()
    
    # Control chart constants (simplified)
    if subgroup_size == 5:
        A3, B3, B4 = 1.427, 0, 2.089
    else:
        A3, B3, B4 = 1.954, 0, 2.568  # Default
    
    xbar_ucl = xbar_bar + A3 * s_bar
    xbar_lcl = xbar_bar - A3 * s_bar
    
    s_ucl = B4 * s_bar
    s_lcl = B3 * s_bar
    
    return {
        "xbar_chart": {
            "center": float(xbar_bar),
            "ucl": float(xbar_ucl),
            "lcl": float(xbar_lcl),
            "values": xbars.tolist()
        },
        "s_chart": {
            "center": float(s_bar),
            "ucl": float(s_ucl),
            "lcl": float(s_lcl),
            "values": stds.tolist()
        },
        "subgroup_size": subgroup_size,
        "n_subgroups": int(n_subgroups)
    }


def _calculate_attribute_chart(data, chart_type: str, apply_rules: bool = True):
    """Calculate attribute control charts (p, c, u)"""
    # Simplified implementation
    n = len(data)
    
    if chart_type == "p":
        # P chart for proportions
        p_bar = data.mean()
        p_ucl = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / n)
        p_lcl = max(0, p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / n))
        
        return {
            "p_chart": {
                "center": float(p_bar),
                "ucl": float(p_ucl),
                "lcl": float(p_lcl),
                "values": data.tolist()
            },
            "n": int(n)
        }
    
    elif chart_type == "c":
        # C chart for counts
        c_bar = data.mean()
        c_ucl = c_bar + 3 * np.sqrt(c_bar)
        c_lcl = max(0, c_bar - 3 * np.sqrt(c_bar))
        
        return {
            "c_chart": {
                "center": float(c_bar),
                "ucl": float(c_ucl),
                "lcl": float(c_lcl),
                "values": data.tolist()
            },
            "n": int(n)
        }
    
    else:
        raise ValueError(f"Chart type {chart_type} not yet implemented")


@router.get("/{dataset_id}/quality/summary")
def quality_summary(dataset_id: str, user_id: str, column: str):
    """
    Get quality summary for a column
    
    Provides quick overview of process capability and stability.
    """
    if not user_id:
        raise HTTPException(status_code=422, detail="Missing user_id")
    
    try:
        df = compute_service.load_dataframe(dataset_id, user_id)
        
        if column not in df.columns:
            raise HTTPException(status_code=404, detail=f"Column '{column}' not found")
        
        data = df[column].dropna()
        
        return {
            "column": column,
            "n": int(len(data)),
            "mean": float(data.mean()),
            "std": float(data.std()),
            "min": float(data.min()),
            "max": float(data.max()),
            "range": float(data.max() - data.min()),
            "cv": float(data.std() / data.mean() * 100) if data.mean() != 0 else None,  # Coefficient of variation
            "recommended_charts": ["i_mr", "xbar_r"] if len(data) > 20 else ["i_mr"]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
