from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from pandas import to_numeric

from app.db import registry
from app.engine.duckdb_engine import DuckDBEngine
from app.engine.pipeline import compile_pipeline_sql
from app.services.stats_service import _get_parquet_local
from app.transformers.registry import transformer_registry


# SPC constants (subgroup size n = 2..25)
A2 = {
    2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373,
    9: 0.337, 10: 0.308, 11: 0.285, 12: 0.266, 13: 0.249, 14: 0.235, 15: 0.223,
    16: 0.212, 17: 0.203, 18: 0.194, 19: 0.187, 20: 0.180, 21: 0.173, 22: 0.167,
    23: 0.162, 24: 0.157, 25: 0.153,
}
A3 = {
    2: 2.659, 3: 1.954, 4: 1.628, 5: 1.427, 6: 1.287, 7: 1.182, 8: 1.099,
    9: 1.032, 10: 0.975, 11: 0.927, 12: 0.886, 13: 0.850, 14: 0.817, 15: 0.789,
    16: 0.763, 17: 0.739, 18: 0.718, 19: 0.698, 20: 0.680, 21: 0.663, 22: 0.647,
    23: 0.633, 24: 0.619, 25: 0.606,
}
B3 = {
    2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.030, 7: 0.118, 8: 0.185, 9: 0.239,
    10: 0.284, 11: 0.321, 12: 0.354, 13: 0.382, 14: 0.406, 15: 0.428, 16: 0.448,
    17: 0.466, 18: 0.482, 19: 0.497, 20: 0.510, 21: 0.523, 22: 0.534, 23: 0.545,
    24: 0.555, 25: 0.565,
}
B4 = {
    2: 3.267, 3: 2.568, 4: 2.266, 5: 2.089, 6: 1.970, 7: 1.882, 8: 1.815,
    9: 1.761, 10: 1.716, 11: 1.679, 12: 1.646, 13: 1.618, 14: 1.594, 15: 1.572,
    16: 1.552, 17: 1.534, 18: 1.518, 19: 1.503, 20: 1.490, 21: 1.477, 22: 1.466,
    23: 1.455, 24: 1.445, 25: 1.435,
}
D3 = {
    2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.076, 8: 0.136, 9: 0.184,
    10: 0.223, 11: 0.256, 12: 0.283, 13: 0.307, 14: 0.328, 15: 0.347, 16: 0.363,
    17: 0.378, 18: 0.391, 19: 0.403, 20: 0.415, 21: 0.425, 22: 0.434, 23: 0.443,
    24: 0.451, 25: 0.459,
}
D4 = {
    2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864,
    9: 1.816, 10: 1.777, 11: 1.744, 12: 1.717, 13: 1.693, 14: 1.672, 15: 1.653,
    16: 1.637, 17: 1.622, 18: 1.608, 19: 1.597, 20: 1.585, 21: 1.575, 22: 1.566,
    23: 1.557, 24: 1.548, 25: 1.541,
}
d2 = {
    2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847,
    9: 2.970, 10: 3.078, 11: 3.173, 12: 3.258, 13: 3.336, 14: 3.407, 15: 3.472,
    16: 3.532, 17: 3.588, 18: 3.640, 19: 3.689, 20: 3.735, 21: 3.778, 22: 3.819,
    23: 3.858, 24: 3.895, 25: 3.931,
}
c4 = {
    2: 0.7979, 3: 0.8862, 4: 0.9213, 5: 0.9400, 6: 0.9515, 7: 0.9594, 8: 0.9650,
    9: 0.9693, 10: 0.9727, 11: 0.9754, 12: 0.9776, 13: 0.9794, 14: 0.9810,
    15: 0.9823, 16: 0.9835, 17: 0.9845, 18: 0.9854, 19: 0.9862, 20: 0.9869,
    21: 0.9876, 22: 0.9882, 23: 0.9887, 24: 0.9892, 25: 0.9896,
}


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _sigma_from_limits(center_line: float, ucl: float, fallback: Sequence[float]) -> float:
    spread = float(ucl) - float(center_line)
    if np.isfinite(spread) and spread > 0:
        return spread / 3.0

    arr = np.asarray(fallback, dtype=float)
    if arr.size >= 2:
        sigma = float(np.std(arr, ddof=1))
        if np.isfinite(sigma) and sigma > 0:
            return sigma

    return 0.0


def nelson_rules(values: np.ndarray, center_line: float, sigma: float) -> List[List[int]]:
    n = len(values)
    out: List[List[int]] = [[] for _ in range(n)]
    if sigma <= 0 or not np.isfinite(sigma):
        return out

    z = (values - center_line) / sigma
    sign = np.sign(z)

    for i in range(n):
        if abs(z[i]) > 3:
            out[i].append(1)
    for i in range(8, n):
        window = sign[i - 8:i + 1]
        if np.all(window == 1) or np.all(window == -1):
            out[i].append(2)
    for i in range(5, n):
        seg = values[i - 5:i + 1]
        diffs = np.diff(seg)
        if np.all(diffs > 0) or np.all(diffs < 0):
            out[i].append(3)
    for i in range(13, n):
        seg = values[i - 13:i + 1]
        diffs = np.diff(seg)
        if len(diffs) >= 2 and np.all(np.sign(diffs[1:]) != np.sign(diffs[:-1])) and np.all(diffs != 0):
            out[i].append(4)
    for i in range(2, n):
        window = z[i - 2:i + 1]
        if (window > 2).sum() >= 2 or (window < -2).sum() >= 2:
            out[i].append(5)
    for i in range(4, n):
        window = z[i - 4:i + 1]
        if (window > 1).sum() >= 4 or (window < -1).sum() >= 4:
            out[i].append(6)
    for i in range(14, n):
        if np.all(np.abs(z[i - 14:i + 1]) < 1):
            out[i].append(7)
    for i in range(7, n):
        if np.all(np.abs(z[i - 7:i + 1]) > 1):
            out[i].append(8)

    return out


def _summary(violations: List[List[int]]) -> Dict[int, int]:
    summary: Dict[int, int] = {}
    for item in violations:
        for rule in item:
            summary[rule] = summary.get(rule, 0) + 1
    return summary


def _points(values: np.ndarray, labels: List[str], violations: List[List[int]], offset: int = 0) -> List[Dict[str, Any]]:
    return [
        {
            "index": i + 1 + offset,
            "label": labels[i],
            "value": float(values[i]),
            "violations": violations[i],
        }
        for i in range(len(values))
    ]


def _points_with_limits(
    values: np.ndarray,
    labels: List[str],
    violations: List[List[int]],
    ucl_values: Sequence[float],
    lcl_values: Sequence[float],
    extras: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for i in range(len(values)):
        point = {
            "index": i + 1,
            "label": labels[i],
            "value": float(values[i]),
            "violations": violations[i],
            "UCL": float(ucl_values[i]),
            "LCL": float(lcl_values[i]),
        }
        if extras:
            point.update(extras[i])
        items.append(point)
    return items


def _build_subgroups(
    values: np.ndarray,
    labels: List[str],
    keys: Optional[List[Any]],
    fixed_size: Optional[int],
) -> tuple[List[np.ndarray], List[str]]:
    if keys is not None and len(keys) == len(values):
        order: List[str] = []
        seen: Dict[str, List[int]] = {}
        for idx, key in enumerate(keys):
            skey = str(key)
            if skey not in seen:
                seen[skey] = []
                order.append(skey)
            seen[skey].append(idx)
        return [values[seen[key]] for key in order], order

    if not fixed_size:
        raise ValueError("subgroup_size is required when subgroup_column is not provided")

    groups: List[np.ndarray] = []
    group_labels: List[str] = []
    n_rows = len(values)
    for start in range(0, n_rows - fixed_size + 1, fixed_size):
        end = start + fixed_size
        groups.append(values[start:end])
        group_labels.append(f"{labels[start]}..{labels[end - 1]}")

    return groups, group_labels


def _enforce_subgroup_size(
    groups: List[np.ndarray],
    labels: List[str],
    requested_size: Optional[int],
    chart_name: str,
) -> tuple[List[np.ndarray], List[str], int]:
    if not groups:
        raise ValueError(f"{chart_name} needs at least 2 subgroups")

    sizes = [len(group) for group in groups]
    expected = int(requested_size or sizes[0])
    if expected < 2:
        raise ValueError(f"{chart_name} needs subgroups of size >= 2")

    bad_sizes = sorted({size for size in sizes if size != expected})
    if bad_sizes:
        raise ValueError(
            f"{chart_name} requires consistent subgroup size {expected}; found mismatched sizes {bad_sizes}"
        )

    if len(groups) < 2:
        raise ValueError(f"{chart_name} needs at least 2 subgroups")

    return groups, labels, expected


def _normalize_label_series(series) -> List[str]:
    return series.astype(str).tolist()


def _aggregate_by_subgroup(
    df,
    subgroup_column: str,
    measure_columns: List[str],
    time_column: Optional[str] = None,
):
    grouped_rows: List[Dict[str, Any]] = []
    for subgroup_value, group in df.groupby(subgroup_column, sort=False, dropna=False):
        row: Dict[str, Any] = {"label": str(subgroup_value)}
        if time_column:
            row["order_value"] = group[time_column].iloc[0]
        for column in measure_columns:
            row[column] = float(to_numeric(group[column], errors="coerce").fillna(0).sum())
        grouped_rows.append(row)

    if time_column:
        grouped_rows.sort(key=lambda item: str(item.get("order_value")))

    return grouped_rows


def compute_imr(values: np.ndarray, labels: List[str], value_column: str, time_column: Optional[str]) -> Dict[str, Any]:
    if len(values) < 2:
        raise ValueError("I-MR requires at least 2 observations")

    center_line = float(values.mean())
    moving_ranges = np.abs(np.diff(values))
    mr_center_line = float(moving_ranges.mean())

    sigma = mr_center_line / d2[2]
    ucl_i = center_line + 3 * sigma
    lcl_i = center_line - 3 * sigma
    ucl_mr = D4[2] * mr_center_line
    lcl_mr = D3[2] * mr_center_line

    iv = nelson_rules(values, center_line, _sigma_from_limits(center_line, ucl_i, values))
    mv = nelson_rules(moving_ranges, mr_center_line, _sigma_from_limits(mr_center_line, ucl_mr, moving_ranges))

    return {
        "chart_type": "i-mr",
        "primary": {
            "name": "Individual Value",
            "points": _points(values, labels, iv),
            "centerLine": center_line,
            "UCL": float(ucl_i),
            "LCL": float(lcl_i),
        },
        "secondary": {
            "name": "Moving Range",
            "points": _points(moving_ranges, labels[1:], mv, offset=1),
            "centerLine": mr_center_line,
            "UCL": float(ucl_mr),
            "LCL": float(lcl_mr),
        },
        "meta": {
            "n_samples": int(len(values)),
            "value_column": value_column,
            "time_column": time_column,
            "rules_summary": _summary(iv + mv),
        },
    }


def compute_xbar_r(
    values: np.ndarray,
    labels: List[str],
    value_column: str,
    keys: Optional[List[Any]],
    fixed_size: Optional[int],
    subgroup_column: Optional[str],
) -> Dict[str, Any]:
    groups, group_labels = _build_subgroups(values, labels, keys, fixed_size)
    groups, group_labels, n = _enforce_subgroup_size(groups, group_labels, fixed_size, "Xbar-R")

    if n > 10:
        raise ValueError("Xbar-R is for subgroup sizes up to 10; use Xbar-S for larger subgroups")

    subgroup_means = np.array([group.mean() for group in groups], dtype=float)
    subgroup_ranges = np.array([group.max() - group.min() for group in groups], dtype=float)

    xbar_bar = float(subgroup_means.mean())
    r_bar = float(subgroup_ranges.mean())
    sigma = r_bar / d2[n]

    ucl_x = xbar_bar + A2[n] * r_bar
    lcl_x = xbar_bar - A2[n] * r_bar
    ucl_r = D4[n] * r_bar
    lcl_r = D3[n] * r_bar

    xv = nelson_rules(subgroup_means, xbar_bar, _sigma_from_limits(xbar_bar, ucl_x, subgroup_means))
    rv = nelson_rules(subgroup_ranges, r_bar, _sigma_from_limits(r_bar, ucl_r, subgroup_ranges))

    return {
        "chart_type": "xbar-r",
        "primary": {
            "name": f"Subgroup Mean (Xbar, n={n})",
            "points": _points(subgroup_means, group_labels, xv),
            "centerLine": xbar_bar,
            "UCL": float(ucl_x),
            "LCL": float(lcl_x),
        },
        "secondary": {
            "name": "Subgroup Range (R)",
            "points": _points(subgroup_ranges, group_labels, rv),
            "centerLine": r_bar,
            "UCL": float(ucl_r),
            "LCL": float(lcl_r),
        },
        "meta": {
            "n_samples": len(groups),
            "subgroup_size": n,
            "value_column": value_column,
            "subgroup_column": subgroup_column,
            "rules_summary": _summary(xv + rv),
        },
    }


def compute_xbar_s(
    values: np.ndarray,
    labels: List[str],
    value_column: str,
    keys: Optional[List[Any]],
    fixed_size: Optional[int],
    subgroup_column: Optional[str],
) -> Dict[str, Any]:
    groups, group_labels = _build_subgroups(values, labels, keys, fixed_size)
    groups, group_labels, n = _enforce_subgroup_size(groups, group_labels, fixed_size, "Xbar-S")

    if n > 25:
        raise ValueError("Xbar-S supports subgroup sizes up to 25")

    subgroup_means = np.array([group.mean() for group in groups], dtype=float)
    subgroup_std = np.array([group.std(ddof=1) for group in groups], dtype=float)

    xbar_bar = float(subgroup_means.mean())
    s_bar = float(subgroup_std.mean())
    sigma = s_bar / c4[n]

    ucl_x = xbar_bar + A3[n] * s_bar
    lcl_x = xbar_bar - A3[n] * s_bar
    ucl_s = B4[n] * s_bar
    lcl_s = B3[n] * s_bar

    xv = nelson_rules(subgroup_means, xbar_bar, _sigma_from_limits(xbar_bar, ucl_x, subgroup_means))
    sv = nelson_rules(subgroup_std, s_bar, _sigma_from_limits(s_bar, ucl_s, subgroup_std))

    return {
        "chart_type": "xbar-s",
        "primary": {
            "name": f"Subgroup Mean (Xbar, n={n})",
            "points": _points(subgroup_means, group_labels, xv),
            "centerLine": xbar_bar,
            "UCL": float(ucl_x),
            "LCL": float(lcl_x),
        },
        "secondary": {
            "name": "Subgroup Std Dev (S)",
            "points": _points(subgroup_std, group_labels, sv),
            "centerLine": s_bar,
            "UCL": float(ucl_s),
            "LCL": float(lcl_s),
        },
        "meta": {
            "n_samples": len(groups),
            "subgroup_size": n,
            "value_column": value_column,
            "subgroup_column": subgroup_column,
            "rules_summary": _summary(xv + sv),
        },
    }


def compute_p_chart(
    defectives: np.ndarray,
    sample_sizes: np.ndarray,
    labels: List[str],
    defectives_column: str,
    sample_size_column: Optional[str],
    subgroup_column: Optional[str],
) -> Dict[str, Any]:
    if len(defectives) < 2:
        raise ValueError("P chart needs at least 2 subgroups")
    if np.any(sample_sizes <= 0):
        raise ValueError("P chart sample sizes must be > 0")
    if np.any(defectives < 0):
        raise ValueError("P chart defectives must be >= 0")
    if np.any(defectives > sample_sizes):
        raise ValueError("P chart defectives cannot exceed sample size")

    proportions = defectives / sample_sizes
    p_bar = float(defectives.sum() / sample_sizes.sum())
    sigma_each = np.sqrt(p_bar * (1.0 - p_bar) / sample_sizes)
    ucl = np.minimum(1.0, p_bar + 3.0 * sigma_each)
    lcl = np.maximum(0.0, p_bar - 3.0 * sigma_each)
    sigma_center = float(np.sqrt(p_bar * (1.0 - p_bar) / float(sample_sizes.mean()))) if p_bar < 1 else 0.0
    violations = nelson_rules(proportions, p_bar, sigma_center)
    for i, value in enumerate(proportions):
        if value > ucl[i] or value < lcl[i]:
            if 1 not in violations[i]:
                violations[i].append(1)

    extras = [
        {"defectives": int(defectives[i]), "sample_size": int(sample_sizes[i])}
        for i in range(len(defectives))
    ]

    return {
        "chart_type": "p",
        "primary": {
            "name": "Proportion Defective (P)",
            "points": _points_with_limits(proportions, labels, violations, ucl, lcl, extras=extras),
            "centerLine": p_bar,
            "UCL": [float(v) for v in ucl],
            "LCL": [float(v) for v in lcl],
        },
        "meta": {
            "n_samples": len(labels),
            "defectives_column": defectives_column,
            "sample_size_column": sample_size_column,
            "subgroup_column": subgroup_column,
            "rules_summary": _summary(violations),
            "total_defective": int(defectives.sum()),
            "total_inspected": int(sample_sizes.sum()),
        },
    }


def compute_np_chart(
    defectives: np.ndarray,
    sample_size: int,
    labels: List[str],
    defectives_column: str,
    subgroup_column: Optional[str],
) -> Dict[str, Any]:
    if len(defectives) < 2:
        raise ValueError("NP chart needs at least 2 subgroups")
    if sample_size <= 0:
        raise ValueError("NP chart sample_size must be > 0")
    if np.any(defectives < 0):
        raise ValueError("NP chart defectives must be >= 0")
    if np.any(defectives > sample_size):
        raise ValueError("NP chart defectives cannot exceed sample_size")

    center_line = float(defectives.mean())
    p_bar = center_line / sample_size
    sigma = float(np.sqrt(sample_size * p_bar * (1.0 - p_bar)))
    ucl = center_line + 3.0 * sigma
    lcl = max(0.0, center_line - 3.0 * sigma)

    violations = nelson_rules(defectives, center_line, sigma)
    for i, value in enumerate(defectives):
        if value > ucl or value < lcl:
            if 1 not in violations[i]:
                violations[i].append(1)

    extras = [
        {"defectives": int(defectives[i]), "sample_size": int(sample_size)}
        for i in range(len(defectives))
    ]

    return {
        "chart_type": "np",
        "primary": {
            "name": "Number Defective (NP)",
            "points": _points(defectives, labels, violations),
            "centerLine": center_line,
            "UCL": float(ucl),
            "LCL": float(lcl),
        },
        "meta": {
            "n_samples": len(labels),
            "defectives_column": defectives_column,
            "subgroup_column": subgroup_column,
            "sample_size": int(sample_size),
            "average_proportion": float(p_bar),
            "rules_summary": _summary(violations),
        },
    }


def compute_c_chart(
    defects: np.ndarray,
    labels: List[str],
    defects_column: str,
    subgroup_column: Optional[str],
) -> Dict[str, Any]:
    if len(defects) < 2:
        raise ValueError("C chart needs at least 2 subgroups")
    if np.any(defects < 0):
        raise ValueError("C chart defects must be >= 0")

    center_line = float(defects.mean())
    sigma = float(np.sqrt(center_line))
    ucl = center_line + 3.0 * sigma
    lcl = max(0.0, center_line - 3.0 * sigma)

    violations = nelson_rules(defects, center_line, sigma)
    for i, value in enumerate(defects):
        if value > ucl or value < lcl:
            if 1 not in violations[i]:
                violations[i].append(1)

    return {
        "chart_type": "c",
        "primary": {
            "name": "Count of Defects (C)",
            "points": _points(defects, labels, violations),
            "centerLine": center_line,
            "UCL": float(ucl),
            "LCL": float(lcl),
        },
        "meta": {
            "n_samples": len(labels),
            "defects_column": defects_column,
            "subgroup_column": subgroup_column,
            "rules_summary": _summary(violations),
            "total_defects": int(defects.sum()),
        },
    }


def compute_u_chart(
    defects: np.ndarray,
    areas: np.ndarray,
    labels: List[str],
    defects_column: str,
    area_column: str,
    subgroup_column: Optional[str],
) -> Dict[str, Any]:
    if len(defects) < 2:
        raise ValueError("U chart needs at least 2 subgroups")
    if np.any(defects < 0):
        raise ValueError("U chart defects must be >= 0")
    if np.any(areas <= 0):
        raise ValueError("U chart area values must be > 0")

    values = defects / areas
    center_line = float(defects.sum() / areas.sum())
    sigma_center = float(np.sqrt(center_line / float(areas.mean())))
    ucl = center_line + 3.0 * np.sqrt(center_line / areas)
    lcl = np.maximum(0.0, center_line - 3.0 * np.sqrt(center_line / areas))

    violations = nelson_rules(values, center_line, sigma_center)
    for i, value in enumerate(values):
        if value > ucl[i] or value < lcl[i]:
            if 1 not in violations[i]:
                violations[i].append(1)

    extras = [
        {"defects": int(defects[i]), "area": float(areas[i])}
        for i in range(len(defects))
    ]

    return {
        "chart_type": "u",
        "primary": {
            "name": "Defects Per Unit (U)",
            "points": _points_with_limits(values, labels, violations, ucl, lcl, extras=extras),
            "centerLine": center_line,
            "UCL": [float(v) for v in ucl],
            "LCL": [float(v) for v in lcl],
        },
        "meta": {
            "n_samples": len(labels),
            "defects_column": defects_column,
            "area_column": area_column,
            "subgroup_column": subgroup_column,
            "rules_summary": _summary(violations),
            "total_defects": int(defects.sum()),
            "total_area": float(areas.sum()),
        },
    }


def compute_ewma(
    values: np.ndarray,
    labels: List[str],
    value_column: str,
    time_column: Optional[str],
    lambda_param: float = 0.2,
    target: Optional[float] = None,
    sigma: Optional[float] = None,
) -> Dict[str, Any]:
    if len(values) < 3:
        raise ValueError("EWMA requires at least 3 observations")
    if not 0 < lambda_param <= 1:
        raise ValueError("EWMA lambda must be between 0 and 1")

    if target is None:
        target = float(values.mean())
    if sigma is None:
        moving_ranges = np.abs(np.diff(values))
        sigma = float(moving_ranges.mean() / d2[2])
    if sigma <= 0:
        raise ValueError("EWMA sigma must be > 0")

    ewma_values = [target]
    for value in values:
        ewma_values.append(lambda_param * float(value) + (1.0 - lambda_param) * ewma_values[-1])
    ewma_arr = np.asarray(ewma_values[1:], dtype=float)

    ucl_values = []
    lcl_values = []
    for i in range(len(values)):
        factor = np.sqrt((lambda_param / (2.0 - lambda_param)) * (1.0 - (1.0 - lambda_param) ** (2 * (i + 1))))
        ucl_values.append(target + 3.0 * sigma * factor)
        lcl_values.append(target - 3.0 * sigma * factor)

    ucl = np.asarray(ucl_values, dtype=float)
    lcl = np.asarray(lcl_values, dtype=float)
    violations = nelson_rules(ewma_arr, target, sigma * np.sqrt(lambda_param / (2.0 - lambda_param)))
    for i, value in enumerate(ewma_arr):
        if value > ucl[i] or value < lcl[i]:
            if 1 not in violations[i]:
                violations[i].append(1)

    extras = [{"original_value": float(values[i])} for i in range(len(values))]
    return {
        "chart_type": "ewma",
        "primary": {
            "name": "EWMA",
            "points": _points_with_limits(ewma_arr, labels, violations, ucl, lcl, extras=extras),
            "centerLine": float(target),
            "UCL": [float(v) for v in ucl],
            "LCL": [float(v) for v in lcl],
        },
        "meta": {
            "n_samples": len(labels),
            "value_column": value_column,
            "time_column": time_column,
            "lambda": float(lambda_param),
            "target": float(target),
            "sigma": float(sigma),
            "rules_summary": _summary(violations),
        },
    }


def compute_cusum(
    values: np.ndarray,
    labels: List[str],
    value_column: str,
    time_column: Optional[str],
    target: Optional[float] = None,
    sigma: Optional[float] = None,
    k: float = 0.5,
    h: float = 5.0,
) -> Dict[str, Any]:
    if len(values) < 3:
        raise ValueError("CUSUM requires at least 3 observations")
    if sigma is None:
        moving_ranges = np.abs(np.diff(values))
        sigma = float(moving_ranges.mean() / d2[2])
    if sigma <= 0:
        raise ValueError("CUSUM sigma must be > 0")
    if target is None:
        target = float(values.mean())

    k_value = float(k) * float(sigma)
    h_value = float(h) * float(sigma)

    c_plus = [0.0]
    c_minus = [0.0]
    for value in values:
        c_plus.append(max(0.0, c_plus[-1] + (float(value) - target) - k_value))
        c_minus.append(max(0.0, c_minus[-1] + (target - float(value)) - k_value))

    cp = np.asarray(c_plus[1:], dtype=float)
    cm = np.asarray(c_minus[1:], dtype=float)

    violations_plus: List[List[int]] = [[] for _ in range(len(values))]
    violations_minus: List[List[int]] = [[] for _ in range(len(values))]
    for i in range(len(values)):
        if cp[i] > h_value:
            violations_plus[i].append(1)
        if cm[i] > h_value:
            violations_minus[i].append(1)

    extras_plus = [{"original_value": float(values[i])} for i in range(len(values))]
    extras_minus = [{"original_value": float(values[i])} for i in range(len(values))]
    limit_line = [h_value] * len(values)
    zero_line = [0.0] * len(values)

    return {
        "chart_type": "cusum",
        "primary": {
            "name": "CUSUM Positive",
            "points": _points_with_limits(cp, labels, violations_plus, limit_line, zero_line, extras=extras_plus),
            "centerLine": 0.0,
            "UCL": float(h_value),
            "LCL": 0.0,
        },
        "secondary": {
            "name": "CUSUM Negative",
            "points": _points_with_limits(cm, labels, violations_minus, limit_line, zero_line, extras=extras_minus),
            "centerLine": 0.0,
            "UCL": float(h_value),
            "LCL": 0.0,
        },
        "meta": {
            "n_samples": len(labels),
            "value_column": value_column,
            "time_column": time_column,
            "target": float(target),
            "sigma": float(sigma),
            "k": float(k),
            "h": float(h),
            "rules_summary": _summary(violations_plus + violations_minus),
        },
    }


async def _load_pipeline_steps(user_id: str, dataset_id: str, pipeline_id: str) -> List[Dict[str, Any]]:
    row = await registry.fetchrow(
        """
        SELECT steps_json
        FROM pipelines
        WHERE id = $1::uuid
          AND user_id = $2
          AND dataset_id = $3::uuid
        """,
        pipeline_id,
        user_id,
        dataset_id,
    )
    if not row:
        raise ValueError("Pipeline not found (or not owned by user / wrong dataset)")

    steps = row["steps_json"]
    if isinstance(steps, str):
        try:
            steps = json.loads(steps)
        except Exception:
            steps = []

    if not isinstance(steps, list):
        raise ValueError("Pipeline steps are invalid")

    for step in steps:
        op = (step or {}).get("op")
        if not op:
            raise ValueError("Pipeline step missing 'op'")
        transformer_registry.get(op)

    return steps


async def _load_chart_dataframe(
    user_id: str,
    dataset_id: str,
    columns: List[str],
    limit: int,
    order_by: Optional[str] = None,
    where: Optional[str] = None,
    pipeline_id: Optional[str] = None,
):
    parquet = await _get_parquet_local(user_id, dataset_id)
    eng = DuckDBEngine(user_id)
    con = eng.connect()

    try:
        view = eng.register_parquet(con, dataset_id, parquet)
        con.execute(f"CREATE OR REPLACE VIEW dataset_base AS SELECT * FROM {view}")
        current_view = "dataset_base"

        if where and str(where).strip():
            con.execute(
                f"CREATE OR REPLACE VIEW dataset_filtered AS "
                f"SELECT * FROM {current_view} WHERE {where}"
            )
            current_view = "dataset_filtered"

        if pipeline_id:
            steps = await _load_pipeline_steps(user_id, dataset_id, pipeline_id)
            pipeline_sql = compile_pipeline_sql(base_view=current_view, steps=steps)
            con.execute(f"CREATE OR REPLACE VIEW dataset_piped AS {pipeline_sql}")
            current_view = "dataset_piped"

        con.execute(f"CREATE OR REPLACE VIEW dataset AS SELECT * FROM {current_view}")
        quoted_columns = ", ".join(_quote_ident(column) for column in columns)
        order_clause = f"ORDER BY {_quote_ident(order_by)}" if order_by else ""
        df = con.execute(
            f"""
            SELECT {quoted_columns}
            FROM dataset
            {order_clause}
            LIMIT {int(limit)}
            """
        ).fetchdf()
    finally:
        con.close()

    return df


async def run_spc(
    user_id: str,
    dataset_id: str,
    chart_type: str,
    value_column: Optional[str] = None,
    subgroup_column: Optional[str] = None,
    subgroup_size: Optional[int] = None,
    time_column: Optional[str] = None,
    limit: int = 10000,
    where: Optional[str] = None,
    pipeline_id: Optional[str] = None,
    defectives_column: Optional[str] = None,
    sample_size_column: Optional[str] = None,
    sample_size: Optional[int] = None,
    defects_column: Optional[str] = None,
    area_column: Optional[str] = None,
    lambda_param: float = 0.2,
    target: Optional[float] = None,
    sigma: Optional[float] = None,
    k: float = 0.5,
    h: float = 5.0,
) -> Dict[str, Any]:
    if chart_type in {"i-mr", "xbar-r", "xbar-s", "ewma", "cusum"}:
        if not value_column:
            raise ValueError("value_column is required for this chart type")

        requested_columns = [value_column]
        if subgroup_column:
            requested_columns.append(subgroup_column)
        if time_column:
            requested_columns.append(time_column)

        df = await _load_chart_dataframe(
            user_id=user_id,
            dataset_id=dataset_id,
            columns=requested_columns,
            limit=limit,
            order_by=time_column,
            where=where,
            pipeline_id=pipeline_id,
        )
        if df.empty:
            raise ValueError("No data returned for selected columns")

        numeric_values = to_numeric(df[value_column], errors="coerce")
        numeric_mask = np.isfinite(numeric_values.to_numpy(dtype=float, na_value=np.nan))
        df = df.loc[numeric_mask].reset_index(drop=True)
        numeric_values = numeric_values.loc[numeric_mask].reset_index(drop=True)
        if len(df) < 2:
            raise ValueError(f'Column "{value_column}" has fewer than 2 numeric values.')

        values = numeric_values.to_numpy(dtype=float)
        labels = (
            _normalize_label_series(df[time_column])
            if time_column
            else [str(i + 1) for i in range(len(df))]
        )
        keys = df[subgroup_column].tolist() if subgroup_column else None

        if chart_type == "i-mr":
            result = compute_imr(values, labels, value_column, time_column)
        elif chart_type == "xbar-r":
            result = compute_xbar_r(values, labels, value_column, keys, subgroup_size, subgroup_column)
        elif chart_type == "xbar-s":
            result = compute_xbar_s(values, labels, value_column, keys, subgroup_size, subgroup_column)
        elif chart_type == "ewma":
            result = compute_ewma(values, labels, value_column, time_column, lambda_param=lambda_param, target=target, sigma=sigma)
        else:
            result = compute_cusum(values, labels, value_column, time_column, target=target, sigma=sigma, k=k, h=h)

    elif chart_type == "p":
        if not defectives_column:
            raise ValueError("defectives_column is required for p chart")
        if sample_size_column is None and sample_size is None:
            raise ValueError("p chart needs sample_size_column or sample_size")

        requested_columns = [defectives_column]
        if sample_size_column:
            requested_columns.append(sample_size_column)
        if subgroup_column:
            requested_columns.append(subgroup_column)
        if time_column:
            requested_columns.append(time_column)

        df = await _load_chart_dataframe(
            user_id=user_id,
            dataset_id=dataset_id,
            columns=requested_columns,
            limit=limit,
            order_by=time_column,
            where=where,
            pipeline_id=pipeline_id,
        )
        if df.empty:
            raise ValueError("No data returned for selected columns")

        if subgroup_column:
            if sample_size_column:
                grouped = _aggregate_by_subgroup(df, subgroup_column, [defectives_column, sample_size_column], time_column=time_column)
                labels = [row["label"] for row in grouped]
                defectives = np.asarray([row[defectives_column] for row in grouped], dtype=float)
                sample_sizes = np.asarray([row[sample_size_column] for row in grouped], dtype=float)
            else:
                grouped = _aggregate_by_subgroup(df, subgroup_column, [defectives_column], time_column=time_column)
                labels = [row["label"] for row in grouped]
                defectives = np.asarray([row[defectives_column] for row in grouped], dtype=float)
                sample_sizes = np.full(len(grouped), int(sample_size), dtype=float)
        else:
            if sample_size_column:
                defectives_series = to_numeric(df[defectives_column], errors="coerce")
                sample_size_series = to_numeric(df[sample_size_column], errors="coerce")
                mask = (
                    np.isfinite(defectives_series.to_numpy(dtype=float, na_value=np.nan))
                    & np.isfinite(sample_size_series.to_numpy(dtype=float, na_value=np.nan))
                )
                defectives = defectives_series.loc[mask].reset_index(drop=True).to_numpy(dtype=float)
                sample_sizes = sample_size_series.loc[mask].reset_index(drop=True).to_numpy(dtype=float)
            else:
                defectives_series = to_numeric(df[defectives_column], errors="coerce").dropna().reset_index(drop=True)
                defectives = defectives_series.to_numpy(dtype=float)
                sample_sizes = np.full(len(defectives), int(sample_size), dtype=float)
            labels = [str(i + 1) for i in range(len(defectives))]
        result = compute_p_chart(defectives, sample_sizes, labels, defectives_column, sample_size_column, subgroup_column)

    elif chart_type == "np":
        if not defectives_column or sample_size is None:
            raise ValueError("np chart requires defectives_column and sample_size")
        requested_columns = [defectives_column]
        if subgroup_column:
            requested_columns.append(subgroup_column)
        if time_column:
            requested_columns.append(time_column)
        df = await _load_chart_dataframe(
            user_id=user_id,
            dataset_id=dataset_id,
            columns=requested_columns,
            limit=limit,
            order_by=time_column,
            where=where,
            pipeline_id=pipeline_id,
        )
        if df.empty:
            raise ValueError("No data returned for selected columns")
        if subgroup_column:
            grouped = _aggregate_by_subgroup(df, subgroup_column, [defectives_column], time_column=time_column)
            labels = [row["label"] for row in grouped]
            defectives = np.asarray([row[defectives_column] for row in grouped], dtype=float)
        else:
            defectives_series = to_numeric(df[defectives_column], errors="coerce").dropna().reset_index(drop=True)
            defectives = defectives_series.to_numpy(dtype=float)
            labels = [str(i + 1) for i in range(len(defectives))]
        result = compute_np_chart(defectives, int(sample_size), labels, defectives_column, subgroup_column)

    elif chart_type == "c":
        if not defects_column:
            raise ValueError("c chart requires defects_column")
        requested_columns = [defects_column]
        if subgroup_column:
            requested_columns.append(subgroup_column)
        if time_column:
            requested_columns.append(time_column)
        df = await _load_chart_dataframe(
            user_id=user_id,
            dataset_id=dataset_id,
            columns=requested_columns,
            limit=limit,
            order_by=time_column,
            where=where,
            pipeline_id=pipeline_id,
        )
        if df.empty:
            raise ValueError("No data returned for selected columns")
        if subgroup_column:
            grouped = _aggregate_by_subgroup(df, subgroup_column, [defects_column], time_column=time_column)
            labels = [row["label"] for row in grouped]
            defects = np.asarray([row[defects_column] for row in grouped], dtype=float)
        else:
            defects_series = to_numeric(df[defects_column], errors="coerce").dropna().reset_index(drop=True)
            defects = defects_series.to_numpy(dtype=float)
            labels = [str(i + 1) for i in range(len(defects))]
        result = compute_c_chart(defects, labels, defects_column, subgroup_column)

    elif chart_type == "u":
        if not defects_column or not area_column:
            raise ValueError("u chart requires defects_column and area_column")
        requested_columns = [defects_column, area_column]
        if subgroup_column:
            requested_columns.append(subgroup_column)
        if time_column:
            requested_columns.append(time_column)
        df = await _load_chart_dataframe(
            user_id=user_id,
            dataset_id=dataset_id,
            columns=requested_columns,
            limit=limit,
            order_by=time_column,
            where=where,
            pipeline_id=pipeline_id,
        )
        if df.empty:
            raise ValueError("No data returned for selected columns")
        if subgroup_column:
            grouped = _aggregate_by_subgroup(df, subgroup_column, [defects_column, area_column], time_column=time_column)
            labels = [row["label"] for row in grouped]
            defects = np.asarray([row[defects_column] for row in grouped], dtype=float)
            areas = np.asarray([row[area_column] for row in grouped], dtype=float)
        else:
            defects_series = to_numeric(df[defects_column], errors="coerce")
            area_series = to_numeric(df[area_column], errors="coerce")
            mask = (
                np.isfinite(defects_series.to_numpy(dtype=float, na_value=np.nan))
                & np.isfinite(area_series.to_numpy(dtype=float, na_value=np.nan))
            )
            defects = defects_series.loc[mask].reset_index(drop=True).to_numpy(dtype=float)
            areas = area_series.loc[mask].reset_index(drop=True).to_numpy(dtype=float)
            labels = [str(i + 1) for i in range(len(defects))]
        result = compute_u_chart(defects, areas, labels, defects_column, area_column, subgroup_column)

    else:
        raise ValueError(f"Unsupported SPC chart type: {chart_type}")

    result["meta"]["limit"] = int(limit)
    result["meta"]["where"] = (where or "").strip() or None
    result["meta"]["pipeline_id"] = pipeline_id
    return result
