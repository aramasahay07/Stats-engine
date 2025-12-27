"""Statistical Process Control - 25 Minitab concepts."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats

class SPCAnalysis:
    """
    Concepts covered:
    1-3. X-bar chart, R chart, S chart
    4-6. I-MR chart (Individuals, Moving Range)
    7-9. P chart, NP chart, C chart
    10-11. U chart, Laney P' chart
    12-14. CUSUM chart, EWMA chart, Zone tests
    15-17. Cp, Cpk (Process Capability)
    18-20. Pp, Ppk (Process Performance)
    21. Cpm (Taguchi capability)
    22. Process Sigma / DPMO
    23. Gage R&R
    24. Measurement System Analysis
    25. Process Capability Sixpack
    """
    
    @staticmethod
    def xbar_r_chart(subgroups: List[np.ndarray]) -> Dict[str, Any]:
        """X-bar and R control charts for subgrouped data."""
        subgroup_means = [np.mean(sg) for sg in subgroups]
        subgroup_ranges = [np.ptp(sg) for sg in subgroups]
        
        xbar = np.mean(subgroup_means)
        rbar = np.mean(subgroup_ranges)
        
        # A2 and D3, D4 constants depend on subgroup size
        n = len(subgroups[0])  # Assuming equal subgroup sizes
        
        # Control chart constants (simplified for n=2-10)
        A2_table = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
        D3_table = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
        D4_table = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}
        
        A2 = A2_table.get(n, 0.577)
        D3 = D3_table.get(n, 0)
        D4 = D4_table.get(n, 2.114)
        
        # X-bar chart limits
        xbar_ucl = xbar + A2 * rbar
        xbar_lcl = xbar - A2 * rbar
        
        # R chart limits
        r_ucl = D4 * rbar
        r_lcl = D3 * rbar
        
        # Detect out-of-control points
        xbar_ooc = [i for i, m in enumerate(subgroup_means) if m > xbar_ucl or m < xbar_lcl]
        r_ooc = [i for i, r in enumerate(subgroup_ranges) if r > r_ucl or r < r_lcl]
        
        return {
            "chart_type": "xbar_r",
            "subgroup_size": n,
            "n_subgroups": len(subgroups),
            "xbar_chart": {
                "center_line": float(xbar),
                "ucl": float(xbar_ucl),
                "lcl": float(xbar_lcl),
                "values": [float(m) for m in subgroup_means],
                "out_of_control_points": xbar_ooc,
            },
            "r_chart": {
                "center_line": float(rbar),
                "ucl": float(r_ucl),
                "lcl": float(r_lcl),
                "values": [float(r) for r in subgroup_ranges],
                "out_of_control_points": r_ooc,
            },
            "process_in_control": len(xbar_ooc) == 0 and len(r_ooc) == 0,
        }
    
    @staticmethod
    def imr_chart(data: np.ndarray) -> Dict[str, Any]:
        """I-MR chart (Individuals and Moving Range)."""
        clean = data[~np.isnan(data)]
        n = len(clean)
        
        # Moving ranges
        mr = np.abs(np.diff(clean))
        mr_bar = np.mean(mr)
        
        # Constants for n=2 (moving range of 2 consecutive points)
        d2 = 1.128
        D3 = 0
        D4 = 3.267
        
        # Estimate sigma
        sigma_est = mr_bar / d2
        
        # I chart limits
        xbar = np.mean(clean)
        i_ucl = xbar + 3 * sigma_est
        i_lcl = xbar - 3 * sigma_est
        
        # MR chart limits
        mr_ucl = D4 * mr_bar
        mr_lcl = D3 * mr_bar
        
        # Out of control points
        i_ooc = [i for i, v in enumerate(clean) if v > i_ucl or v < i_lcl]
        mr_ooc = [i for i, v in enumerate(mr) if v > mr_ucl]
        
        return {
            "chart_type": "imr",
            "n": n,
            "i_chart": {
                "center_line": float(xbar),
                "ucl": float(i_ucl),
                "lcl": float(i_lcl),
                "values": clean.tolist(),
                "out_of_control_points": i_ooc,
            },
            "mr_chart": {
                "center_line": float(mr_bar),
                "ucl": float(mr_ucl),
                "lcl": float(mr_lcl),
                "values": mr.tolist(),
                "out_of_control_points": mr_ooc,
            },
            "sigma_estimate": float(sigma_est),
            "process_in_control": len(i_ooc) == 0 and len(mr_ooc) == 0,
        }
    
    @staticmethod
    def process_capability(data: np.ndarray, lsl: float, usl: float,
                           target: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate Cp, Cpk, Pp, Ppk, Cpm.
        
        Within-subgroup variation (Cp, Cpk): Use R-bar/d2 for sigma
        Overall variation (Pp, Ppk): Use sample standard deviation
        """
        clean = data[~np.isnan(data)]
        n = len(clean)
        
        mean = np.mean(clean)
        std_overall = np.std(clean, ddof=1)  # Overall (long-term) sigma
        
        # For within-subgroup, use moving range estimate
        mr = np.abs(np.diff(clean))
        sigma_within = np.mean(mr) / 1.128  # d2 for n=2
        
        spec_range = usl - lsl
        
        if target is None:
            target = (usl + lsl) / 2
        
        # Capability indices (within-subgroup variation)
        cp = spec_range / (6 * sigma_within) if sigma_within > 0 else np.inf
        cpu = (usl - mean) / (3 * sigma_within) if sigma_within > 0 else np.inf
        cpl = (mean - lsl) / (3 * sigma_within) if sigma_within > 0 else np.inf
        cpk = min(cpu, cpl)
        
        # Performance indices (overall variation)
        pp = spec_range / (6 * std_overall) if std_overall > 0 else np.inf
        ppu = (usl - mean) / (3 * std_overall) if std_overall > 0 else np.inf
        ppl = (mean - lsl) / (3 * std_overall) if std_overall > 0 else np.inf
        ppk = min(ppu, ppl)
        
        # Cpm (Taguchi)
        tau = np.sqrt(std_overall**2 + (mean - target)**2)
        cpm = spec_range / (6 * tau) if tau > 0 else np.inf
        
        # DPMO and Process Sigma
        z_upper = (usl - mean) / std_overall if std_overall > 0 else np.inf
        z_lower = (mean - lsl) / std_overall if std_overall > 0 else np.inf
        
        ppm_above = (1 - stats.norm.cdf(z_upper)) * 1_000_000
        ppm_below = stats.norm.cdf(-z_lower) * 1_000_000
        dpmo = ppm_above + ppm_below
        
        # Process sigma level (with 1.5 sigma shift)
        process_sigma = stats.norm.ppf(1 - dpmo/1_000_000) + 1.5 if dpmo < 1_000_000 else 0
        
        return {
            "n": n,
            "mean": float(mean),
            "std_overall": float(std_overall),
            "std_within": float(sigma_within),
            "lsl": lsl,
            "usl": usl,
            "target": target,
            "capability_indices": {
                "cp": float(cp),
                "cpl": float(cpl),
                "cpu": float(cpu),
                "cpk": float(cpk),
                "cpk_interpretation": "Excellent" if cpk >= 1.33 else "Capable" if cpk >= 1.0 else "Not Capable",
            },
            "performance_indices": {
                "pp": float(pp),
                "ppl": float(ppl),
                "ppu": float(ppu),
                "ppk": float(ppk),
            },
            "taguchi": {
                "cpm": float(cpm),
            },
            "six_sigma": {
                "dpmo": float(dpmo),
                "process_sigma": float(process_sigma),
                "yield_percent": float((1 - dpmo/1_000_000) * 100),
            },
            "out_of_spec": {
                "ppm_below_lsl": float(ppm_below),
                "ppm_above_usl": float(ppm_above),
                "total_ppm": float(dpmo),
            }
        }
