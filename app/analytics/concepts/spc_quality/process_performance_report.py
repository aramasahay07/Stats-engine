from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='process-report-001',
    topic_id='quality-tools-topic',
    topic_slug='spc-quality',
    slug='process-performance-report',
    title='Process Performance Report (Comprehensive Summary)',
    concept_type='report',
    level='intermediate',
    status='published',
    output_keys=['process_report', 'performance_report'],
    tags=['spc', 'quality', 'summary'],
    quality_score=90,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive process performance report."""
    import numpy as np
    from scipy import stats
    
    measure_column = params.get('measure_column')
    usl = params.get('usl')
    lsl = params.get('lsl')
    target = params.get('target')
    
    if not measure_column:
        raise ValueError('measure_column required')
    
    query = f"SELECT {measure_column} FROM dataset WHERE {measure_column} IS NOT NULL"
    values = np.array([r[0] for r in ctx.con.execute(query).fetchall()])
    
    if len(values) < 2:
        return {'error': 'Insufficient data'}
    
    # Descriptive statistics
    mean = np.mean(values)
    median = np.median(values)
    std = np.std(values, ddof=1)
    min_val = np.min(values)
    max_val = np.max(values)
    range_val = max_val - min_val
    
    # Process capability (if spec limits provided)
    if usl and lsl:
        # Short-term sigma
        mr = np.abs(np.diff(values))
        sigma_st = np.mean(mr) / 1.128
        
        # Capability indices
        cp = (usl - lsl) / (6 * sigma_st)
        cpu = (usl - mean) / (3 * sigma_st)
        cpl = (mean - lsl) / (3 * sigma_st)
        cpk = min(cpu, cpl)
        
        # Performance indices (long-term)
        pp = (usl - lsl) / (6 * std)
        ppu = (usl - mean) / (3 * std)
        ppl = (mean - lsl) / (3 * std)
        ppk = min(ppu, ppl)
        
        # Defect rates
        ppm_upper = (1 - stats.norm.cdf((usl - mean) / sigma_st)) * 1e6
        ppm_lower = stats.norm.cdf((lsl - mean) / sigma_st) * 1e6
        ppm_total = ppm_upper + ppm_lower
        
        capability_summary = {
            'cp': float(cp),
            'cpk': float(cpk),
            'pp': float(pp),
            'ppk': float(ppk),
            'ppm_expected': float(ppm_total),
            'sigma_level': float(cpk * 3),
            'capable': cpk >= 1.33,
        }
    else:
        capability_summary = None
    
    # Normality test
    _, p_normality = stats.shapiro(values) if len(values) <= 5000 else stats.normaltest(values)
    
    # Stability indicators
    n_sigma_3 = sum(np.abs((values - mean) / std) > 3)
    
    report = {
        'descriptive_statistics': {
            'n': len(values),
            'mean': float(mean),
            'median': float(median),
            'std': float(std),
            'min': float(min_val),
            'max': float(max_val),
            'range': float(range_val),
        },
        'normality': {
            'p_value': float(p_normality),
            'normal': p_normality > 0.05,
        },
        'stability': {
            'n_beyond_3sigma': int(n_sigma_3),
            'percent_beyond_3sigma': float(n_sigma_3 / len(values) * 100),
        }
    }
    
    if capability_summary:
        report['process_capability'] = capability_summary
        if usl:
            report['specifications'] = {'usl': float(usl)}
        if lsl:
            if 'specifications' not in report:
                report['specifications'] = {}
            report['specifications']['lsl'] = float(lsl)
        if target:
            report['specifications']['target'] = float(target)
    
    return report

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
