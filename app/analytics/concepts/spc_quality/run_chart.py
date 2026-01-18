from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='run-chart-001',
    topic_id='quality-tools-topic',
    topic_slug='spc-quality',
    slug='run-chart',
    title='Run Chart (Time Series with Runs Analysis)',
    concept_type='quality_tool',
    level='intro',
    status='published',
    output_keys=['run_chart'],
    tags=['spc', 'quality', 'trending'],
    quality_score=90,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate run chart with runs analysis for detecting non-random patterns."""
    import numpy as np
    
    measure_column = params.get('measure_column')
    time_column = params.get('time_column')
    
    if not measure_column:
        raise ValueError('measure_column required')
    
    # Get data in time order
    if time_column:
        query = f"SELECT {measure_column} FROM dataset WHERE {measure_column} IS NOT NULL ORDER BY {time_column}"
    else:
        query = f"SELECT {measure_column} FROM dataset WHERE {measure_column} IS NOT NULL ORDER BY rowid"
    
    values = np.array([r[0] for r in ctx.con.execute(query).fetchall()])
    
    if len(values) < 3:
        return {'error': 'Need at least 3 observations'}
    
    # Calculate center line (median for run chart)
    center_line = np.median(values)
    
    # Count runs (sequences above/below median)
    above_median = values > center_line
    runs = 1
    for i in range(1, len(above_median)):
        if above_median[i] != above_median[i-1]:
            runs += 1
    
    # Expected number of runs and standard deviation
    n = len(values)
    n_above = sum(above_median)
    n_below = n - n_above
    
    expected_runs = ((2 * n_above * n_below) / n) + 1
    sigma_runs = np.sqrt((2 * n_above * n_below * (2 * n_above * n_below - n)) / (n**2 * (n - 1)))
    
    # Test for non-randomness
    z_runs = (runs - expected_runs) / sigma_runs if sigma_runs > 0 else 0
    p_value_runs = 2 * (1 - abs(z_runs))  # Two-tailed
    
    # Detect patterns
    patterns = []
    
    # Pattern 1: Too few runs (suggests trend or mixture)
    if runs < expected_runs - 2 * sigma_runs:
        patterns.append('Too few runs - suggests trend or systematic pattern')
    
    # Pattern 2: Too many runs (suggests oscillation)
    if runs > expected_runs + 2 * sigma_runs:
        patterns.append('Too many runs - suggests oscillation or over-control')
    
    # Pattern 3: Long runs above or below median
    max_run_length = 0
    current_run = 1
    for i in range(1, len(above_median)):
        if above_median[i] == above_median[i-1]:
            current_run += 1
            max_run_length = max(max_run_length, current_run)
        else:
            current_run = 1
    
    expected_max_run = int(np.log2(n)) + 3
    if max_run_length > expected_max_run:
        patterns.append(f'Unusually long run detected ({max_run_length} points)')
    
    return {
        'center_line': float(center_line),
        'values': values.tolist(),
        'n': n,
        'n_runs': runs,
        'expected_runs': float(expected_runs),
        'sigma_runs': float(sigma_runs),
        'z_score': float(z_runs),
        'random_pattern': abs(z_runs) < 2,
        'patterns_detected': patterns,
        'max_run_length': max_run_length,
        'n_above_median': int(n_above),
        'n_below_median': int(n_below),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
