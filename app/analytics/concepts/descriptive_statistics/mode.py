from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='mode-001',
    topic_id='desc-stats-topic',
    topic_slug='descriptive-statistics',
    slug='mode',
    title='Mode (Most Frequent Value)',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['mode', 'modal_value'],
    tags=['descriptive', 'central_tendency'],
    quality_score=90,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate mode with frequency analysis."""
    column = params.get('column', params.get('measure_column'))
    
    if not column:
        raise ValueError('column parameter required')
    
    # Get mode and frequency
    query = f"""
        SELECT 
            MODE() WITHIN GROUP (ORDER BY {column}) as mode,
            COUNT({column}) as n
        FROM dataset 
        WHERE {column} IS NOT NULL
    """
    
    result = ctx.con.execute(query).fetchone()
    mode_val = result[0]
    n = int(result[1])
    
    # Get frequency distribution (top 10)
    freq_query = f"""
        SELECT {column}, COUNT(*) as frequency
        FROM dataset
        WHERE {column} IS NOT NULL
        GROUP BY {column}
        ORDER BY frequency DESC
        LIMIT 10
    """
    
    freq_data = ctx.con.execute(freq_query).fetchall()
    
    if not freq_data:
        return {'error': 'No valid data', 'n': n}
    
    # Check for multimodal
    max_freq = freq_data[0][1]
    modes = [row[0] for row in freq_data if row[1] == max_freq]
    
    # Get mode frequency
    mode_frequency = max_freq
    mode_percent = (mode_frequency / n * 100) if n > 0 else 0
    
    return {
        'mode': mode_val,
        'modal_value': mode_val,
        'mode_frequency': int(mode_frequency),
        'mode_percent': float(mode_percent),
        'is_multimodal': len(modes) > 1,
        'n_modes': len(modes),
        'all_modes': modes if len(modes) > 1 else [mode_val],
        'n': n,
        'frequency_table': [
            {'value': row[0], 'frequency': int(row[1]), 'percent': float(row[1]/n*100)}
            for row in freq_data[:5]
        ],
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
