from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='oee-001',
    topic_id='quality-tools-topic',
    topic_slug='spc-quality',
    slug='oee',
    title='OEE (Overall Equipment Effectiveness)',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['oee', 'overall_equipment_effectiveness'],
    tags=['quality', 'manufacturing', 'efficiency'],
    quality_score=95,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate OEE - combination of Availability, Performance, and Quality."""
    # OEE = Availability × Performance × Quality
    
    # Method 1: Provide individual components
    availability = params.get('availability')
    performance = params.get('performance')
    quality = params.get('quality')
    
    # Method 2: Calculate from raw data
    planned_production_time = params.get('planned_production_time')
    actual_run_time = params.get('actual_run_time')
    ideal_cycle_time = params.get('ideal_cycle_time')
    total_pieces = params.get('total_pieces')
    good_pieces = params.get('good_pieces')
    
    if all(x is not None for x in [availability, performance, quality]):
        # Direct calculation
        oee = availability * performance * quality
        
    elif all(x is not None for x in [planned_production_time, actual_run_time, ideal_cycle_time, total_pieces, good_pieces]):
        # Calculate from components
        
        # Availability = (Run Time) / (Planned Production Time)
        availability = actual_run_time / planned_production_time if planned_production_time > 0 else 0
        
        # Performance = (Ideal Cycle Time × Total Count) / Run Time
        performance = (ideal_cycle_time * total_pieces) / actual_run_time if actual_run_time > 0 else 0
        
        # Quality = Good Count / Total Count
        quality = good_pieces / total_pieces if total_pieces > 0 else 0
        
        # OEE
        oee = availability * performance * quality
        
    else:
        raise ValueError('Need either (availability, performance, quality) or (planned_production_time, actual_run_time, ideal_cycle_time, total_pieces, good_pieces)')
    
    # Interpret OEE
    if oee >= 0.85:
        classification = 'World Class'
        interpretation = 'Excellent OEE (≥85%)'
    elif oee >= 0.60:
        classification = 'Good'
        interpretation = 'Good OEE (60-85%)'
    elif oee >= 0.40:
        classification = 'Fair'
        interpretation = 'Fair OEE (40-60%) - improvement needed'
    else:
        classification = 'Poor'
        interpretation = 'Poor OEE (<40%) - significant improvement needed'
    
    # Calculate losses
    availability_loss = 1 - availability
    performance_loss = 1 - performance
    quality_loss = 1 - quality
    total_loss = 1 - oee
    
    return {
        'oee': float(oee),
        'oee_percent': float(oee * 100),
        'availability': float(availability),
        'availability_percent': float(availability * 100),
        'performance': float(performance),
        'performance_percent': float(performance * 100),
        'quality': float(quality),
        'quality_percent': float(quality * 100),
        'availability_loss_percent': float(availability_loss * 100),
        'performance_loss_percent': float(performance_loss * 100),
        'quality_loss_percent': float(quality_loss * 100),
        'total_loss_percent': float(total_loss * 100),
        'classification': classification,
        'interpretation': interpretation,
        'world_class': oee >= 0.85,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
