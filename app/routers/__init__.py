"""
API Routers for AI Data Lab
"""

# Import all routers
try:
    from .datasets import router as datasets_router
except ImportError as e:
    print(f"Warning: Could not import datasets router: {e}")
    datasets_router = None

try:
    from .stats import router as stats_router
except ImportError as e:
    print(f"Warning: Could not import stats router: {e}")
    stats_router = None

try:
    from .transforms import router as transforms_router
except ImportError as e:
    print(f"Warning: Could not import transforms router: {e}")
    transforms_router = None

try:
    from .agents import router as agents_router
except ImportError as e:
    print(f"Warning: Could not import agents router: {e}")
    agents_router = None

try:
    from .quality import router as quality_router
except ImportError as e:
    print(f"Warning: Could not import quality router: {e}")
    quality_router = None

# Export available routers
__all__ = [
    'datasets_router',
    'stats_router',
    'transforms_router',
    'agents_router',
    'quality_router'
]
