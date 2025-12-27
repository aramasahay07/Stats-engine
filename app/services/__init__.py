"""
Service layer for AI Data Lab
"""

# Import core services
try:
    from .compute import compute_service
except ImportError as e:
    print(f"Warning: Could not import compute service: {e}")
    compute_service = None

try:
    from .dataset_registry import DatasetRegistry
except ImportError as e:
    print(f"Warning: Could not import DatasetRegistry: {e}")
    DatasetRegistry = None

try:
    from .storage_client import SupabaseStorageClient
except ImportError as e:
    print(f"Warning: Could not import SupabaseStorageClient: {e}")
    SupabaseStorageClient = None

try:
    from .cache_paths import CachePaths
except ImportError as e:
    print(f"Warning: Could not import CachePaths: {e}")
    CachePaths = None

try:
    from .duckdb_manager import DuckDBManager
except ImportError as e:
    print(f"Warning: Could not import DuckDBManager: {e}")
    DuckDBManager = None

__all__ = [
    'compute_service',
    'DatasetRegistry',
    'SupabaseStorageClient',
    'CachePaths',
    'DuckDBManager'
]
