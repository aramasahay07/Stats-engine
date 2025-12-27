"""
Session management for data frames and transform caching
"""
from typing import Dict, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta


class SessionStore:
    """In-memory session storage with TTL and caching"""
    
    def __init__(self, ttl_hours: int = 24):
        self.sessions: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, dict] = {}
        self.transform_cache: Dict[str, pd.Series] = {}
        self.ttl_hours = ttl_hours
    
    def set(self, session_id: str, df: pd.DataFrame, metadata: Optional[dict] = None):
        """Store dataframe with optional metadata"""
        self.sessions[session_id] = df
        self.metadata[session_id] = {
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "row_count": len(df),
            "col_count": len(df.columns),
            **(metadata or {})
        }
    
    def get(self, session_id: str) -> Optional[pd.DataFrame]:
        """Retrieve dataframe and update access time"""
        if session_id in self.sessions:
            self.metadata[session_id]["last_accessed"] = datetime.now()
            return self.sessions[session_id]
        return None
    
    def get_metadata(self, session_id: str) -> Optional[dict]:
        """Get session metadata"""
        return self.metadata.get(session_id)
    
    def exists(self, session_id: str) -> bool:
        """Check if session exists"""
        return session_id in self.sessions
    
    def delete(self, session_id: str):
        """Delete session and associated cache"""
        if session_id in self.sessions:
            del self.sessions[session_id]
        if session_id in self.metadata:
            del self.metadata[session_id]
        
        # Clear related transform cache
        cache_keys = [k for k in self.transform_cache.keys() if k.startswith(f"{session_id}:")]
        for key in cache_keys:
            del self.transform_cache[key]
    
    def cleanup_expired(self):
        """Remove expired sessions"""
        now = datetime.now()
        expired = []
        
        for session_id, meta in self.metadata.items():
            if now - meta["last_accessed"] > timedelta(hours=self.ttl_hours):
                expired.append(session_id)
        
        for session_id in expired:
            self.delete(session_id)
        
        return len(expired)
    
    def cache_transform(self, session_id: str, column: str, transform_key: str, result: pd.Series):
        """Cache transformed column"""
        cache_key = f"{session_id}:{column}:{transform_key}"
        self.transform_cache[cache_key] = result
    
    def get_cached_transform(self, session_id: str, column: str, transform_key: str) -> Optional[pd.Series]:
        """Retrieve cached transform"""
        cache_key = f"{session_id}:{column}:{transform_key}"
        return self.transform_cache.get(cache_key)
    
    def clear_cache(self, session_id: Optional[str] = None):
        """Clear transform cache for session or all"""
        if session_id:
            cache_keys = [k for k in self.transform_cache.keys() if k.startswith(f"{session_id}:")]
            for key in cache_keys:
                del self.transform_cache[key]
        else:
            self.transform_cache.clear()
    
    def get_stats(self) -> dict:
        """Get storage statistics"""
        return {
            "active_sessions": len(self.sessions),
            "cached_transforms": len(self.transform_cache),
            "total_memory_mb": sum(
                df.memory_usage(deep=True).sum() / (1024 * 1024)
                for df in self.sessions.values()
            )
        }


# Global instance
store = SessionStore()
