"""
Core Orchestrator - Base classes and shared memory for AI agents
Simplified version for agent endpoints architecture
"""
from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime
import pandas as pd


class AgentRole(Enum):
    """Agent role identifiers"""
    DATA_EXPLORER = "data_explorer"
    PATTERN_DETECTIVE = "pattern_detective"
    CAUSAL_REASONER = "causal_reasoner"
    VALIDATOR = "validator"
    STORYTELLER = "storyteller"
    RECOMMENDER = "recommender"


class BaseAgent:
    """
    Base class for all AI agents
    Provides common functionality and interface
    """
    
    def __init__(self, role: AgentRole):
        self.role = role
        self.name = role.value
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent's main functionality
        
        Args:
            inputs: Dictionary containing required inputs
        
        Returns:
            Dictionary containing agent's results
        """
        raise NotImplementedError("Subclasses must implement execute()")
    
    def log(self, message: str, level: str = "INFO"):
        """Simple logging"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{self.name}] [{level}] {message}")


class SharedMemory:
    """
    Simplified shared memory for agent communication
    In production, this would be Redis or database
    """
    
    def __init__(self):
        self.short_term: Dict[str, Any] = {}
        self.working: Dict[str, Any] = {}
        self.long_term: Dict[str, Any] = {}
    
    def store(self, key: str, value: Any, memory_type: str = "working"):
        """Store data in specified memory type"""
        if memory_type == "short_term":
            self.short_term[key] = value
        elif memory_type == "working":
            self.working[key] = value
        elif memory_type == "long_term":
            self.long_term[key] = value
    
    def retrieve(self, key: str, memory_type: str = "working") -> Optional[Any]:
        """Retrieve data from specified memory type"""
        if memory_type == "short_term":
            return self.short_term.get(key)
        elif memory_type == "working":
            return self.working.get(key)
        elif memory_type == "long_term":
            return self.long_term.get(key)
        return None
    
    def clear(self, memory_type: Optional[str] = None):
        """Clear specified memory type or all"""
        if memory_type == "short_term":
            self.short_term.clear()
        elif memory_type == "working":
            self.working.clear()
        elif memory_type == "long_term":
            self.long_term.clear()
        else:
            self.short_term.clear()
            self.working.clear()
            self.long_term.clear()


# Utility functions for agents

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division with default value"""
    try:
        return a / b if b != 0 else default
    except:
        return default


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage"""
    return f"{value * 100:.{decimals}f}%"


def calculate_confidence(
    sample_size: int,
    p_value: float,
    effect_size: float
) -> str:
    """
    Calculate confidence level based on statistical metrics
    
    Returns: 'high', 'medium', or 'low'
    """
    if sample_size < 30:
        return "low"
    
    if p_value < 0.01 and effect_size > 0.5:
        return "high"
    elif p_value < 0.05 and effect_size > 0.3:
        return "medium"
    else:
        return "low"
