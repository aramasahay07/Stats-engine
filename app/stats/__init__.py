"""Statistical analysis modules for Minitab-equivalent capabilities."""

from .descriptive import DescriptiveStats
from .hypothesis import HypothesisTesting
from .anova import AnovaAnalysis
from .regression import RegressionAnalysis
from .spc import SPCAnalysis
from .nonparametric import NonParametricTests
from .multivariate import MultivariateAnalysis

__all__ = [
    "DescriptiveStats",
    "HypothesisTesting", 
    "AnovaAnalysis",
    "RegressionAnalysis",
    "SPCAnalysis",
    "NonParametricTests",
    "MultivariateAnalysis",
]
