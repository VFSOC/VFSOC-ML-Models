"""
VFSOC ML - Irregular Energy Consumption Detection

Machine learning models for detecting irregular energy consumption patterns
in EV charging stations.
"""

__version__ = "1.0.0"
__author__ = "VFSOC Team"

# Import main components
from .data.data_loader import EnergyConsumptionDataLoader
from .data.feature_engineering import EnergyConsumptionFeatureEngineer

__all__ = [
    "EnergyConsumptionDataLoader",
    "EnergyConsumptionFeatureEngineer",
] 