"""
VFSOC ML - EV Charging Anomaly Detection

Machine learning model for detecting irregular energy consumption patterns
in EV charging stations.
"""

__version__ = "1.0.0"
__author__ = "VFSOC Team"

# Import main components
from .models.ev_charging_anomaly_detector import EVChargingAnomalyDetector

__all__ = [
    "EVChargingAnomalyDetector",
] 