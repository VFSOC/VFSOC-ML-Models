"""
VFSOC ML Models - GPS Jamming Detection

A comprehensive machine learning framework for detecting GPS jamming attacks
in vehicle fleet management systems.
"""

__version__ = "0.1.0"
__author__ = "VFSOC Team"
__email__ = "vfsoc@example.com"

# Core imports for easy access
from .models.isolation_forest import IsolationForestDetector

from .data.data_loader import VFSOCDataLoader
from .data.synthetic_generator import SyntheticDataGenerator
from .deployment.onnx_converter import ONNXConverter

__all__ = [
    # Models
    "IsolationForestDetector",
    
    # Data handling
    "VFSOCDataLoader",
    "SyntheticDataGenerator",
    
    # Deployment
    "ONNXConverter",
] 