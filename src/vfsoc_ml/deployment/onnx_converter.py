"""
ONNX converter for GPS jamming detection models.

This module provides functionality to convert trained scikit-learn models
to ONNX format for production deployment and faster inference.
"""

from typing import Union, Optional, Dict, Any
from pathlib import Path
import logging
import numpy as np
import joblib

try:
    import onnx
    import onnxruntime as ort
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

logger = logging.getLogger(__name__)


class ONNXConverter:
    """
    Converter for transforming trained models to ONNX format.
    
    Supports conversion of scikit-learn based models including:
    - Isolation Forest
    - Random Forest
    - SVM models
    - Preprocessing pipelines
    """
    
    def __init__(self):
        """Initialize the ONNX converter."""
        if not ONNX_AVAILABLE:
            raise ImportError(
                "ONNX dependencies not available. "
                "Install with: pip install onnx onnxruntime skl2onnx"
            )
        
        self.supported_models = {
            'IsolationForest',
            'RandomForestClassifier', 
            'SVC',
            'OneClassSVM'
        }
    
    def convert_model(self, 
                     model_path: Union[str, Path],
                     output_path: Union[str, Path],
                     input_shape: Optional[tuple] = None,
                     model_name: str = "GPSJammingDetector") -> str:
        """
        Convert a trained model to ONNX format.
        
        Args:
            model_path: Path to the trained model (.pkl file)
            output_path: Path to save the ONNX model
            input_shape: Shape of input features (n_features,)
            model_name: Name for the ONNX model
            
        Returns:
            Path to the saved ONNX model
        """
        model_path = Path(model_path)
        output_path = Path(output_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load the trained model
        logger.info(f"Loading model from {model_path}")
        model_data = joblib.load(model_path)
        
        if isinstance(model_data, dict):
            sklearn_model = model_data['model']
            feature_names = model_data.get('feature_names', None)
        else:
            sklearn_model = model_data
            feature_names = None
        
        # Determine input shape
        if input_shape is None:
            if hasattr(sklearn_model, 'n_features_in_'):
                n_features = sklearn_model.n_features_in_
            elif feature_names:
                n_features = len(feature_names)
            else:
                raise ValueError("Cannot determine input shape. Please provide input_shape parameter.")
        else:
            n_features = input_shape[0]
        
        logger.info(f"Converting model with {n_features} input features")
        
        # Check if model is supported
        model_type = type(sklearn_model).__name__
        if model_type not in self.supported_models:
            logger.warning(f"Model type {model_type} may not be fully supported")
        
        # Define initial input type
        initial_type = [('float_input', FloatTensorType([None, n_features]))]
        
        try:
            # Convert to ONNX
            onnx_model = convert_sklearn(
                sklearn_model,
                initial_types=initial_type,
                target_opset=12,
                verbose=0
            )
            
            # Set model metadata
            onnx_model.doc_string = f"GPS Jamming Detection Model - {model_type}"
            onnx_model.model_version = 1
            
            # Save ONNX model
            output_path.parent.mkdir(parents=True, exist_ok=True)
            onnx.save_model(onnx_model, str(output_path))
            
            logger.info(f"Model converted successfully to {output_path}")
            
            # Validate the converted model
            self._validate_onnx_model(output_path, n_features)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to convert model: {e}")
            raise
    
    def _validate_onnx_model(self, onnx_path: Path, n_features: int) -> None:
        """
        Validate the converted ONNX model.
        
        Args:
            onnx_path: Path to the ONNX model
            n_features: Number of input features
        """
        try:
            # Load ONNX model
            onnx_model = onnx.load(str(onnx_path))
            
            # Check model validity
            onnx.checker.check_model(onnx_model)
            
            # Test inference with random data
            session = ort.InferenceSession(str(onnx_path))
            
            # Get input name
            input_name = session.get_inputs()[0].name
            
            # Create test input
            test_input = np.random.random((1, n_features)).astype(np.float32)
            
            # Run inference
            result = session.run(None, {input_name: test_input})
            
            logger.info(f"ONNX model validation successful. Output shape: {[r.shape for r in result]}")
            
        except Exception as e:
            logger.warning(f"ONNX model validation failed: {e}")
    
    def benchmark_model(self, 
                       onnx_path: Union[str, Path],
                       original_model_path: Union[str, Path],
                       n_samples: int = 1000,
                       n_features: Optional[int] = None) -> Dict[str, Any]:
        """
        Benchmark ONNX model against original model.
        
        Args:
            onnx_path: Path to ONNX model
            original_model_path: Path to original model
            n_samples: Number of test samples
            n_features: Number of features (auto-detect if None)
            
        Returns:
            Benchmark results dictionary
        """
        import time
        
        # Load original model
        original_model_data = joblib.load(original_model_path)
        if isinstance(original_model_data, dict):
            original_model = original_model_data['model']
            if n_features is None:
                n_features = len(original_model_data.get('feature_names', []))
        else:
            original_model = original_model_data
            if n_features is None:
                n_features = original_model.n_features_in_
        
        # Load ONNX model
        onnx_session = ort.InferenceSession(str(onnx_path))
        input_name = onnx_session.get_inputs()[0].name
        
        # Generate test data
        test_data = np.random.random((n_samples, n_features)).astype(np.float32)
        
        # Benchmark original model
        start_time = time.time()
        original_predictions = original_model.predict(test_data)
        original_time = time.time() - start_time
        
        # Benchmark ONNX model
        start_time = time.time()
        onnx_predictions = onnx_session.run(None, {input_name: test_data})[0]
        onnx_time = time.time() - start_time
        
        # Compare predictions
        if len(onnx_predictions.shape) > 1:
            # Convert multi-output to single prediction
            onnx_predictions = np.argmax(onnx_predictions, axis=1) * 2 - 1  # Convert 0,1 to -1,1
        
        # Calculate accuracy between models
        accuracy = np.mean(original_predictions == onnx_predictions.flatten())
        
        results = {
            'original_time_sec': original_time,
            'onnx_time_sec': onnx_time,
            'speedup': original_time / onnx_time,
            'prediction_accuracy': accuracy,
            'original_throughput': n_samples / original_time,
            'onnx_throughput': n_samples / onnx_time,
            'n_samples': n_samples,
            'n_features': n_features
        }
        
        logger.info(f"Benchmark results: {results}")
        return results
    
    def get_model_info(self, onnx_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about an ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
            
        Returns:
            Model information dictionary
        """
        onnx_model = onnx.load(str(onnx_path))
        
        # Get input/output info
        inputs = []
        for input_tensor in onnx_model.graph.input:
            shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
            inputs.append({
                'name': input_tensor.name,
                'type': input_tensor.type.tensor_type.elem_type,
                'shape': shape
            })
        
        outputs = []
        for output_tensor in onnx_model.graph.output:
            shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
            outputs.append({
                'name': output_tensor.name,
                'type': output_tensor.type.tensor_type.elem_type,
                'shape': shape
            })
        
        return {
            'model_version': onnx_model.model_version,
            'doc_string': onnx_model.doc_string,
            'producer_name': onnx_model.producer_name,
            'producer_version': onnx_model.producer_version,
            'inputs': inputs,
            'outputs': outputs,
            'opset_version': onnx_model.opset_import[0].version if onnx_model.opset_import else None
        } 