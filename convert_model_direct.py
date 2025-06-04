#!/usr/bin/env python3
"""Direct ONNX conversion script."""

import sys
from pathlib import Path
import joblib
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import onnx
    import onnxruntime as ort
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    print("Loading trained model...")
    model_data = joblib.load("models/trained/isolation_forest_detector.pkl")
    
    if isinstance(model_data, dict):
        sklearn_model = model_data['model']
        feature_names = model_data.get('feature_names', None)
        print(f"Model type: {type(sklearn_model).__name__}")
        print(f"Features: {len(feature_names) if feature_names else 'unknown'}")
    else:
        sklearn_model = model_data
        feature_names = None
    
    # Determine number of features
    if hasattr(sklearn_model, 'n_features_in_'):
        n_features = sklearn_model.n_features_in_
    elif feature_names:
        n_features = len(feature_names)
    else:
        n_features = 25  # Default based on our data
    
    print(f"Converting model with {n_features} features...")
    
    # Define initial input type
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    
    # Convert to ONNX
    onnx_model = convert_sklearn(
        sklearn_model,
        initial_types=initial_type,
        target_opset={'': 12, 'ai.onnx.ml': 3},
        verbose=0
    )
    
    # Set model metadata
    onnx_model.doc_string = f"GPS Jamming Detection Model - IsolationForest"
    onnx_model.model_version = 1
    
    # Save ONNX model
    output_path = Path("models/onnx/gps_jamming_detector.onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(onnx_model, str(output_path))
    
    print(f"Model converted successfully to {output_path}")
    
    # Test the model
    session = ort.InferenceSession(str(output_path))
    input_name = session.get_inputs()[0].name
    test_input = np.random.random((1, n_features)).astype(np.float32)
    result = session.run(None, {input_name: test_input})
    
    print(f"ONNX model test successful. Output shape: {[r.shape for r in result]}")
    print("ONNX conversion completed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 