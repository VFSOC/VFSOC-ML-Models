#!/usr/bin/env python3
"""
ONNX Conversion Script for Energy Consumption Anomaly Detection

This script converts the trained Isolation Forest model to ONNX format
for production deployment.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import joblib
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from vfsoc_ml.utils.logger import setup_logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_trained_model(model_path: str) -> IsolationForest:
    """Load the trained Isolation Forest model."""
    logger = logging.getLogger(__name__)
    
    try:
        model = joblib.load(model_path)
        logger.info(f"Successfully loaded model from: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def load_scaler(scaler_path: str) -> StandardScaler:
    """Load the feature scaler."""
    logger = logging.getLogger(__name__)
    
    try:
        scaler = joblib.load(scaler_path)
        logger.info(f"Successfully loaded scaler from: {scaler_path}")
        return scaler
    except Exception as e:
        logger.error(f"Error loading scaler: {str(e)}")
        raise


def convert_model_to_onnx(
    model: IsolationForest,
    scaler: StandardScaler,
    n_features: int,
    output_path: str,
    model_name: str = "energy_anomaly_detector"
) -> str:
    """
    Convert the trained model and scaler to ONNX format.
    
    Args:
        model: Trained Isolation Forest model
        scaler: Fitted StandardScaler
        n_features: Number of input features
        output_path: Path to save ONNX model
        model_name: Name for the ONNX model
        
    Returns:
        Path to saved ONNX model
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define input type (batch_size, n_features)
        initial_type = [('float_input', FloatTensorType([None, n_features]))]
        
        # Convert scaler to ONNX
        logger.info("Converting scaler to ONNX...")
        scaler_onnx = convert_sklearn(
            scaler,
            initial_types=initial_type,
            target_opset=11
        )
        
        # Save scaler ONNX
        scaler_onnx_path = output_dir / f"{model_name}_scaler.onnx"
        with open(scaler_onnx_path, "wb") as f:
            f.write(scaler_onnx.SerializeToString())
        logger.info(f"Scaler ONNX saved to: {scaler_onnx_path}")
        
        # Convert model to ONNX
        logger.info("Converting Isolation Forest model to ONNX...")
        model_onnx = convert_sklearn(
            model,
            initial_types=initial_type,
            target_opset=11,
            options={IsolationForest: {'zipmap': False}}  # Disable zipmap for cleaner output
        )
        
        # Save model ONNX
        model_onnx_path = output_dir / f"{model_name}.onnx"
        with open(model_onnx_path, "wb") as f:
            f.write(model_onnx.SerializeToString())
        logger.info(f"Model ONNX saved to: {model_onnx_path}")
        
        # Verify ONNX models
        logger.info("Verifying ONNX models...")
        scaler_model = onnx.load(str(scaler_onnx_path))
        onnx.checker.check_model(scaler_model)
        
        model_model = onnx.load(str(model_onnx_path))
        onnx.checker.check_model(model_model)
        
        logger.info("ONNX models verified successfully!")
        
        return str(model_onnx_path)
        
    except Exception as e:
        logger.error(f"Error converting model to ONNX: {str(e)}")
        raise


def test_onnx_model(
    onnx_model_path: str,
    onnx_scaler_path: str,
    original_model: IsolationForest,
    original_scaler: StandardScaler,
    test_data: np.ndarray
) -> bool:
    """
    Test the ONNX model against the original model to ensure consistency.
    
    Args:
        onnx_model_path: Path to ONNX model
        onnx_scaler_path: Path to ONNX scaler
        original_model: Original sklearn model
        original_scaler: Original sklearn scaler
        test_data: Test data for comparison
        
    Returns:
        True if models produce consistent results
    """
    logger = logging.getLogger(__name__)
    
    try:
        import onnxruntime as ort
        
        # Load ONNX models
        scaler_session = ort.InferenceSession(onnx_scaler_path)
        model_session = ort.InferenceSession(onnx_model_path)
        
        # Get input names
        scaler_input_name = scaler_session.get_inputs()[0].name
        model_input_name = model_session.get_inputs()[0].name
        
        # Test with original sklearn pipeline
        scaled_data_sklearn = original_scaler.transform(test_data)
        predictions_sklearn = original_model.predict(scaled_data_sklearn)
        scores_sklearn = original_model.decision_function(scaled_data_sklearn)
        
        # Test with ONNX pipeline
        scaled_data_onnx = scaler_session.run(None, {scaler_input_name: test_data.astype(np.float32)})[0]
        onnx_outputs = model_session.run(None, {model_input_name: scaled_data_onnx})
        
        # ONNX outputs: [labels, scores]
        predictions_onnx = onnx_outputs[0].flatten()
        scores_onnx = onnx_outputs[1].flatten()
        
        # Compare results
        predictions_match = np.allclose(predictions_sklearn, predictions_onnx, rtol=1e-5)
        scores_match = np.allclose(scores_sklearn, scores_onnx, rtol=1e-5)
        
        if predictions_match and scores_match:
            logger.info("ONNX model test passed! Results match original model.")
            return True
        else:
            logger.warning("ONNX model test failed! Results don't match original model.")
            logger.warning(f"Predictions match: {predictions_match}")
            logger.warning(f"Scores match: {scores_match}")
            return False
            
    except Exception as e:
        logger.error(f"Error testing ONNX model: {str(e)}")
        return False


def create_model_metadata(
    config: Dict[str, Any],
    model_path: str,
    n_features: int,
    feature_names: list
) -> Dict[str, Any]:
    """Create metadata for the ONNX model."""
    metadata = {
        "model_info": {
            "name": "Energy Consumption Anomaly Detector",
            "version": "1.0.0",
            "description": "Isolation Forest model for detecting irregular energy consumption patterns in EV charging stations",
            "model_type": "IsolationForest",
            "framework": "scikit-learn",
            "format": "ONNX"
        },
        "input_info": {
            "n_features": n_features,
            "feature_names": feature_names,
            "input_type": "float32",
            "input_shape": [-1, n_features],
            "preprocessing": "StandardScaler normalization required"
        },
        "output_info": {
            "outputs": [
                {
                    "name": "labels",
                    "description": "Anomaly labels (1=normal, -1=anomaly)",
                    "type": "int64",
                    "shape": [-1]
                },
                {
                    "name": "scores", 
                    "description": "Anomaly scores (higher = more normal)",
                    "type": "float32",
                    "shape": [-1]
                }
            ]
        },
        "model_parameters": config.get('model', {}).get('isolation_forest', {}),
        "deployment_info": {
            "recommended_batch_size": 100,
            "max_batch_size": 1000,
            "memory_requirements": "Low",
            "inference_time": "< 10ms per sample"
        }
    }
    
    return metadata


def save_metadata(metadata: Dict[str, Any], output_path: str, model_name: str) -> str:
    """Save model metadata to JSON file."""
    import json
    
    metadata_path = Path(output_path) / f"{model_name}_metadata.json"
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return str(metadata_path)


def main():
    """Main conversion pipeline."""
    parser = argparse.ArgumentParser(description="Convert trained model to ONNX format")
    parser.add_argument(
        "--config",
        type=str,
        default="config/energy_consumption_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/trained/energy_anomaly_detector.pkl",
        help="Path to trained model file"
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default="models/trained/feature_scaler.pkl",
        help="Path to feature scaler file"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output directory for ONNX models (default: from config)"
    )
    parser.add_argument(
        "--test-conversion",
        action="store_true",
        help="Test ONNX model against original model"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("onnx_conversion", level=args.log_level)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Set output path
        if args.output_path:
            output_path = args.output_path
        else:
            output_path = config['deployment']['onnx']['export_path']
        
        model_name = config['deployment']['onnx']['model_name']
        
        # Load trained model and scaler
        logger.info("Loading trained model and scaler...")
        model = load_trained_model(args.model_path)
        scaler = load_scaler(args.scaler_path)
        
        # Get number of features from scaler
        n_features = scaler.n_features_in_
        logger.info(f"Model expects {n_features} input features")
        
        # Convert to ONNX
        logger.info("Converting model to ONNX format...")
        onnx_model_path = convert_model_to_onnx(
            model, scaler, n_features, output_path, model_name
        )
        
        # Test conversion if requested
        if args.test_conversion:
            logger.info("Testing ONNX conversion...")
            
            # Create test data
            test_data = np.random.randn(10, n_features).astype(np.float32)
            
            # Test ONNX model
            onnx_scaler_path = Path(output_path) / f"{model_name}_scaler.onnx"
            test_passed = test_onnx_model(
                onnx_model_path, str(onnx_scaler_path), model, scaler, test_data
            )
            
            if not test_passed:
                logger.error("ONNX conversion test failed!")
                sys.exit(1)
        
        # Create and save metadata
        logger.info("Creating model metadata...")
        
        # Get feature names (you might want to load these from somewhere)
        feature_names = [f"feature_{i}" for i in range(n_features)]
        
        metadata = create_model_metadata(config, onnx_model_path, n_features, feature_names)
        metadata_path = save_metadata(metadata, output_path, model_name)
        
        logger.info(f"Model metadata saved to: {metadata_path}")
        logger.info("ONNX conversion completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("ONNX CONVERSION SUMMARY")
        print("="*50)
        print(f"Original model: {args.model_path}")
        print(f"ONNX model: {onnx_model_path}")
        print(f"ONNX scaler: {Path(output_path) / f'{model_name}_scaler.onnx'}")
        print(f"Metadata: {metadata_path}")
        print(f"Input features: {n_features}")
        print(f"Model ready for deployment!")
        print("="*50)
        
    except Exception as e:
        logger.error(f"ONNX conversion failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 