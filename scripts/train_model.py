#!/usr/bin/env python3
"""
Train GPS jamming detection models.

This script trains various machine learning models for GPS jamming detection
using synthetic or real data.
"""

import sys
import argparse
import logging
import yaml
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vfsoc_ml.models.isolation_forest import IsolationForestDetector
from vfsoc_ml.data.data_loader import VFSOCDataLoader


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_isolation_forest(config: dict, X_train, y_train, X_test, y_test) -> IsolationForestDetector:
    """Train Isolation Forest model."""
    logger = logging.getLogger(__name__)
    
    # Get model configuration
    model_config = config.get('isolation_forest', {})
    
    logger.info("Training Isolation Forest model...")
    logger.info(f"Model configuration: {model_config}")
    
    # Create model
    model = IsolationForestDetector(**model_config)
    
    # Train model
    training_metrics = model.train(X_train, feature_names=X_train.columns.tolist())
    
    # Evaluate on test set
    test_metrics = model.evaluate(X_test, y_test)
    
    logger.info(f"Training metrics: {training_metrics}")
    logger.info(f"Test metrics: {test_metrics}")
    
    return model


def main():
    """Main function to train models."""
    parser = argparse.ArgumentParser(description="Train GPS jamming detection models")
    parser.add_argument("--data", "-d", type=str, default="data/synthetic",
                       help="Path to training data directory")
    parser.add_argument("--config", "-c", type=str, default="config/model_config.yaml",
                       help="Path to model configuration file")
    parser.add_argument("--model", "-m", type=str, default="isolation_forest",
                       choices=["isolation_forest", "all"],
                       help="Model to train")
    parser.add_argument("--output", "-o", type=str, default="models/trained",
                       help="Output directory for trained models")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting model training...")
    logger.info(f"Data directory: {args.data}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output directory: {args.output}")
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Load data
        logger.info("Loading training data...")
        data_loader = VFSOCDataLoader()
        features_df, labels = data_loader.load_synthetic_data(args.data)
        
        # Split data
        X_train, X_test, y_train, y_test = data_loader.get_train_test_split(
            test_size=config.get('training', {}).get('test_size', 0.2),
            random_state=config.get('training', {}).get('random_state', 42)
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Create output directory
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        trained_models = {}
        
        # Train models
        if args.model == "isolation_forest" or args.model == "all":
            model = train_isolation_forest(config, X_train, y_train, X_test, y_test)
            
            # Save model
            model_path = output_path / "isolation_forest_detector.pkl"
            model.save_model(model_path)
            logger.info(f"Isolation Forest model saved to {model_path}")
            
            trained_models["isolation_forest"] = {
                "model_path": str(model_path),
                "metrics": model.get_model_info()
            }
        
        # Save training summary
        summary = {
            "training_config": config,
            "data_info": data_loader.get_feature_info(),
            "trained_models": trained_models,
            "training_timestamp": str(Path().cwd())
        }
        
        summary_path = output_path / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Training summary saved to {summary_path}")
        logger.info("Model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 