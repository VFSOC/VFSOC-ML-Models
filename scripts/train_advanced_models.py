#!/usr/bin/env python3
"""
Advanced Model Training for GPS Jamming Detection.

This script implements comprehensive model training with hyperparameter optimization,
cross-validation, and automated model selection for optimal performance.
"""

import sys
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import time
import json
from typing import Dict, List, Tuple, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vfsoc_ml.models.isolation_forest import IsolationForestDetector
from vfsoc_ml.models.advanced_models import (
    XGBoostDetector, 
    LightGBMDetector, 
    EnhancedRandomForestDetector,
    NeuralNetworkDetector,
    EnsembleDetector,
    AutoMLDetector
)
from vfsoc_ml.data.data_loader import VFSOCDataLoader
from vfsoc_ml.preprocessing.enhanced_feature_extractor import EnhancedFeatureExtractor

# Additional imports for optimization
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score, 
    StratifiedKFold,
    RandomizedSearchCV,
    GridSearchCV
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


class AdvancedModelTrainer:
    """
    Advanced model trainer with hyperparameter optimization and evaluation.
    """
    
    def __init__(self, 
                 cv_folds: int = 5,
                 test_size: float = 0.2,
                 val_size: float = 0.1,
                 random_state: int = 42,
                 n_jobs: int = -1):
        """
        Initialize advanced model trainer.
        
        Args:
            cv_folds: Number of cross-validation folds
            test_size: Test set size ratio
            val_size: Validation set size ratio
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs
        """
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.logger = logging.getLogger(__name__)
        
        # Training results storage
        self.model_results = {}
        self.best_model = None
        self.best_score = -np.inf
        
    def load_data(self, data_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load training data from disk."""
        self.logger.info(f"Loading data from {data_path}...")
        
        data_path = Path(data_path)
        
        if (data_path / "features.parquet").exists():
            # Load from parquet format (large dataset)
            features = pd.read_parquet(data_path / "features.parquet")
            labels = np.load(data_path / "labels.npy")
            
            self.logger.info(f"Loaded {len(features)} samples with {features.shape[1]} features")
            
        else:
            # Fallback to existing data loader
            loader = VFSOCDataLoader()
            features, labels = loader.load_synthetic_data(str(data_path))
        
        return features, labels
    
    def prepare_data(self, features: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
        """Prepare data for training with train/validation/test splits."""
        self.logger.info("Preparing data splits...")
        
        # Convert labels to binary format for sklearn compatibility
        y_binary = (labels == 1).astype(int)
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, y_binary, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y_binary
        )
        
        # Second split: train vs val
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=y_temp
        )
        
        data_splits = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': features.columns.tolist()
        }
        
        self.logger.info(f"Data splits created:")
        self.logger.info(f"  - Training: {len(X_train)} samples")
        self.logger.info(f"  - Validation: {len(X_val)} samples")
        self.logger.info(f"  - Test: {len(X_test)} samples")
        
        # Check class distribution
        for split_name, y_split in [('train', y_train), ('val', y_val), ('test', y_test)]:
            jamming_ratio = np.mean(y_split == 0)  # 0 = jamming in binary format
            self.logger.info(f"  - {split_name} jamming ratio: {jamming_ratio:.3f}")
        
        return data_splits
    
    def get_hyperparameter_grids(self) -> Dict[str, Dict]:
        """Get hyperparameter grids for different models."""
        return {
            'isolation_forest': {
                'contamination': [0.05, 0.08, 0.1, 0.12],
                'n_estimators': [100, 200, 300],
                'max_features': [0.5, 0.8, 1.0]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'scale_pos_weight': [5, 10, 15]
            },
            'lightgbm': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 63, 127]
            },
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'max_features': ['sqrt', 'log2', 0.8]
            },
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 75, 25)],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }
    
    def train_isolation_forest(self, data_splits: Dict[str, Any], optimize: bool = True) -> Dict[str, Any]:
        """Train Isolation Forest with optional hyperparameter optimization."""
        self.logger.info("Training Isolation Forest...")
        
        if optimize:
            param_grid = self.get_hyperparameter_grids()['isolation_forest']
            
            best_score = -np.inf
            best_params = None
            
            # Manual grid search for Isolation Forest (unsupervised)
            for contamination in param_grid['contamination']:
                for n_estimators in param_grid['n_estimators']:
                    for max_features in param_grid['max_features']:
                        
                        model = IsolationForestDetector(
                            contamination=contamination,
                            n_estimators=n_estimators,
                            max_features=max_features,
                            random_state=self.random_state
                        )
                        
                        # Train on training set
                        model.train(data_splits['X_train'])
                        
                        # Evaluate on validation set
                        val_predictions = model.predict(data_splits['X_val'])
                        val_predictions_binary = (val_predictions == 1).astype(int)
                        
                        # Calculate F1 score
                        from sklearn.metrics import f1_score
                        score = f1_score(data_splits['y_val'], val_predictions_binary)
                        
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'contamination': contamination,
                                'n_estimators': n_estimators,
                                'max_features': max_features
                            }
            
            self.logger.info(f"Best Isolation Forest params: {best_params}")
            self.logger.info(f"Best validation F1 score: {best_score:.4f}")
            
            # Train final model with best parameters
            final_model = IsolationForestDetector(**best_params, random_state=self.random_state)
        else:
            final_model = IsolationForestDetector(random_state=self.random_state)
        
        final_model.train(data_splits['X_train'])
        
        # Evaluate on test set
        test_predictions = final_model.predict(data_splits['X_test'])
        test_metrics = self.evaluate_model_predictions(
            data_splits['y_test'], 
            (test_predictions == 1).astype(int),
            "Isolation Forest"
        )
        
        return {
            'model': final_model,
            'best_params': best_params if optimize else None,
            'test_metrics': test_metrics,
            'model_type': 'isolation_forest'
        }
    
    def train_supervised_model(self, model_class, model_name: str, data_splits: Dict[str, Any], 
                             optimize: bool = True) -> Dict[str, Any]:
        """Train a supervised model with hyperparameter optimization."""
        self.logger.info(f"Training {model_name}...")
        
        if optimize and model_name.lower() in self.get_hyperparameter_grids():
            param_grid = self.get_hyperparameter_grids()[model_name.lower()]
            
            # Create base model
            base_model = model_class(random_state=self.random_state)
            
            # Create sklearn-compatible model for hyperparameter search
            sklearn_model = base_model._create_model()
            
            # Randomized search for efficiency
            search = RandomizedSearchCV(
                sklearn_model,
                param_grid,
                n_iter=20,  # Limit iterations for speed
                cv=self.cv_folds,
                scoring='f1',
                n_jobs=self.n_jobs,
                random_state=self.random_state
            )
            
            # Convert labels back to original format for training
            y_train_original = np.where(data_splits['y_train'] == 0, -1, 1)
            
            search.fit(data_splits['X_train'], data_splits['y_train'])
            
            self.logger.info(f"Best {model_name} params: {search.best_params_}")
            self.logger.info(f"Best CV F1 score: {search.best_score_:.4f}")
            
            # Create final model with best parameters
            final_model = model_class(**search.best_params_, random_state=self.random_state)
            best_params = search.best_params_
        else:
            final_model = model_class(random_state=self.random_state)
            best_params = None
        
        # Train final model
        y_train_original = np.where(data_splits['y_train'] == 0, -1, 1)
        final_model.train(data_splits['X_train'], y_train_original)
        
        # Evaluate on test set
        test_predictions = final_model.predict(data_splits['X_test'])
        test_predictions_binary = (test_predictions == 1).astype(int)
        
        test_metrics = self.evaluate_model_predictions(
            data_splits['y_test'], 
            test_predictions_binary,
            model_name
        )
        
        return {
            'model': final_model,
            'best_params': best_params,
            'test_metrics': test_metrics,
            'model_type': model_name.lower()
        }
    
    def evaluate_model_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 model_name: str) -> Dict[str, Any]:
        """Evaluate model predictions and return comprehensive metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        self.logger.info(f"{model_name} Test Metrics:")
        self.logger.info(f"  - Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"  - Precision: {metrics['precision']:.4f}")
        self.logger.info(f"  - Recall: {metrics['recall']:.4f}")
        self.logger.info(f"  - F1 Score: {metrics['f1_score']:.4f}")
        self.logger.info(f"  - ROC AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def train_all_models(self, data_splits: Dict[str, Any], 
                        model_list: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train all available models and compare performance."""
        self.logger.info("Starting comprehensive model training...")
        
        if model_list is None:
            model_list = ['isolation_forest', 'xgboost', 'lightgbm', 'random_forest', 'neural_network']
        
        results = {}
        
        for model_name in model_list:
            try:
                if model_name == 'isolation_forest':
                    result = self.train_isolation_forest(data_splits, optimize=True)
                elif model_name == 'xgboost':
                    try:
                        result = self.train_supervised_model(XGBoostDetector, 'XGBoost', data_splits)
                    except ImportError:
                        self.logger.warning("XGBoost not available, skipping...")
                        continue
                elif model_name == 'lightgbm':
                    try:
                        result = self.train_supervised_model(LightGBMDetector, 'LightGBM', data_splits)
                    except ImportError:
                        self.logger.warning("LightGBM not available, skipping...")
                        continue
                elif model_name == 'random_forest':
                    result = self.train_supervised_model(EnhancedRandomForestDetector, 'Random_Forest', data_splits)
                elif model_name == 'neural_network':
                    result = self.train_supervised_model(NeuralNetworkDetector, 'Neural_Network', data_splits)
                else:
                    self.logger.warning(f"Unknown model type: {model_name}")
                    continue
                
                results[model_name] = result
                
                # Track best model
                f1_score = result['test_metrics']['f1_score']
                if f1_score > self.best_score:
                    self.best_score = f1_score
                    self.best_model = result
                
            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        self.model_results = results
        
        # Log summary
        self.logger.info("\n=== Model Comparison Summary ===")
        for model_name, result in results.items():
            metrics = result['test_metrics']
            self.logger.info(f"{model_name}: F1={metrics['f1_score']:.4f}, "
                           f"Precision={metrics['precision']:.4f}, "
                           f"Recall={metrics['recall']:.4f}")
        
        if self.best_model:
            best_name = next(name for name, result in results.items() if result == self.best_model)
            self.logger.info(f"\nBest Model: {best_name} (F1 Score: {self.best_score:.4f})")
        
        return results
    
    def save_results(self, output_dir: str, data_splits: Dict[str, Any]):
        """Save training results and models."""
        self.logger.info(f"Saving results to {output_dir}...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual models
        for model_name, result in self.model_results.items():
            model_path = output_path / f"{model_name}_model.pkl"
            result['model'].save_model(model_path)
            self.logger.info(f"Saved {model_name} model to {model_path}")
        
        # Save best model separately
        if self.best_model:
            best_name = next(name for name, result in self.model_results.items() 
                           if result == self.best_model)
            best_model_path = output_path / "best_model.pkl"
            self.best_model['model'].save_model(best_model_path)
            self.logger.info(f"Saved best model ({best_name}) to {best_model_path}")
        
        # Save training summary
        summary = {
            'training_timestamp': pd.Timestamp.now().isoformat(),
            'data_info': {
                'total_samples': len(data_splits['X_train']) + len(data_splits['X_val']) + len(data_splits['X_test']),
                'num_features': len(data_splits['feature_names']),
                'feature_names': data_splits['feature_names']
            },
            'training_config': {
                'cv_folds': self.cv_folds,
                'test_size': self.test_size,
                'val_size': self.val_size,
                'random_state': self.random_state
            },
            'model_results': {}
        }
        
        # Add model results (excluding the actual model objects)
        for model_name, result in self.model_results.items():
            summary['model_results'][model_name] = {
                'test_metrics': result['test_metrics'],
                'best_params': result['best_params'],
                'model_type': result['model_type']
            }
        
        # Add best model info
        if self.best_model:
            best_name = next(name for name, result in self.model_results.items() 
                           if result == self.best_model)
            summary['best_model'] = {
                'name': best_name,
                'f1_score': self.best_score,
                'metrics': self.best_model['test_metrics']
            }
        
        summary_path = output_path / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Training summary saved to {summary_path}")


def main():
    """Main function for advanced model training."""
    parser = argparse.ArgumentParser(description="Advanced GPS jamming detection model training")
    parser.add_argument("--data", "-d", type=str, required=True,
                       help="Path to training data directory")
    parser.add_argument("--output", "-o", type=str, default="models/advanced_trained",
                       help="Output directory for trained models")
    parser.add_argument("--models", "-m", nargs='+', 
                       default=['isolation_forest', 'xgboost', 'lightgbm', 'random_forest'],
                       help="Models to train")
    parser.add_argument("--cv-folds", type=int, default=5,
                       help="Number of cross-validation folds")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Test set size ratio")
    parser.add_argument("--val-size", type=float, default=0.1,
                       help="Validation set size ratio")
    parser.add_argument("--n-jobs", type=int, default=-1,
                       help="Number of parallel jobs")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("=== Advanced GPS Jamming Detection Model Training ===")
    logger.info(f"Data path: {args.data}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Models to train: {args.models}")
    logger.info(f"CV folds: {args.cv_folds}")
    logger.info(f"Test size: {args.test_size}")
    logger.info(f"Validation size: {args.val_size}")
    
    try:
        # Initialize trainer
        trainer = AdvancedModelTrainer(
            cv_folds=args.cv_folds,
            test_size=args.test_size,
            val_size=args.val_size,
            n_jobs=args.n_jobs
        )
        
        # Load and prepare data
        features, labels = trainer.load_data(args.data)
        data_splits = trainer.prepare_data(features, labels)
        
        # Train all models
        results = trainer.train_all_models(data_splits, args.models)
        
        # Save results
        trainer.save_results(args.output, data_splits)
        
        logger.info("=== Training Complete ===")
        logger.info(f"Trained {len(results)} models successfully")
        logger.info(f"Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main()) 