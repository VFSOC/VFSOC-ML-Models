#!/usr/bin/env python3
"""
Irregular Energy Consumption Detection Training Script

This script trains machine learning models to detect irregular energy consumption
patterns in EV charging stations, focusing on:
- Meter tampering
- Unauthorized power drain  
- Broken billing logic
- Station configuration errors

Uses Isolation Forest as primary algorithm with Z-score analysis as secondary method.
"""

import os
import sys
import yaml
import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import joblib
import mlflow
import mlflow.sklearn

from vfsoc_ml.data.data_loader import EnergyConsumptionDataLoader
from vfsoc_ml.data.feature_engineering import EnergyConsumptionFeatureEngineer
from vfsoc_ml.utils.logger import setup_logger
from vfsoc_ml.utils.metrics import calculate_anomaly_metrics
from vfsoc_ml.utils.visualization import plot_anomaly_analysis


class IrregularEnergyConsumptionTrainer:
    """
    Trainer for irregular energy consumption detection models.
    
    Implements the ML approach:
    - Primary: Isolation Forest for sparse outlier detection
    - Secondary: Z-score analysis for statistical detection
    - Optional: One-Class SVM for tight boundary learning
    """
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = setup_logger("irregular_energy_trainer")
        
        # Initialize components
        self.data_loader = EnergyConsumptionDataLoader(self.config)
        self.feature_engineer = EnergyConsumptionFeatureEngineer(self.config)
        
        # Model components
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        
        # Results storage
        self.training_results = {}
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file."""
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def train(self) -> Dict[str, Any]:
        """
        Execute the complete training pipeline for irregular energy consumption detection.
        
        Returns:
            Dictionary containing training results and model performance
        """
        self.logger.info("Starting irregular energy consumption detection training...")
        
        # Setup MLflow experiment
        self._setup_mlflow()
        
        with mlflow.start_run(run_name=f"irregular_energy_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # 1. Load and preprocess data
            self.logger.info("Loading and preprocessing data...")
            data = self._load_and_preprocess_data()
            
            # 2. Engineer features (focus on core features)
            self.logger.info("Engineering features for energy anomaly detection...")
            features = self._engineer_features(data)
            
            # 3. Prepare training data
            X_train, X_test = self._prepare_training_data(features)
            
            # 4. Train models
            self.logger.info("Training anomaly detection models...")
            self._train_models(X_train)
            
            # 5. Evaluate models
            self.logger.info("Evaluating model performance...")
            evaluation_results = self._evaluate_models(X_test, data.iloc[len(X_train):])
            
            # 6. Generate sample alerts
            self.logger.info("Generating sample alert outputs...")
            sample_alerts = self._generate_sample_alerts(X_test, data.iloc[len(X_train):])
            
            # 7. Save models and results
            self._save_models_and_results(evaluation_results, sample_alerts)
            
            # Log to MLflow
            self._log_to_mlflow(evaluation_results)
            
            self.logger.info("Training completed successfully!")
            
        return {
            'evaluation_results': evaluation_results,
            'sample_alerts': sample_alerts,
            'model_paths': self._get_model_paths()
        }
    
    def _load_and_preprocess_data(self) -> pd.DataFrame:
        """Load and preprocess the energy consumption data."""
        # Load real data
        data = self.data_loader.load_station_data()
        
        # Apply preprocessing
        cleaned_data = self.data_loader.preprocess_data(data)
        
        self.logger.info(f"Loaded {len(cleaned_data)} charging sessions")
        self.logger.info(f"Energy range: {cleaned_data['kwhTotal'].min():.2f} - {cleaned_data['kwhTotal'].max():.2f} kWh")
        
        return cleaned_data
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features focusing on core energy anomaly detection features."""
        features = self.feature_engineer.create_features(data)
        
        # Get core features as specified in the approach
        core_features = self.feature_engineer.get_core_features()
        self.logger.info(f"Core features for anomaly detection: {core_features}")
        
        # Validate feature quality
        validation_results = self.feature_engineer.validate_features(features)
        self.logger.info(f"Feature validation: {len(validation_results['core_features_present'])}/4 core features present")
        
        self.feature_names = features.columns.tolist()
        return features
    
    def _prepare_training_data(self, features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare training and test data."""
        # Split data (unsupervised, so no labels needed)
        train_ratio = self.config['data']['split']['train_ratio']
        test_ratio = self.config['data']['split']['test_ratio']
        
        # For unsupervised learning, we still split to evaluate performance
        X_train, X_test = train_test_split(
            features,
            test_size=(test_ratio + self.config['data']['split']['validation_ratio']),
            random_state=self.config['data']['split']['random_state']
        )
        
        self.logger.info(f"Training set: {len(X_train)} sessions")
        self.logger.info(f"Test set: {len(X_test)} sessions")
        
        return X_train, X_test
    
    def _train_models(self, X_train: pd.DataFrame) -> None:
        """Train the anomaly detection models."""
        
        # 1. Primary Model: Isolation Forest
        self.logger.info("Training Isolation Forest (primary model)...")
        self._train_isolation_forest(X_train)
        
        # 2. Secondary Model: Z-score Analysis
        self.logger.info("Training Z-score detector (secondary model)...")
        self._train_zscore_detector(X_train)
        
        # 3. Optional: One-Class SVM
        if 'one_class_svm' in self.config['model']:
            self.logger.info("Training One-Class SVM (optional model)...")
            self._train_one_class_svm(X_train)
    
    def _train_isolation_forest(self, X_train: pd.DataFrame) -> None:
        """Train Isolation Forest model."""
        config = self.config['model']['isolation_forest']
        
        if self.config['training']['hyperparameter_tuning']:
            # Hyperparameter tuning
            param_grid = self.config['training']['param_grid']['isolation_forest']
            
            # Create base model
            base_model = IsolationForest(
                random_state=config['random_state'],
                n_jobs=config['n_jobs']
            )
            
            # Custom scoring for anomaly detection
            def anomaly_scorer(estimator, X):
                scores = estimator.decision_function(X)
                # Use negative anomaly scores (lower is more anomalous)
                return np.mean(scores)
            
            # Grid search
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                scoring=anomaly_scorer,
                cv=3,  # Reduced for unsupervised learning
                n_jobs=-1
            )
            
            grid_search.fit(X_train)
            self.models['isolation_forest'] = grid_search.best_estimator_
            
            self.logger.info(f"Best Isolation Forest parameters: {grid_search.best_params_}")
        else:
            # Use default configuration
            self.models['isolation_forest'] = IsolationForest(**config)
            self.models['isolation_forest'].fit(X_train)
        
        self.logger.info("Isolation Forest training completed")
    
    def _train_zscore_detector(self, X_train: pd.DataFrame) -> None:
        """Train Z-score based detector."""
        config = self.config['model']['z_score']
        
        # Calculate statistics for Z-score detection
        stats = {
            'mean': X_train.mean(),
            'std': X_train.std(),
            'threshold': config['threshold']
        }
        
        # Special handling for core energy features
        if 'energy' in X_train.columns and config.get('per_vehicle', False):
            # This would require vehicle info, for now use global stats
            stats['energy_mean'] = X_train['energy'].mean()
            stats['energy_std'] = X_train['energy'].std()
        
        self.models['z_score'] = stats
        self.logger.info(f"Z-score detector trained with threshold: {config['threshold']}")
    
    def _train_one_class_svm(self, X_train: pd.DataFrame) -> None:
        """Train One-Class SVM model."""
        config = self.config['model']['one_class_svm']
        
        # Scale features for SVM
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers['one_class_svm'] = scaler
        
        # Train model
        model = OneClassSVM(**config)
        model.fit(X_train_scaled)
        self.models['one_class_svm'] = model
        
        self.logger.info("One-Class SVM training completed")
    
    def _evaluate_models(self, X_test: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate trained models on test data."""
        results = {}
        
        # Generate predictions from all models
        predictions = {}
        scores = {}
        
        # Isolation Forest predictions
        if 'isolation_forest' in self.models:
            iso_predictions = self.models['isolation_forest'].predict(X_test)
            iso_scores = self.models['isolation_forest'].decision_function(X_test)
            predictions['isolation_forest'] = (iso_predictions == -1).astype(int)
            scores['isolation_forest'] = -iso_scores  # Convert to positive anomaly scores
        
        # Z-score predictions
        if 'z_score' in self.models:
            z_predictions, z_scores = self._predict_zscore(X_test)
            predictions['z_score'] = z_predictions
            scores['z_score'] = z_scores
        
        # One-Class SVM predictions  
        if 'one_class_svm' in self.models:
            X_test_scaled = self.scalers['one_class_svm'].transform(X_test)
            svm_predictions = self.models['one_class_svm'].predict(X_test_scaled)
            svm_scores = self.models['one_class_svm'].score_samples(X_test_scaled)
            predictions['one_class_svm'] = (svm_predictions == -1).astype(int)
            scores['one_class_svm'] = -svm_scores
        
        # Ensemble prediction (if multiple models)
        if len(predictions) > 1:
            ensemble_weights = self.config['model']['ensemble'].get('weights', [1.0] * len(predictions))
            ensemble_scores = np.zeros(len(X_test))
            
            for i, (model_name, weight) in enumerate(zip(predictions.keys(), ensemble_weights)):
                ensemble_scores += weight * scores[model_name]
            
            # Normalize ensemble scores
            ensemble_scores = ensemble_scores / sum(ensemble_weights)
            
            # Use contamination rate to determine threshold
            contamination = self.config['model']['isolation_forest']['contamination']
            threshold = np.percentile(ensemble_scores, (1 - contamination) * 100)
            
            predictions['ensemble'] = (ensemble_scores > threshold).astype(int)
            scores['ensemble'] = ensemble_scores
        
        # Calculate metrics for each model
        for model_name in predictions.keys():
            model_results = self._calculate_metrics(
                predictions[model_name], 
                scores[model_name],
                X_test,
                test_data
            )
            results[model_name] = model_results
        
        return results
    
    def _predict_zscore(self, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Z-score based predictions."""
        stats = self.models['z_score']
        threshold = stats['threshold']
        
        # Calculate Z-scores for all features
        z_scores = np.abs((X_test - stats['mean']) / stats['std']).fillna(0)
        
        # Use maximum Z-score across features as anomaly score
        max_z_scores = z_scores.max(axis=1)
        
        # Predictions based on threshold
        predictions = (max_z_scores > threshold).astype(int)
        
        return predictions, max_z_scores.values
    
    def _calculate_metrics(self, predictions: np.ndarray, scores: np.ndarray, 
                          X_test: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate evaluation metrics for anomaly detection."""
        
        # Since we don't have true labels, we'll use synthetic evaluation
        # Create synthetic labels based on energy thresholds
        energy_col = 'energy' if 'energy' in X_test.columns else 'kwhTotal'
        true_anomalies = None
        
        if energy_col in X_test.columns:
            # Define anomalies as <5 kWh or >80 kWh based on our approach
            true_anomalies = ((X_test[energy_col] < 5.0) | (X_test[energy_col] > 80.0)).astype(int)
        
        # Use the utility function to calculate metrics
        metrics = calculate_anomaly_metrics(
            anomaly_scores=scores,
            true_labels=true_anomalies,
            predicted_labels=predictions
        )
        
        return metrics
    
    def _generate_sample_alerts(self, X_test: pd.DataFrame, test_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate sample alerts in the specified JSON format."""
        alerts = []
        
        # Use best performing model (assuming ensemble or isolation forest)
        model_name = 'ensemble' if 'ensemble' in self.models else 'isolation_forest'
        
        if model_name == 'ensemble':
            # Generate ensemble predictions
            predictions, scores = self._get_ensemble_predictions(X_test)
        else:
            predictions = self.models['isolation_forest'].predict(X_test)
            scores = -self.models['isolation_forest'].decision_function(X_test)
            predictions = (predictions == -1).astype(int)
        
        # Find anomalous sessions
        anomaly_indices = np.where(predictions == 1)[0]
        
        # Generate alerts for anomalies (limit to first 10 for demo)
        for idx in anomaly_indices[:10]:
            test_idx = test_data.index[idx]
            session_data = test_data.loc[test_idx]
            
            # Extract energy value
            energy = session_data.get('kwhTotal', session_data.get('energy', 0))
            
            # Determine expected range based on energy level
            if energy < 12:
                expected_range = "12-30 kWh"  # City EV
            elif energy <= 45:
                expected_range = "12-45 kWh"  # Normal range
            else:
                expected_range = "25-45 kWh"  # Long-range EV
            
            # Determine severity based on anomaly score
            score = scores[idx] if len(scores) > idx else 0.5
            if score >= self.config['alerts']['severity_thresholds']['high']:
                severity = "high"
            elif score >= self.config['alerts']['severity_thresholds']['medium']:
                severity = "medium"
            else:
                severity = "low"
            
            # Create alert in specified format
            alert = {
                "alert_type": "IrregularEnergyConsumption",
                "vehicle_id": str(session_data.get('userId', 'UNKNOWN')),
                "station_id": str(session_data.get('stationId', 'UNKNOWN')),
                "timestamp": session_data.get('startTime', datetime.now().isoformat()),
                "energy": float(energy),
                "expected_range": expected_range,
                "anomaly_score": float(score),
                "severity": severity
            }
            
            alerts.append(alert)
        
        self.logger.info(f"Generated {len(alerts)} sample alerts")
        return alerts
    
    def _get_ensemble_predictions(self, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Get ensemble predictions and scores."""
        predictions = {}
        scores = {}
        
        # Get predictions from all models
        if 'isolation_forest' in self.models:
            iso_pred = self.models['isolation_forest'].predict(X_test)
            iso_scores = -self.models['isolation_forest'].decision_function(X_test)
            predictions['isolation_forest'] = (iso_pred == -1).astype(int)
            scores['isolation_forest'] = iso_scores
        
        if 'z_score' in self.models:
            z_pred, z_scores = self._predict_zscore(X_test)
            predictions['z_score'] = z_pred
            scores['z_score'] = z_scores
        
        # Weighted ensemble
        weights = self.config['model']['ensemble'].get('weights', [1.0] * len(predictions))
        ensemble_scores = np.zeros(len(X_test))
        ensemble_predictions = np.zeros(len(X_test))
        
        for i, (model_name, weight) in enumerate(zip(predictions.keys(), weights)):
            ensemble_scores += weight * scores[model_name]
            ensemble_predictions += weight * predictions[model_name]
        
        # Normalize
        ensemble_scores = ensemble_scores / sum(weights)
        ensemble_predictions = (ensemble_predictions / sum(weights) > 0.5).astype(int)
        
        return ensemble_predictions, ensemble_scores
    
    def _save_models_and_results(self, evaluation_results: Dict[str, Any], 
                                sample_alerts: List[Dict[str, Any]]) -> None:
        """Save trained models and results."""
        # Create directories
        models_dir = Path(self.config['training']['model_save_path'])
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            if model_name == 'z_score':
                # Save statistics for Z-score model
                joblib.dump(model, models_dir / f"{model_name}_stats.pkl")
            else:
                joblib.dump(model, models_dir / f"{model_name}_model.pkl")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            joblib.dump(scaler, models_dir / f"{scaler_name}_scaler.pkl")
        
        # Save feature names
        joblib.dump(self.feature_names, models_dir / "feature_names.pkl")
        
        # Save evaluation results
        with open(models_dir / "evaluation_results.json", 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Save sample alerts
        with open(models_dir / "sample_alerts.json", 'w') as f:
            json.dump(sample_alerts, f, indent=2)
        
        self.logger.info(f"Models and results saved to {models_dir}")
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow experiment tracking."""
        mlflow_config = self.config['mlflow']
        mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
        mlflow.set_experiment(mlflow_config['experiment_name'])
    
    def _log_to_mlflow(self, evaluation_results: Dict[str, Any]) -> None:
        """Log training results to MLflow."""
        # Log parameters
        mlflow.log_params(self.config['model']['isolation_forest'])
        
        # Log metrics for each model
        for model_name, metrics in evaluation_results.items():
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{model_name}_{metric_name}", value)
        
        # Log model artifacts
        if 'isolation_forest' in self.models:
            mlflow.sklearn.log_model(
                self.models['isolation_forest'],
                "isolation_forest_model"
            )
    
    def _get_model_paths(self) -> Dict[str, str]:
        """Get paths to saved models."""
        models_dir = Path(self.config['training']['model_save_path'])
        return {
            'isolation_forest': str(models_dir / "isolation_forest_model.pkl"),
            'z_score': str(models_dir / "z_score_stats.pkl"),
            'feature_names': str(models_dir / "feature_names.pkl"),
            'evaluation_results': str(models_dir / "evaluation_results.json"),
            'sample_alerts': str(models_dir / "sample_alerts.json")
        }


def main():
    """Main training function."""
    # Configuration path
    config_path = Path(__file__).parent.parent / "config" / "energy_consumption_config.yaml"
    
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Initialize and run trainer
    trainer = IrregularEnergyConsumptionTrainer(str(config_path))
    
    try:
        results = trainer.train()
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*50)
        
        # Print evaluation summary
        print("\nModel Performance Summary:")
        for model_name, metrics in results['evaluation_results'].items():
            print(f"\n{model_name.upper()}:")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1-Score: {metrics['f1_score']:.3f}")
            print(f"  Anomaly Rate: {metrics['anomaly_rate']:.1%}")
        
        # Show sample alert
        if results['sample_alerts']:
            print(f"\nSample Alert (JSON format):")
            print(json.dumps(results['sample_alerts'][0], indent=2))
        
        print(f"\nModel files saved to: {Path(trainer.config['training']['model_save_path']).absolute()}")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 