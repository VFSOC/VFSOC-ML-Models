#!/usr/bin/env python3
"""
High-Accuracy GPS Jamming Detection Training Workflow.

This script implements a comprehensive ML pipeline designed to achieve maximum accuracy
while preventing overfitting through advanced techniques:

1. Generate 60,000+ balanced samples with enhanced feature engineering
2. Advanced cross-validation with stratified sampling
3. Multiple model architectures with ensemble methods
4. Hyperparameter optimization with Bayesian search
5. Early stopping and regularization to prevent overfitting
6. Comprehensive evaluation and model selection
"""

import sys
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import time
import json
import pickle
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ML Libraries
from sklearn.model_selection import (
    StratifiedKFold, train_test_split, cross_val_score,
    RandomizedSearchCV, cross_validate
)
from sklearn.ensemble import (
    IsolationForest, RandomForestClassifier, 
    GradientBoostingClassifier, VotingClassifier
)
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
import matplotlib.pyplot as plt
import seaborn as sns

# Advanced ML Libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Bayesian Optimization
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False


def setup_logging(verbose: bool = False):
    """Setup comprehensive logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training_workflow.log'),
            logging.StreamHandler()
        ]
    )


class HighAccuracyTrainingWorkflow:
    """
    High-accuracy training workflow with anti-overfitting measures.
    """
    
    def __init__(self, 
                 target_samples: int = 60000,
                 target_accuracy: float = 0.95,
                 max_iterations: int = 10,
                 random_state: int = 42):
        """
        Initialize high-accuracy training workflow.
        
        Args:
            target_samples: Target number of training samples
            target_accuracy: Target accuracy threshold
            max_iterations: Maximum training iterations
            random_state: Random state for reproducibility
        """
        self.target_samples = target_samples
        self.target_accuracy = target_accuracy
        self.max_iterations = max_iterations
        self.random_state = random_state
        
        self.logger = logging.getLogger(__name__)
        
        # Results tracking
        self.training_results = {}
        self.best_model = None
        self.best_score = 0.0
        self.iteration_results = []
        
        # Anti-overfitting configuration
        self.anti_overfitting_config = {
            'validation_strategy': 'stratified_kfold',
            'cv_folds': 10,
            'test_size': 0.15,
            'validation_size': 0.15,
            'early_stopping_rounds': 50,
            'regularization_strength': 'adaptive',
            'feature_selection_ratio': 0.8,
            'ensemble_diversity_threshold': 0.3
        }
        
    def generate_enhanced_dataset(self, output_dir: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate enhanced dataset with proper distribution."""
        self.logger.info(f"Generating enhanced dataset with {self.target_samples} samples...")
        
        # Advanced data generation configuration
        generation_config = {
            'base_samples': self.target_samples,
            'jamming_ratio': 0.08,  # 8% for realistic distribution
            'scenario_diversity': {
                'urban_scenarios': 0.35,
                'highway_scenarios': 0.25, 
                'rural_scenarios': 0.20,
                'industrial_scenarios': 0.20
            },
            'temporal_patterns': {
                'morning_rush': 0.25,
                'afternoon_rush': 0.25,
                'business_hours': 0.30,
                'off_hours': 0.20
            },
            'attack_patterns': {
                'simple_jamming': 0.40,
                'sweep_jamming': 0.25,
                'pulse_jamming': 0.20,
                'chirp_jamming': 0.15
            }
        }
        
        # Generate base dataset
        features, labels = self._generate_base_data(generation_config)
        
        # Apply enhanced feature engineering
        features = self._apply_advanced_feature_engineering(features)
        
        # Add synthetic noise and variations for robustness
        features, labels = self._add_robustness_variations(features, labels)
        
        # Save generated dataset
        self._save_dataset(features, labels, output_dir)
        
        self.logger.info(f"Enhanced dataset generated:")
        self.logger.info(f"  - Total samples: {len(features)}")
        self.logger.info(f"  - Features: {features.shape[1]}")
        self.logger.info(f"  - Jamming samples: {np.sum(labels == -1)}")
        self.logger.info(f"  - Normal samples: {np.sum(labels == 1)}")
        
        return features, labels
    
    def _generate_base_data(self, config: Dict) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate base synthetic data with scenario diversity."""
        all_features = []
        all_labels = []
        
        total_samples = config['base_samples']
        jamming_samples = int(total_samples * config['jamming_ratio'])
        normal_samples = total_samples - jamming_samples
        
        # Generate scenario-based data
        for scenario, ratio in config['scenario_diversity'].items():
            scenario_samples = int(total_samples * ratio)
            scenario_jamming = int(scenario_samples * config['jamming_ratio'])
            scenario_normal = scenario_samples - scenario_jamming
            
            self.logger.info(f"Generating {scenario}: {scenario_samples} samples")
            
            # Generate features for this scenario
            scenario_features = self._generate_scenario_features(
                scenario, scenario_normal, scenario_jamming, config
            )
            scenario_labels = np.concatenate([
                np.ones(scenario_normal),  # Normal = 1
                -np.ones(scenario_jamming)  # Jamming = -1
            ])
            
            all_features.append(scenario_features)
            all_labels.append(scenario_labels)
        
        # Combine all scenarios
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_labels = np.concatenate(all_labels)
        
        # Shuffle the dataset
        indices = np.random.permutation(len(combined_features))
        combined_features = combined_features.iloc[indices].reset_index(drop=True)
        combined_labels = combined_labels[indices]
        
        return combined_features, combined_labels
    
    def _generate_scenario_features(self, scenario: str, normal_count: int, 
                                   jamming_count: int, config: Dict) -> pd.DataFrame:
        """Generate features for a specific scenario."""
        total_count = normal_count + jamming_count
        
        # Base GPS features
        if scenario == 'urban_scenarios':
            # Urban: lower speeds, more GPS interruptions, more satellites
            speed = np.random.normal(25, 15, total_count)
            satellite_count = np.random.normal(8, 2, total_count)
            gps_fix_quality = np.random.normal(2.5, 0.8, total_count)
            signal_strength = np.random.normal(-30, 8, total_count)
            
        elif scenario == 'highway_scenarios':
            # Highway: higher speeds, fewer interruptions, stable GPS
            speed = np.random.normal(75, 20, total_count)
            satellite_count = np.random.normal(10, 1.5, total_count)
            gps_fix_quality = np.random.normal(3.2, 0.5, total_count)
            signal_strength = np.random.normal(-25, 5, total_count)
            
        elif scenario == 'rural_scenarios':
            # Rural: moderate speeds, fewer satellites, variable quality
            speed = np.random.normal(45, 25, total_count)
            satellite_count = np.random.normal(6, 2.5, total_count)
            gps_fix_quality = np.random.normal(2.0, 1.0, total_count)
            signal_strength = np.random.normal(-35, 10, total_count)
            
        else:  # industrial_scenarios
            # Industrial: low speeds, interference, variable GPS
            speed = np.random.normal(15, 10, total_count)
            satellite_count = np.random.normal(7, 3, total_count)
            gps_fix_quality = np.random.normal(1.8, 1.2, total_count)
            signal_strength = np.random.normal(-40, 12, total_count)
        
        # Clip values to realistic ranges  
        speed = np.clip(speed, 0, 120)
        satellite_count = np.clip(satellite_count, 0, 20).astype(int)
        gps_fix_quality = np.clip(gps_fix_quality, 0, 4)
        signal_strength = np.clip(signal_strength, -60, -15)
        
        # Apply jamming effects to jamming samples
        if jamming_count > 0:
            jamming_indices = np.arange(normal_count, total_count)
            
            # Reduce signal quality for jamming samples
            signal_strength[jamming_indices] += np.random.normal(-15, 5, jamming_count)
            satellite_count[jamming_indices] = np.maximum(
                satellite_count[jamming_indices] - np.random.poisson(3, jamming_count), 0
            )
            gps_fix_quality[jamming_indices] *= np.random.uniform(0.1, 0.5, jamming_count)
        
        # Create additional advanced features
        features_df = pd.DataFrame({
            'speed': speed,
            'satellite_count': satellite_count,
            'gps_fix_quality': gps_fix_quality,
            'signal_strength': signal_strength,
            'latitude': np.random.uniform(25, 49, total_count),
            'longitude': np.random.uniform(-125, -67, total_count),
            'altitude': np.random.normal(300, 500, total_count),
            'bearing': np.random.uniform(0, 360, total_count),
            'ignition_status': np.random.choice([0, 1], total_count, p=[0.1, 0.9]),
            'engine_hours': np.random.exponential(100, total_count),
            'odometer': np.random.exponential(50000, total_count),
            'fuel_level': np.random.uniform(0, 100, total_count),
            'device_temperature': np.random.normal(35, 15, total_count),
            'scenario_type': scenario
        })
        
        return features_df
    
    def _apply_advanced_feature_engineering(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply advanced feature engineering techniques."""
        self.logger.info("Applying advanced feature engineering...")
        
        # Create derived features
        features['speed_acceleration'] = features['speed'].diff().fillna(0)
        features['signal_to_noise_ratio'] = features['signal_strength'] / (features['satellite_count'] + 1)
        features['gps_reliability_score'] = (
            features['gps_fix_quality'] * features['satellite_count'] / 
            (abs(features['signal_strength']) + 1)
        )
        
        # Temporal features (simulated)
        features['hour_of_day'] = np.random.randint(0, 24, len(features))
        features['day_of_week'] = np.random.randint(0, 7, len(features))
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        features['is_rush_hour'] = (
            ((features['hour_of_day'] >= 7) & (features['hour_of_day'] <= 9)) |
            ((features['hour_of_day'] >= 17) & (features['hour_of_day'] <= 19))
        ).astype(int)
        
        # Statistical rolling features (simulated)
        for col in ['speed', 'signal_strength', 'satellite_count']:
            features[f'{col}_rolling_mean'] = (
                features[col] + np.random.normal(0, features[col].std() * 0.1, len(features))
            )
            features[f'{col}_rolling_std'] = abs(
                np.random.normal(features[col].std(), features[col].std() * 0.2, len(features))
            )
        
        # Geographic clustering features
        from sklearn.cluster import KMeans
        coords = features[['latitude', 'longitude']].values
        kmeans = KMeans(n_clusters=10, random_state=self.random_state)
        features['location_cluster'] = kmeans.fit_predict(coords)
        
        # Distance from cluster center
        cluster_centers = kmeans.cluster_centers_
        features['distance_from_cluster_center'] = np.array([
            np.linalg.norm(coords[i] - cluster_centers[features['location_cluster'].iloc[i]])
            for i in range(len(features))
        ])
        
        # One-hot encode categorical features
        categorical_features = ['scenario_type']
        features = pd.get_dummies(features, columns=categorical_features, prefix=categorical_features)
        
        # Remove any infinite or NaN values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(features.median())
        
        self.logger.info(f"Feature engineering completed. Total features: {features.shape[1]}")
        
        return features
    
    def _add_robustness_variations(self, features: pd.DataFrame, 
                                  labels: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """Add variations to improve model robustness."""
        self.logger.info("Adding robustness variations...")
        
        original_size = len(features)
        
        # Add 10% more samples with noise for robustness
        additional_samples = int(original_size * 0.1)
        
        # Randomly select samples to duplicate with noise
        sample_indices = np.random.choice(original_size, additional_samples, replace=True)
        
        additional_features = features.iloc[sample_indices].copy()
        additional_labels = labels[sample_indices].copy()
        
        # Add controlled noise to numerical features
        numerical_cols = features.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col not in ['hour_of_day', 'day_of_week', 'is_weekend', 'is_rush_hour', 'location_cluster']:
                noise_std = additional_features[col].std() * 0.05  # 5% noise
                noise = np.random.normal(0, noise_std, len(additional_features))
                additional_features[col] += noise
        
        # Combine original and additional data
        combined_features = pd.concat([features, additional_features], ignore_index=True)
        combined_labels = np.concatenate([labels, additional_labels])
        
        # Shuffle again
        indices = np.random.permutation(len(combined_features))
        combined_features = combined_features.iloc[indices].reset_index(drop=True)
        combined_labels = combined_labels[indices]
        
        self.logger.info(f"Robustness variations added. New size: {len(combined_features)}")
        
        return combined_features, combined_labels
    
    def _save_dataset(self, features: pd.DataFrame, labels: np.ndarray, output_dir: str):
        """Save the generated dataset."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save features and labels
        features.to_parquet(output_path / "enhanced_features.parquet", index=False)
        np.save(output_path / "enhanced_labels.npy", labels)
        
        # Save metadata
        metadata = {
            'total_samples': len(features),
            'num_features': features.shape[1],
            'jamming_samples': int(np.sum(labels == -1)),
            'normal_samples': int(np.sum(labels == 1)),
            'jamming_ratio': float(np.mean(labels == -1)),
            'feature_names': features.columns.tolist(),
            'generation_timestamp': time.time()
        }
        
        with open(output_path / "enhanced_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Dataset saved to {output_path}")
    
    def train_high_accuracy_models(self, features: pd.DataFrame, 
                                  labels: np.ndarray) -> Dict[str, Any]:
        """Train multiple models with high accuracy focus."""
        self.logger.info("Starting high-accuracy model training...")
        
        # Convert labels to binary (0 = jamming, 1 = normal)
        y_binary = (labels == 1).astype(int)
        
        # Prepare data splits with anti-overfitting measures
        data_splits = self._prepare_anti_overfitting_splits(features, y_binary)
        
        # Define models with anti-overfitting configurations
        models_config = self._get_anti_overfitting_models_config()
        
        # Train each model
        model_results = {}
        for model_name, model_config in models_config.items():
            self.logger.info(f"\n=== Training {model_name} ===")
            
            try:
                result = self._train_single_model(
                    model_name, model_config, data_splits
                )
                model_results[model_name] = result
                
                # Update best model if this one is better
                if result['cv_scores']['f1_mean'] > self.best_score:
                    self.best_score = result['cv_scores']['f1_mean']
                    self.best_model = result['model']
                    self.logger.info(f"New best model: {model_name} (F1: {self.best_score:.4f})")
                    
            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {str(e)}")
                model_results[model_name] = {'error': str(e)}
        
        # Create ensemble from best models
        ensemble_result = self._create_ensemble_model(model_results, data_splits)
        model_results['ensemble'] = ensemble_result
        
        return model_results
    
    def _prepare_anti_overfitting_splits(self, features: pd.DataFrame, 
                                        labels: np.ndarray) -> Dict[str, Any]:
        """Prepare data splits with anti-overfitting measures."""
        self.logger.info("Preparing anti-overfitting data splits...")
        
        # Stratified splits to maintain class distribution
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels,
            test_size=self.anti_overfitting_config['test_size'],
            random_state=self.random_state,
            stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.anti_overfitting_config['validation_size'] / (1 - self.anti_overfitting_config['test_size']),
            random_state=self.random_state,
            stratify=y_temp
        )
        
        # Feature scaling for algorithms that need it
        scaler = RobustScaler()  # More robust to outliers than StandardScaler
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), 
            columns=X_train.columns, 
            index=X_train.index
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val), 
            columns=X_val.columns, 
            index=X_val.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), 
            columns=X_test.columns, 
            index=X_test.index
        )
        
        # Feature selection to reduce overfitting
        feature_selector = SelectKBest(
            f_classif, 
            k=int(len(features.columns) * self.anti_overfitting_config['feature_selection_ratio'])
        )
        X_train_selected = feature_selector.fit_transform(X_train_scaled, y_train)
        X_val_selected = feature_selector.transform(X_val_scaled)
        X_test_selected = feature_selector.transform(X_test_scaled)
        
        selected_features = features.columns[feature_selector.get_support()]
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'X_train_scaled': X_train_scaled,
            'X_val_scaled': X_val_scaled,
            'X_test_scaled': X_test_scaled,
            'X_train_selected': X_train_selected,
            'X_val_selected': X_val_selected,
            'X_test_selected': X_test_selected,
            'scaler': scaler,
            'feature_selector': feature_selector,
            'selected_features': selected_features
        }
    
    def _get_anti_overfitting_models_config(self) -> Dict[str, Dict]:
        """Get model configurations with anti-overfitting measures."""
        config = {}
        
        # Random Forest with strong regularization
        config['random_forest'] = {
            'model_class': RandomForestClassifier,
            'base_params': {
                'n_estimators': 200,
                'max_depth': 15,  # Limited depth to prevent overfitting
                'min_samples_split': 10,  # Higher values prevent overfitting
                'min_samples_leaf': 5,
                'max_features': 'sqrt',  # Feature subsampling
                'bootstrap': True,
                'oob_score': True,  # Out-of-bag validation
                'class_weight': 'balanced',
                'random_state': self.random_state,
                'n_jobs': -1
            },
            'param_search': {
                'n_estimators': [150, 200, 300],
                'max_depth': [10, 15, 20],
                'min_samples_split': [5, 10, 15],
                'min_samples_leaf': [3, 5, 7],
                'max_features': ['sqrt', 'log2', 0.8]
            },
            'use_scaled_data': False
        }
        
        # Gradient Boosting with early stopping
        config['gradient_boosting'] = {
            'model_class': GradientBoostingClassifier,
            'base_params': {
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'subsample': 0.8,  # Stochastic gradient boosting
                'max_features': 'sqrt',
                'validation_fraction': 0.1,
                'n_iter_no_change': 10,  # Early stopping
                'random_state': self.random_state
            },
            'param_search': {
                'n_estimators': [150, 200, 300],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [4, 6, 8],
                'subsample': [0.7, 0.8, 0.9]
            },
            'use_scaled_data': False
        }
        
        # XGBoost with regularization (if available)
        if XGBOOST_AVAILABLE:
            config['xgboost'] = {
                'model_class': xgb.XGBClassifier,
                'base_params': {
                    'n_estimators': 200,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'min_child_weight': 3,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,  # L1 regularization
                    'reg_lambda': 1.0,  # L2 regularization
                    'scale_pos_weight': 10,  # Handle class imbalance
                    'random_state': self.random_state,
                    'n_jobs': -1,
                    'eval_metric': 'logloss'
                },
                'param_search': {
                    'n_estimators': [150, 200, 300],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [4, 6, 8],
                    'reg_alpha': [0.0, 0.1, 0.5],
                    'reg_lambda': [0.5, 1.0, 2.0]
                },
                'use_scaled_data': False,
                'early_stopping': True
            }
        
        # Isolation Forest for anomaly detection
        config['isolation_forest'] = {
            'model_class': IsolationForest,
            'base_params': {
                'n_estimators': 200,
                'contamination': 0.08,  # Expected anomaly rate
                'max_samples': 'auto',
                'max_features': 1.0,
                'bootstrap': False,
                'random_state': self.random_state,
                'n_jobs': -1
            },
            'param_search': {
                'n_estimators': [150, 200, 300],
                'contamination': [0.06, 0.08, 0.10],
                'max_features': [0.8, 0.9, 1.0]
            },
            'use_scaled_data': True,
            'is_anomaly_detector': True
        }
        
        return config
    
    def _train_single_model(self, model_name: str, model_config: Dict, 
                           data_splits: Dict) -> Dict[str, Any]:
        """Train a single model with hyperparameter optimization."""
        self.logger.info(f"Training {model_name}...")
        
        # Select appropriate data
        if model_config.get('use_scaled_data', False):
            X_train = data_splits['X_train_scaled']
            X_val = data_splits['X_val_scaled']
            X_test = data_splits['X_test_scaled']
        else:
            X_train = data_splits['X_train']
            X_val = data_splits['X_val'] 
            X_test = data_splits['X_test']
        
        y_train = data_splits['y_train']
        y_val = data_splits['y_val']
        y_test = data_splits['y_test']
        
        # Hyperparameter optimization
        if BAYESIAN_OPT_AVAILABLE and len(model_config.get('param_search', {})) > 0:
            self.logger.info(f"Performing Bayesian optimization for {model_name}...")
            best_model = self._bayesian_hyperparameter_search(
                model_config, X_train, y_train
            )
        else:
            # Fallback to grid search
            self.logger.info(f"Performing grid search for {model_name}...")
            best_model = self._grid_hyperparameter_search(
                model_config, X_train, y_train
            )
        
        # Cross-validation evaluation
        cv_scores = self._perform_cross_validation(best_model, X_train, y_train)
        
        # Final training and evaluation
        best_model.fit(X_train, y_train)
        
        # Predictions
        if model_config.get('is_anomaly_detector', False):
            # For anomaly detectors, convert predictions
            y_pred_val = best_model.predict(X_val)
            y_pred_test = best_model.predict(X_test)
            # Convert anomaly predictions (-1, 1) to binary (0, 1)
            y_pred_val = (y_pred_val == 1).astype(int)
            y_pred_test = (y_pred_test == 1).astype(int)
        else:
            y_pred_val = best_model.predict(X_val)
            y_pred_test = best_model.predict(X_test)
        
        # Calculate metrics
        val_metrics = self._calculate_metrics(y_val, y_pred_val)
        test_metrics = self._calculate_metrics(y_test, y_pred_test)
        
        return {
            'model': best_model,
            'cv_scores': cv_scores,
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics,
            'model_config': model_config
        }
    
    def _bayesian_hyperparameter_search(self, model_config: Dict, 
                                       X_train: pd.DataFrame, 
                                       y_train: np.ndarray):
        """Perform Bayesian hyperparameter optimization."""
        # Convert param_search to skopt search spaces
        search_spaces = {}
        for param, values in model_config['param_search'].items():
            if isinstance(values[0], int):
                search_spaces[param] = Integer(min(values), max(values))
            elif isinstance(values[0], float):
                search_spaces[param] = Real(min(values), max(values))
            else:
                # Categorical - use RandomizedSearchCV instead
                pass
        
        if search_spaces:
            bayes_search = BayesSearchCV(
                model_config['model_class'](**model_config['base_params']),
                search_spaces,
                n_iter=30,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                scoring='f1',
                n_jobs=-1,
                random_state=self.random_state
            )
            bayes_search.fit(X_train, y_train)
            return bayes_search.best_estimator_
        else:
            return self._grid_hyperparameter_search(model_config, X_train, y_train)
    
    def _grid_hyperparameter_search(self, model_config: Dict, 
                                   X_train: pd.DataFrame, 
                                   y_train: np.ndarray):
        """Perform grid search hyperparameter optimization."""
        grid_search = RandomizedSearchCV(
            model_config['model_class'](**model_config['base_params']),
            model_config.get('param_search', {}),
            n_iter=20,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='f1',
            n_jobs=-1,
            random_state=self.random_state
        )
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    
    def _perform_cross_validation(self, model, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """Perform comprehensive cross-validation."""
        cv = StratifiedKFold(
            n_splits=self.anti_overfitting_config['cv_folds'], 
            shuffle=True, 
            random_state=self.random_state
        )
        
        scoring = ['precision', 'recall', 'f1', 'roc_auc']
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        return {
            'precision_mean': cv_results['test_precision'].mean(),
            'precision_std': cv_results['test_precision'].std(),
            'recall_mean': cv_results['test_recall'].mean(),
            'recall_std': cv_results['test_recall'].std(),
            'f1_mean': cv_results['test_f1'].mean(),
            'f1_std': cv_results['test_f1'].std(),
            'roc_auc_mean': cv_results['test_roc_auc'].mean(),
            'roc_auc_std': cv_results['test_roc_auc'].std()
        }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        return {
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
        }
    
    def _create_ensemble_model(self, model_results: Dict, data_splits: Dict) -> Dict[str, Any]:
        """Create ensemble model from best performing models."""
        self.logger.info("Creating ensemble model...")
        
        # Select models that performed well and are diverse
        good_models = []
        for name, result in model_results.items():
            if 'error' not in result and result['cv_scores']['f1_mean'] > 0.80:
                good_models.append((name, result['model']))
        
        if len(good_models) < 2:
            self.logger.warning("Not enough good models for ensemble")
            return {'error': 'Insufficient models for ensemble'}
        
        # Create voting classifier
        voting_clf = VotingClassifier(
            estimators=good_models,
            voting='soft'  # Use probabilities for voting
        )
        
        # Train ensemble
        X_train = data_splits['X_train_scaled']
        y_train = data_splits['y_train']
        X_val = data_splits['X_val_scaled']
        y_val = data_splits['y_val']
        X_test = data_splits['X_test_scaled']
        y_test = data_splits['y_test']
        
        voting_clf.fit(X_train, y_train)
        
        # Evaluate ensemble
        cv_scores = self._perform_cross_validation(voting_clf, X_train, y_train)
        
        y_pred_val = voting_clf.predict(X_val)
        y_pred_test = voting_clf.predict(X_test)
        
        val_metrics = self._calculate_metrics(y_val, y_pred_val)
        test_metrics = self._calculate_metrics(y_test, y_pred_test)
        
        return {
            'model': voting_clf,
            'cv_scores': cv_scores,
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics,
            'component_models': [name for name, _ in good_models]
        }
    
    def save_best_model(self, model_results: Dict, output_dir: str):
        """Save the best performing model."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find best model
        best_model_name = None
        best_f1_score = 0.0
        
        for name, result in model_results.items():
            if 'error' not in result:
                f1_score = result['cv_scores']['f1_mean']
                if f1_score > best_f1_score:
                    best_f1_score = f1_score
                    best_model_name = name
        
        if best_model_name:
            best_result = model_results[best_model_name]
            
            # Save model
            model_file = output_path / f"best_model_{best_model_name}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(best_result['model'], f)
            
            # Save results
            results_file = output_path / "training_results.json"
            
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = {}
            for name, result in model_results.items():
                if 'error' not in result:
                    serializable_results[name] = {
                        'cv_scores': {k: float(v) for k, v in result['cv_scores'].items()},
                        'validation_metrics': {k: float(v) for k, v in result['validation_metrics'].items()},
                        'test_metrics': {k: float(v) for k, v in result['test_metrics'].items()}
                    }
                else:
                    serializable_results[name] = result
            
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            self.logger.info(f"Best model ({best_model_name}) saved with F1 score: {best_f1_score:.4f}")
            self.logger.info(f"Model saved to: {model_file}")
            self.logger.info(f"Results saved to: {results_file}")
        
        return best_model_name, best_f1_score
    
    def run_complete_workflow(self, output_dir: str = "results/high_accuracy_training"):
        """Run the complete high-accuracy training workflow."""
        self.logger.info("=" * 80)
        self.logger.info("STARTING HIGH-ACCURACY GPS JAMMING DETECTION TRAINING WORKFLOW")
        self.logger.info("=" * 80)
        
        workflow_start_time = time.time()
        
        try:
            # Step 1: Generate enhanced dataset
            self.logger.info("\n" + "=" * 50)
            self.logger.info("STEP 1: ENHANCED DATASET GENERATION")
            self.logger.info("=" * 50)
            
            data_dir = Path(output_dir) / "data"
            features, labels = self.generate_enhanced_dataset(str(data_dir))
            
            # Step 2: Train high-accuracy models
            self.logger.info("\n" + "=" * 50)
            self.logger.info("STEP 2: HIGH-ACCURACY MODEL TRAINING")
            self.logger.info("=" * 50)
            
            model_results = self.train_high_accuracy_models(features, labels)
            
            # Step 3: Save best model
            self.logger.info("\n" + "=" * 50)
            self.logger.info("STEP 3: MODEL SELECTION AND SAVING")
            self.logger.info("=" * 50)
            
            models_dir = Path(output_dir) / "models"
            best_model_name, best_score = self.save_best_model(model_results, str(models_dir))
            
            # Final summary
            workflow_duration = time.time() - workflow_start_time
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("WORKFLOW COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 80)
            self.logger.info(f"Total Duration: {workflow_duration:.2f} seconds ({workflow_duration/3600:.2f} hours)")
            self.logger.info(f"Best Model: {best_model_name}")
            self.logger.info(f"Best F1 Score: {best_score:.4f}")
            self.logger.info(f"Dataset Size: {len(features)} samples")
            self.logger.info(f"Features: {features.shape[1]}")
            
            # Check if target accuracy achieved
            if best_score >= self.target_accuracy:
                self.logger.info(f"🎉 TARGET ACCURACY ACHIEVED! ({best_score:.4f} >= {self.target_accuracy})")
            else:
                self.logger.info(f"Target accuracy not yet achieved ({best_score:.4f} < {self.target_accuracy})")
                self.logger.info("Consider running additional iterations or increasing dataset size")
            
            return {
                'success': True,
                'best_model': best_model_name,
                'best_score': best_score,
                'dataset_size': len(features),
                'num_features': features.shape[1],
                'duration': workflow_duration,
                'all_results': model_results
            }
            
        except Exception as e:
            self.logger.error(f"Workflow failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'duration': time.time() - workflow_start_time
            }


def main():
    """Main function to run the high-accuracy training workflow."""
    parser = argparse.ArgumentParser(
        description="High-Accuracy GPS Jamming Detection Training Workflow"
    )
    parser.add_argument(
        '--target-samples', type=int, default=60000,
        help='Target number of training samples (default: 60000)'
    )
    parser.add_argument(
        '--target-accuracy', type=float, default=0.95,
        help='Target accuracy threshold (default: 0.95)'
    )
    parser.add_argument(
        '--max-iterations', type=int, default=10,
        help='Maximum training iterations (default: 10)'
    )
    parser.add_argument(
        '--output-dir', type=str, default="results/high_accuracy_training",
        help='Output directory for results (default: results/high_accuracy_training)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--random-state', type=int, default=42,
        help='Random state for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Initialize workflow
    workflow = HighAccuracyTrainingWorkflow(
        target_samples=args.target_samples,
        target_accuracy=args.target_accuracy,
        max_iterations=args.max_iterations,
        random_state=args.random_state
    )
    
    # Run workflow
    result = workflow.run_complete_workflow(args.output_dir)
    
    # Exit with appropriate code
    if result['success']:
        print(f"\n Workflow completed successfully!")
        print(f"Best Model: {result['best_model']}")
        print(f"Best F1 Score: {result['best_score']:.4f}")
        exit(0)
    else:
        print(f"\nWorkflow failed: {result['error']}")
        exit(1)


if __name__ == "__main__":
    main() 