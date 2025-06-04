"""
Advanced ML Models for GPS Jamming Detection.

This module implements state-of-the-art machine learning models specifically
optimized for GPS jamming detection, including gradient boosting, ensemble methods,
and neural networks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from abc import ABC, abstractmethod
import joblib
import time
from pathlib import Path

# Import base model
from .base_model import BaseJammingDetector

# ML Models
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier,
    BaggingClassifier
)
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Advanced models (with fallback)
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

# Deep learning (with fallback)
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logger = logging.getLogger(__name__)


class XGBoostDetector(BaseJammingDetector):
    """
    XGBoost-based GPS jamming detector.
    
    Uses gradient boosting with extreme gradient boosting for high performance
    anomaly detection and classification.
    """
    
    def __init__(self, 
                 n_estimators: int = 200,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 scale_pos_weight: float = 10.0,  # Handle class imbalance
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize XGBoost detector.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            subsample: Fraction of samples used for training each tree
            colsample_bytree: Fraction of features used for training each tree
            scale_pos_weight: Weight for minority class (jamming)
            random_state: Random state for reproducibility
        """
        super().__init__("XGBoost_Detector", **kwargs)
        
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        
    def _create_model(self) -> xgb.XGBClassifier:
        """Create XGBoost classifier."""
        return xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )
    
    def _train_model(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Train XGBoost model."""
        if y is None:
            raise ValueError("XGBoost requires labeled data for training")
        
        # Convert labels to binary (0, 1) format
        y_binary = (y == 1).astype(int)
        
        self.model.fit(X, y_binary)
    
    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with XGBoost."""
        predictions = self.model.predict(X)
        # Convert back to anomaly detection format (-1 for jamming, 1 for normal)
        return np.where(predictions == 0, -1, 1)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained XGBoost model."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        importance_dict = self.model.get_booster().get_score(importance_type='weight')
        return importance_dict


class LightGBMDetector(BaseJammingDetector):
    """
    LightGBM-based GPS jamming detector.
    
    Uses gradient boosting with LightGBM for fast and accurate anomaly detection.
    """
    
    def __init__(self,
                 n_estimators: int = 200,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 num_leaves: int = 31,
                 feature_fraction: float = 0.8,
                 bagging_fraction: float = 0.8,
                 bagging_freq: int = 5,
                 class_weight: str = 'balanced',
                 random_state: int = 42,
                 **kwargs):
        """Initialize LightGBM detector."""
        super().__init__("LightGBM_Detector", **kwargs)
        
        if not LGB_AVAILABLE:
            raise ImportError("LightGBM not available. Install with: pip install lightgbm")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.class_weight = class_weight
        self.random_state = random_state
    
    def _create_model(self) -> lgb.LGBMClassifier:
        """Create LightGBM classifier."""
        return lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            feature_fraction=self.feature_fraction,
            bagging_fraction=self.bagging_fraction,
            bagging_freq=self.bagging_freq,
            class_weight=self.class_weight,
            random_state=self.random_state,
            verbosity=-1
        )
    
    def _train_model(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Train LightGBM model."""
        if y is None:
            raise ValueError("LightGBM requires labeled data for training")
        
        # Convert labels to binary format
        y_binary = (y == 1).astype(int)
        
        self.model.fit(X, y_binary)
    
    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with LightGBM."""
        predictions = self.model.predict(X)
        return np.where(predictions == 0, -1, 1)


class EnhancedRandomForestDetector(BaseJammingDetector):
    """
    Enhanced Random Forest detector with advanced configurations.
    """
    
    def __init__(self,
                 n_estimators: int = 300,
                 max_depth: int = 15,
                 min_samples_split: int = 5,
                 min_samples_leaf: int = 2,
                 max_features: str = 'sqrt',
                 class_weight: str = 'balanced_subsample',
                 bootstrap: bool = True,
                 oob_score: bool = True,
                 n_jobs: int = -1,
                 random_state: int = 42,
                 **kwargs):
        """Initialize Enhanced Random Forest detector."""
        super().__init__("Enhanced_RandomForest_Detector", **kwargs)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
    
    def _create_model(self) -> RandomForestClassifier:
        """Create Enhanced Random Forest classifier."""
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
    
    def _train_model(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Train Enhanced Random Forest."""
        if y is None:
            raise ValueError("Random Forest requires labeled data for training")
        
        y_binary = (y == 1).astype(int)
        self.model.fit(X, y_binary)
    
    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with Enhanced Random Forest."""
        predictions = self.model.predict(X)
        return np.where(predictions == 0, -1, 1)
    
    def get_oob_score(self) -> float:
        """Get out-of-bag score."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get OOB score")
        return self.model.oob_score_


class NeuralNetworkDetector(BaseJammingDetector):
    """
    Multi-layer Perceptron for GPS jamming detection.
    """
    
    def __init__(self,
                 hidden_layer_sizes: Tuple[int, ...] = (100, 50, 25),
                 activation: str = 'relu',
                 solver: str = 'adam',
                 alpha: float = 0.001,
                 learning_rate_init: float = 0.001,
                 max_iter: int = 500,
                 early_stopping: bool = True,
                 validation_fraction: float = 0.1,
                 random_state: int = 42,
                 **kwargs):
        """Initialize Neural Network detector."""
        super().__init__("NeuralNetwork_Detector", **kwargs)
        
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.random_state = random_state
    
    def _create_model(self) -> MLPClassifier:
        """Create MLP classifier."""
        return MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            random_state=self.random_state
        )
    
    def _train_model(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Train Neural Network."""
        if y is None:
            raise ValueError("Neural Network requires labeled data for training")
        
        y_binary = (y == 1).astype(int)
        self.model.fit(X, y_binary)
    
    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with Neural Network."""
        predictions = self.model.predict(X)
        return np.where(predictions == 0, -1, 1)


class EnsembleDetector(BaseJammingDetector):
    """
    Ensemble detector combining multiple models for superior performance.
    """
    
    def __init__(self,
                 models: Optional[List[BaseJammingDetector]] = None,
                 voting: str = 'soft',
                 weights: Optional[List[float]] = None,
                 **kwargs):
        """
        Initialize Ensemble detector.
        
        Args:
            models: List of base models to ensemble
            voting: Voting strategy ('hard' or 'soft')
            weights: Weights for each model
        """
        super().__init__("Ensemble_Detector", **kwargs)
        
        # Default models if none provided
        if models is None:
            if XGB_AVAILABLE:
                xgb_model = XGBoostDetector()
            else:
                xgb_model = None
            
            if LGB_AVAILABLE:
                lgb_model = LightGBMDetector()
            else:
                lgb_model = None
            
            rf_model = EnhancedRandomForestDetector()
            
            # Filter out None models
            self.base_models = [m for m in [xgb_model, lgb_model, rf_model] if m is not None]
        else:
            self.base_models = models
        
        self.voting = voting
        self.weights = weights
        
        logger.info(f"Ensemble initialized with {len(self.base_models)} base models")
    
    def _create_model(self) -> VotingClassifier:
        """Create ensemble voting classifier."""
        # Create estimators list
        estimators = []
        for i, model in enumerate(self.base_models):
            model_name = f"model_{i}_{model.model_name}"
            if model.model is None:
                model.model = model._create_model()
            estimators.append((model_name, model.model))
        
        return VotingClassifier(
            estimators=estimators,
            voting=self.voting,
            weights=self.weights
        )
    
    def _train_model(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Train ensemble model."""
        if y is None:
            raise ValueError("Ensemble requires labeled data for training")
        
        y_binary = (y == 1).astype(int)
        
        # Train individual models first
        for model in self.base_models:
            logger.info(f"Training base model: {model.model_name}")
            model.train(X, y)
        
        # Train ensemble
        self.model.fit(X, y_binary)
    
    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        predictions = self.model.predict(X)
        return np.where(predictions == 0, -1, 1)
    
    def get_base_model_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from individual base models."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before getting base predictions")
        
        predictions = {}
        for model in self.base_models:
            predictions[model.model_name] = model.predict(X)
        
        return predictions


class AutoMLDetector(BaseJammingDetector):
    """
    Automated ML detector that tries multiple models and selects the best one.
    """
    
    def __init__(self,
                 cv_folds: int = 5,
                 scoring: str = 'f1',
                 n_jobs: int = -1,
                 **kwargs):
        """Initialize AutoML detector."""
        super().__init__("AutoML_Detector", **kwargs)
        
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.n_jobs = n_jobs
        
        # Model candidates
        self.candidate_models = self._get_candidate_models()
        self.best_model = None
        self.model_scores = {}
    
    def _get_candidate_models(self) -> List[BaseJammingDetector]:
        """Get list of candidate models for AutoML."""
        candidates = []
        
        # Add available models
        if XGB_AVAILABLE:
            candidates.append(XGBoostDetector())
        
        if LGB_AVAILABLE:
            candidates.append(LightGBMDetector())
        
        candidates.extend([
            EnhancedRandomForestDetector(),
            NeuralNetworkDetector()
        ])
        
        return candidates
    
    def _create_model(self):
        """AutoML doesn't create a single model upfront."""
        return None
    
    def _train_model(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Train and select best model using cross-validation."""
        if y is None:
            raise ValueError("AutoML requires labeled data for training")
        
        from sklearn.model_selection import cross_val_score
        
        logger.info(f"AutoML: Evaluating {len(self.candidate_models)} candidate models...")
        
        best_score = -np.inf
        
        for model in self.candidate_models:
            try:
                logger.info(f"Evaluating {model.model_name}...")
                
                # Create and prepare model
                model.model = model._create_model()
                
                # Convert labels for sklearn compatibility
                y_binary = (y == 1).astype(int)
                
                # Cross-validation evaluation
                scores = cross_val_score(
                    model.model, X, y_binary,
                    cv=self.cv_folds,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs
                )
                
                mean_score = np.mean(scores)
                self.model_scores[model.model_name] = {
                    'mean_score': mean_score,
                    'std_score': np.std(scores),
                    'scores': scores
                }
                
                logger.info(f"{model.model_name}: {mean_score:.4f} ± {np.std(scores):.4f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    self.best_model = model
                
            except Exception as e:
                logger.warning(f"Failed to evaluate {model.model_name}: {e}")
                continue
        
        if self.best_model is None:
            raise ValueError("No models could be successfully evaluated")
        
        logger.info(f"AutoML: Best model is {self.best_model.model_name} with score {best_score:.4f}")
        
        # Train the best model on full data
        self.best_model.train(X, y)
        self.model = self.best_model.model
    
    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the best model."""
        if self.best_model is None:
            raise ValueError("AutoML must be trained before making predictions")
        
        return self.best_model.predict(X)
    
    def get_model_evaluation_summary(self) -> pd.DataFrame:
        """Get summary of model evaluation results."""
        if not self.model_scores:
            raise ValueError("No model evaluation results available")
        
        summary_data = []
        for model_name, scores in self.model_scores.items():
            summary_data.append({
                'Model': model_name,
                'Mean_Score': scores['mean_score'],
                'Std_Score': scores['std_score'],
                'Is_Best': model_name == self.best_model.model_name if self.best_model else False
            })
        
        return pd.DataFrame(summary_data).sort_values('Mean_Score', ascending=False) 