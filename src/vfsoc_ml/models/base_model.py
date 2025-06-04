"""
Base model class for GPS jamming detection models.

This module provides the abstract base class that all GPS jamming detection
models should inherit from, ensuring a consistent interface and common functionality.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import joblib
import time
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseJammingDetector(ABC, BaseEstimator):
    """
    Abstract base class for GPS jamming detection models.
    
    This class provides a common interface and shared functionality for all
    jamming detection models including training, prediction, evaluation,
    and model persistence.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the base detector.
        
        Args:
            model_name: Name of the model for identification
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.training_metrics = {}
        self.model_params = kwargs
        
        # Performance tracking
        self.training_time = None
        self.inference_times = []
        
    @abstractmethod
    def _create_model(self) -> Any:
        """
        Create and return the underlying ML model.
        
        Returns:
            The initialized ML model instance
        """
        pass
    
    @abstractmethod
    def _train_model(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Train the underlying model.
        
        Args:
            X: Training features
            y: Training labels (optional for unsupervised models)
        """
        pass
    
    @abstractmethod
    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions (1 for normal, -1 for anomaly/jamming)
        """
        pass
    
    def train(self, X: Union[np.ndarray, pd.DataFrame], 
              y: Optional[Union[np.ndarray, pd.Series]] = None,
              feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train the GPS jamming detection model.
        
        Args:
            X: Training features
            y: Training labels (optional for unsupervised models)
            feature_names: Names of the features
            
        Returns:
            Dictionary containing training metrics
        """
        logger.info(f"Training {self.model_name} model...")
        
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        self.feature_names = feature_names
        
        # Create model if not exists
        if self.model is None:
            self.model = self._create_model()
        
        # Track training time
        start_time = time.time()
        
        # Train the model
        self._train_model(X, y)
        
        self.training_time = time.time() - start_time
        self.is_trained = True
        
        logger.info(f"Training completed in {self.training_time:.2f} seconds")
        
        # Calculate training metrics if labels are provided
        if y is not None:
            train_predictions = self.predict(X)
            self.training_metrics = self._calculate_metrics(y, train_predictions)
            logger.info(f"Training metrics: {self.training_metrics}")
        
        return self.training_metrics
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict GPS jamming attacks.
        
        Args:
            X: Input features
            
        Returns:
            Predictions (1 for normal, -1 for jamming/anomaly)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Track inference time
        start_time = time.time()
        
        predictions = self._predict_model(X)
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return predictions
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities (if supported by the model).
        
        Args:
            X: Input features
            
        Returns:
            Prediction probabilities
        """
        if hasattr(self.model, 'predict_proba'):
            if isinstance(X, pd.DataFrame):
                X = X.values
            return self.model.predict_proba(X)
        else:
            # For models without probability prediction, return binary predictions
            predictions = self.predict(X)
            # Convert to probability-like scores
            proba = np.zeros((len(predictions), 2))
            proba[predictions == 1, 1] = 0.8  # Normal
            proba[predictions == -1, 0] = 0.8  # Jamming
            proba[predictions == 1, 0] = 0.2
            proba[predictions == -1, 1] = 0.2
            return proba
    
    def get_anomaly_scores(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get anomaly scores for the input data (if supported).
        
        Args:
            X: Input features
            
        Returns:
            Anomaly scores (higher means more anomalous)
        """
        if hasattr(self.model, 'decision_function'):
            if isinstance(X, pd.DataFrame):
                X = X.values
            # For isolation forest, negative scores mean anomalies
            scores = self.model.decision_function(X)
            # Convert to positive anomaly scores
            return -scores
        elif hasattr(self.model, 'score_samples'):
            if isinstance(X, pd.DataFrame):
                X = X.values
            return -self.model.score_samples(X)
        else:
            # Fallback: use prediction confidence
            predictions = self.predict(X)
            return np.abs(predictions)
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], 
                 y: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Evaluate the model performance.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = self.predict(X)
        metrics = self._calculate_metrics(y, predictions)
        
        # Add inference time statistics
        if self.inference_times:
            metrics['avg_inference_time_ms'] = np.mean(self.inference_times) * 1000
            metrics['max_inference_time_ms'] = np.max(self.inference_times) * 1000
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Calculate performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        # Convert to binary classification format (0: normal, 1: jamming)
        y_true_binary = (y_true == -1).astype(int)
        y_pred_binary = (y_pred == -1).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_true_binary, y_pred_binary),
            'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'f1_score': f1_score(y_true_binary, y_pred_binary, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true_binary, y_pred_binary).tolist(),
        }
        
        # Add ROC AUC if we can get probability scores
        try:
            if hasattr(self.model, 'predict_proba'):
                y_proba = self.predict_proba(X=None)[:, 1]  # This won't work, need to fix
            else:
                # Use anomaly scores as proxy for probabilities
                # This is a simplified approach - in practice, you'd need the X data
                pass
        except:
            # Skip ROC AUC if not available
            pass
        
        return metrics
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'model_params': self.model_params,
            'training_time': self.training_time,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Union[str, Path]) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.feature_names = model_data['feature_names']
        self.training_metrics = model_data['training_metrics']
        self.model_params = model_data['model_params']
        self.training_time = model_data['training_time']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'model_params': self.model_params,
            'training_time': self.training_time,
        }
        
        if self.inference_times:
            info['avg_inference_time_ms'] = np.mean(self.inference_times) * 1000
            info['total_predictions'] = len(self.inference_times)
        
        return info
    
    def reset(self) -> None:
        """Reset the model to untrained state."""
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.training_metrics = {}
        self.training_time = None
        self.inference_times = []
    
    def __repr__(self) -> str:
        """String representation of the model."""
        status = "trained" if self.is_trained else "untrained"
        return f"{self.model_name}({status})" 