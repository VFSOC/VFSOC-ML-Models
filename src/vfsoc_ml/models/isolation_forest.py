"""
Isolation Forest model for GPS jamming detection.

This module implements an Isolation Forest-based anomaly detection model
specifically designed for identifying GPS jamming patterns in vehicle telemetry data.
Based on research showing Isolation Forest's effectiveness for unsupervised
anomaly detection in time-series and signal processing applications.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import ParameterGrid
import logging

from .base_model import BaseJammingDetector

logger = logging.getLogger(__name__)


class IsolationForestDetector(BaseJammingDetector):
    """
    Isolation Forest-based GPS jamming detection model.
    
    This model uses the Isolation Forest algorithm to detect anomalous patterns
    in GPS signal data that may indicate jamming attacks. The algorithm works
    by isolating anomalies instead of profiling normal data points.
    
    Key advantages for GPS jamming detection:
    - Unsupervised learning (no labeled jamming data required)
    - Effective for high-dimensional feature spaces
    - Fast training and inference
    - Robust to noise and signal variations
    - Well-suited for real-time applications
    
    References:
    - Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. 
      IEEE ICDM.
    - Breunig, M. M., et al. (2000). LOF: identifying density-based 
      local outliers. ACM SIGMOD.
    """
    
    def __init__(self, 
                 n_estimators: int = 200,
                 max_samples: Union[int, float, str] = "auto",
                 contamination: float = 0.05,
                 max_features: float = 1.0,
                 bootstrap: bool = False,
                 n_jobs: int = -1,
                 random_state: int = 42,
                 verbose: int = 0,
                 preprocessing: str = "robust",
                 **kwargs):
        """
        Initialize the Isolation Forest GPS jamming detector.
        
        Args:
            n_estimators: Number of base estimators in the ensemble
            max_samples: Number of samples to draw to train each base estimator
            contamination: Expected proportion of outliers in the data
            max_features: Number of features to draw to train each base estimator
            bootstrap: Whether to sample with replacement
            n_jobs: Number of parallel jobs to run
            random_state: Random state for reproducibility
            verbose: Verbosity level
            preprocessing: Preprocessing method ('standard', 'robust', 'none')
            **kwargs: Additional parameters
        """
        super().__init__(model_name="IsolationForest", **kwargs)
        
        # Model parameters
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.preprocessing = preprocessing
        
        # Store all parameters
        self.model_params.update({
            'n_estimators': n_estimators,
            'max_samples': max_samples,
            'contamination': contamination,
            'max_features': max_features,
            'bootstrap': bootstrap,
            'n_jobs': n_jobs,
            'random_state': random_state,
            'verbose': verbose,
            'preprocessing': preprocessing
        })
        
        # Preprocessing components
        self.scaler = None
        self._setup_preprocessing()
        
        # Model-specific attributes
        self.feature_importances_ = None
        self.anomaly_threshold_ = None
        
    def _setup_preprocessing(self) -> None:
        """Setup the preprocessing pipeline."""
        if self.preprocessing == "standard":
            self.scaler = StandardScaler()
        elif self.preprocessing == "robust":
            self.scaler = RobustScaler()
        elif self.preprocessing == "none":
            self.scaler = None
        else:
            raise ValueError(f"Unknown preprocessing method: {self.preprocessing}")
    
    def _create_model(self) -> IsolationForest:
        """
        Create and return the Isolation Forest model.
        
        Returns:
            Configured IsolationForest instance
        """
        model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose
        )
        
        logger.info(f"Created Isolation Forest with parameters: {self.model_params}")
        return model
    
    def _train_model(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Train the Isolation Forest model.
        
        Args:
            X: Training features
            y: Not used (unsupervised learning)
        """
        # Preprocess features
        if self.scaler is not None:
            X_processed = self.scaler.fit_transform(X)
            logger.info(f"Applied {self.preprocessing} scaling to features")
        else:
            X_processed = X
        
        # Train the model
        self.model.fit(X_processed)
        
        # Calculate anomaly threshold based on decision function
        decision_scores = self.model.decision_function(X_processed)
        self.anomaly_threshold_ = np.percentile(decision_scores, 
                                               self.contamination * 100)
        
        logger.info(f"Training completed. Anomaly threshold: {self.anomaly_threshold_:.4f}")
        
        # Compute feature importances (approximation)
        self._compute_feature_importances(X_processed)
    
    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions (1 for normal, -1 for jamming/anomaly)
        """
        # Preprocess features
        if self.scaler is not None:
            X_processed = self.scaler.transform(X)
        else:
            X_processed = X
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        return predictions
    
    def get_anomaly_scores(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get anomaly scores for the input data.
        
        Args:
            X: Input features
            
        Returns:
            Anomaly scores (lower scores indicate higher anomaly likelihood)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting anomaly scores")
        
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Preprocess features
        if self.scaler is not None:
            X_processed = self.scaler.transform(X)
        else:
            X_processed = X
        
        # Get decision function scores
        scores = self.model.decision_function(X_processed)
        
        return scores
    
    def get_jamming_probability(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get GPS jamming probability estimates.
        
        Args:
            X: Input features
            
        Returns:
            Jamming probabilities (0-1 scale)
        """
        anomaly_scores = self.get_anomaly_scores(X)
        
        # Convert anomaly scores to probabilities
        # Lower scores (more negative) indicate higher anomaly likelihood
        # Use sigmoid transformation for probability-like output
        probabilities = 1 / (1 + np.exp(anomaly_scores))
        
        return probabilities
    
    def _compute_feature_importances(self, X: np.ndarray) -> None:
        """
        Compute approximate feature importances for interpretability.
        
        Args:
            X: Training features
        """
        if X.shape[1] == 0:
            self.feature_importances_ = np.array([])
            return
        
        n_features = X.shape[1]
        importances = np.zeros(n_features)
        
        # For each feature, measure how much its perturbation affects anomaly scores
        baseline_scores = self.model.decision_function(X)
        
        for i in range(n_features):
            # Create perturbed version by shuffling feature i
            X_perturbed = X.copy()
            np.random.shuffle(X_perturbed[:, i])
            
            # Calculate new scores
            perturbed_scores = self.model.decision_function(X_perturbed)
            
            # Importance is the change in average anomaly score
            importances[i] = np.mean(np.abs(baseline_scores - perturbed_scores))
        
        # Normalize importances
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        
        self.feature_importances_ = importances
        
        logger.info(f"Computed feature importances for {n_features} features")
    
    def get_feature_importances(self) -> Optional[np.ndarray]:
        """
        Get feature importances.
        
        Returns:
            Feature importances array or None if not computed
        """
        return self.feature_importances_
    
    def explain_prediction(self, X: Union[np.ndarray, pd.DataFrame], 
                          sample_idx: int = 0) -> Dict[str, Any]:
        """
        Explain a specific prediction.
        
        Args:
            X: Input features
            sample_idx: Index of the sample to explain
            
        Returns:
            Dictionary containing explanation information
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before explaining predictions")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if sample_idx >= len(X):
            raise ValueError(f"Sample index {sample_idx} out of range")
        
        sample = X[sample_idx:sample_idx+1]
        
        # Get prediction and anomaly score
        prediction = self.predict(sample)[0]
        anomaly_score = self.get_anomaly_scores(sample)[0]
        jamming_prob = self.get_jamming_probability(sample)[0]
        
        explanation = {
            'sample_index': sample_idx,
            'prediction': 'Jamming' if prediction == -1 else 'Normal',
            'anomaly_score': float(anomaly_score),
            'jamming_probability': float(jamming_prob),
            'threshold': float(self.anomaly_threshold_) if self.anomaly_threshold_ is not None else None,
        }
        
        # Add feature contributions if available
        if self.feature_importances_ is not None and self.feature_names is not None:
            feature_contributions = {}
            for i, (feature_name, importance) in enumerate(zip(self.feature_names, 
                                                               self.feature_importances_)):
                feature_contributions[feature_name] = {
                    'value': float(sample[0, i]),
                    'importance': float(importance)
                }
            explanation['feature_contributions'] = feature_contributions
        
        return explanation
    
    def tune_hyperparameters(self, X: Union[np.ndarray, pd.DataFrame],
                           param_grid: Optional[Dict[str, List]] = None,
                           cv_folds: int = 3,
                           scoring_metric: str = 'anomaly_detection',
                           n_jobs: int = -1) -> Dict[str, Any]:
        """
        Tune hyperparameters using cross-validation.
        
        Args:
            X: Training data
            param_grid: Parameter grid for search
            cv_folds: Number of cross-validation folds
            scoring_metric: Metric to optimize
            n_jobs: Number of parallel jobs
            
        Returns:
            Best parameters and scores
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'contamination': [0.03, 0.05, 0.07, 0.1],
                'max_features': [0.8, 1.0],
                'max_samples': ['auto', 0.8, 1.0]
            }
        
        logger.info(f"Starting hyperparameter tuning with {len(list(ParameterGrid(param_grid)))} combinations")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        best_score = float('-inf')
        best_params = None
        all_results = []
        
        for params in ParameterGrid(param_grid):
            scores = []
            
            # Perform cross-validation
            fold_size = len(X) // cv_folds
            
            for fold in range(cv_folds):
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size if fold < cv_folds - 1 else len(X)
                
                # Split data
                train_indices = list(range(0, start_idx)) + list(range(end_idx, len(X)))
                test_indices = list(range(start_idx, end_idx))
                
                X_train_fold = X[train_indices]
                X_test_fold = X[test_indices]
                
                # Create and train model with current parameters
                temp_model = IsolationForestDetector(**params, random_state=self.random_state)
                temp_model.train(X_train_fold)
                
                # Score the model
                if scoring_metric == 'anomaly_detection':
                    # Use silhouette-like score for unsupervised evaluation
                    anomaly_scores = temp_model.get_anomaly_scores(X_test_fold)
                    score = -np.mean(anomaly_scores)  # Lower scores are better for anomalies
                else:
                    score = 0  # Placeholder for other metrics
                
                scores.append(score)
            
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            
            result = {
                'params': params,
                'mean_score': avg_score,
                'std_score': std_score,
                'scores': scores
            }
            all_results.append(result)
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = params
            
            logger.info(f"Params: {params}, Score: {avg_score:.4f} ± {std_score:.4f}")
        
        logger.info(f"Best parameters: {best_params}, Best score: {best_score:.4f}")
        
        # Update model with best parameters
        for param, value in best_params.items():
            setattr(self, param, value)
        
        # Recreate model with best parameters
        self.model = self._create_model()
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': all_results
        }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the model.
        
        Returns:
            Model summary dictionary
        """
        summary = self.get_model_info()
        
        if self.is_trained:
            summary.update({
                'anomaly_threshold': float(self.anomaly_threshold_) if self.anomaly_threshold_ is not None else None,
                'n_features': len(self.feature_names) if self.feature_names else None,
                'contamination_rate': self.contamination,
                'preprocessing_method': self.preprocessing,
            })
            
            if self.feature_importances_ is not None:
                summary['feature_importances_available'] = True
                if self.feature_names:
                    top_features = np.argsort(self.feature_importances_)[-5:][::-1]
                    summary['top_5_features'] = [
                        {
                            'feature': self.feature_names[i],
                            'importance': float(self.feature_importances_[i])
                        }
                        for i in top_features
                    ]
        
        return summary 