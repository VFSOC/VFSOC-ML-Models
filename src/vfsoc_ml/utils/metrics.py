"""
Metrics utilities for anomaly detection evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, silhouette_score, confusion_matrix
)
import warnings

warnings.filterwarnings('ignore')


def calculate_anomaly_metrics(
    anomaly_scores: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
    predicted_labels: Optional[np.ndarray] = None,
    contamination: float = 0.05
) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for anomaly detection.
    
    Args:
        anomaly_scores: Anomaly scores from the model
        true_labels: True binary labels (1=normal, 0=anomaly) - optional for unsupervised
        predicted_labels: Predicted binary labels (1=normal, 0=anomaly)
        contamination: Expected proportion of anomalies
        
    Returns:
        Dictionary containing various metrics
    """
    metrics = {}
    
    # If no predicted labels provided, create them from scores
    if predicted_labels is None:
        threshold = np.percentile(anomaly_scores, contamination * 100)
        predicted_labels = (anomaly_scores >= threshold).astype(int)
    
    # Basic statistics
    metrics['anomaly_rate'] = 1 - np.mean(predicted_labels)
    metrics['mean_anomaly_score'] = np.mean(anomaly_scores)
    metrics['std_anomaly_score'] = np.std(anomaly_scores)
    metrics['min_anomaly_score'] = np.min(anomaly_scores)
    metrics['max_anomaly_score'] = np.max(anomaly_scores)
    
    # If true labels are available (supervised evaluation)
    if true_labels is not None:
        # Convert to binary format if needed (assuming -1/1 format from isolation forest)
        if np.min(predicted_labels) == -1:
            predicted_labels = (predicted_labels == 1).astype(int)
        if np.min(true_labels) == -1:
            true_labels = (true_labels == 1).astype(int)
        
        # Classification metrics
        metrics['precision'] = precision_score(true_labels, predicted_labels, zero_division=0)
        metrics['recall'] = recall_score(true_labels, predicted_labels, zero_division=0)
        metrics['f1_score'] = f1_score(true_labels, predicted_labels, zero_division=0)
        
        # ROC AUC (if we have probability scores)
        try:
            # Convert anomaly scores to probabilities (higher score = more normal)
            probabilities = (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores))
            metrics['roc_auc'] = roc_auc_score(true_labels, probabilities)
            metrics['average_precision'] = average_precision_score(true_labels, probabilities)
        except:
            metrics['roc_auc'] = 0.5
            metrics['average_precision'] = np.mean(true_labels)
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
        metrics['true_positives'] = tp
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        
        # Additional metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
    
    # Unsupervised metrics (always available)
    try:
        # Silhouette score (for clustering quality)
        if len(np.unique(predicted_labels)) > 1:
            # Create feature matrix from scores for silhouette calculation
            X_scores = anomaly_scores.reshape(-1, 1)
            metrics['silhouette_score'] = silhouette_score(X_scores, predicted_labels)
        else:
            metrics['silhouette_score'] = 0.0
    except:
        metrics['silhouette_score'] = 0.0
    
    # Anomaly score distribution metrics
    normal_mask = predicted_labels == 1
    anomaly_mask = predicted_labels == 0
    
    if np.sum(normal_mask) > 0:
        metrics['normal_score_mean'] = np.mean(anomaly_scores[normal_mask])
        metrics['normal_score_std'] = np.std(anomaly_scores[normal_mask])
    else:
        metrics['normal_score_mean'] = 0.0
        metrics['normal_score_std'] = 0.0
    
    if np.sum(anomaly_mask) > 0:
        metrics['anomaly_score_mean'] = np.mean(anomaly_scores[anomaly_mask])
        metrics['anomaly_score_std'] = np.std(anomaly_scores[anomaly_mask])
    else:
        metrics['anomaly_score_mean'] = 0.0
        metrics['anomaly_score_std'] = 0.0
    
    # Score separation
    if np.sum(normal_mask) > 0 and np.sum(anomaly_mask) > 0:
        metrics['score_separation'] = abs(metrics['normal_score_mean'] - metrics['anomaly_score_mean'])
    else:
        metrics['score_separation'] = 0.0
    
    return metrics


def evaluate_contamination_rates(
    anomaly_scores: np.ndarray,
    contamination_rates: list,
    true_labels: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Evaluate model performance across different contamination rates.
    
    Args:
        anomaly_scores: Anomaly scores from the model
        contamination_rates: List of contamination rates to test
        true_labels: True binary labels (optional)
        
    Returns:
        DataFrame with metrics for each contamination rate
    """
    results = []
    
    for contamination in contamination_rates:
        # Create predicted labels based on contamination rate
        threshold = np.percentile(anomaly_scores, contamination * 100)
        predicted_labels = (anomaly_scores >= threshold).astype(int)
        
        # Calculate metrics
        metrics = calculate_anomaly_metrics(
            anomaly_scores, true_labels, predicted_labels, contamination
        )
        
        # Add contamination rate to metrics
        metrics['contamination_rate'] = contamination
        results.append(metrics)
    
    return pd.DataFrame(results)


def calculate_feature_importance_proxy(
    model,
    X: np.ndarray,
    feature_names: list,
    n_iterations: int = 10
) -> Dict[str, float]:
    """
    Calculate feature importance proxy for isolation forest using permutation.
    
    Args:
        model: Trained isolation forest model
        X: Feature matrix
        feature_names: List of feature names
        n_iterations: Number of permutation iterations
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    # Get baseline scores
    baseline_scores = model.decision_function(X)
    baseline_mean = np.mean(baseline_scores)
    
    importance_scores = {}
    
    for i, feature_name in enumerate(feature_names):
        importance_values = []
        
        for _ in range(n_iterations):
            # Create permuted version
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            
            # Get scores with permuted feature
            permuted_scores = model.decision_function(X_permuted)
            permuted_mean = np.mean(permuted_scores)
            
            # Calculate importance as change in mean score
            importance = abs(baseline_mean - permuted_mean)
            importance_values.append(importance)
        
        # Average importance across iterations
        importance_scores[feature_name] = np.mean(importance_values)
    
    # Normalize importance scores
    max_importance = max(importance_scores.values()) if importance_scores else 1
    if max_importance > 0:
        importance_scores = {k: v / max_importance for k, v in importance_scores.items()}
    
    return importance_scores


def get_anomaly_summary(
    df: pd.DataFrame,
    anomaly_labels: np.ndarray,
    top_features: Optional[list] = None
) -> Dict[str, Any]:
    """
    Get summary statistics for detected anomalies.
    
    Args:
        df: Original dataframe with features
        anomaly_labels: Binary labels (1=normal, 0=anomaly)
        top_features: List of most important features to summarize
        
    Returns:
        Dictionary containing anomaly summary
    """
    anomaly_mask = anomaly_labels == 0
    normal_mask = anomaly_labels == 1
    
    summary = {
        'total_samples': len(df),
        'anomalies_detected': np.sum(anomaly_mask),
        'anomaly_rate': np.mean(anomaly_mask),
        'normal_samples': np.sum(normal_mask)
    }
    
    if top_features and len(top_features) > 0:
        # Compare feature distributions between normal and anomalous samples
        feature_comparison = {}
        
        for feature in top_features:
            if feature in df.columns:
                normal_values = df.loc[normal_mask, feature]
                anomaly_values = df.loc[anomaly_mask, feature]
                
                feature_comparison[feature] = {
                    'normal_mean': normal_values.mean() if len(normal_values) > 0 else 0,
                    'anomaly_mean': anomaly_values.mean() if len(anomaly_values) > 0 else 0,
                    'normal_std': normal_values.std() if len(normal_values) > 0 else 0,
                    'anomaly_std': anomaly_values.std() if len(anomaly_values) > 0 else 0,
                    'difference': abs(normal_values.mean() - anomaly_values.mean()) if len(normal_values) > 0 and len(anomaly_values) > 0 else 0
                }
        
        summary['feature_comparison'] = feature_comparison
    
    return summary 