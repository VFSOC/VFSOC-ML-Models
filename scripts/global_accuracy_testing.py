#!/usr/bin/env python3
"""
Global Accuracy Testing for GPS Jamming Detection Models.

This script provides comprehensive evaluation of trained models with detailed
metrics, statistical analysis, and performance comparisons.
"""

import sys
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vfsoc_ml.models.base_model import BaseJammingDetector

# ML evaluation imports
from sklearn.model_selection import (
    cross_val_score, 
    cross_validate,
    StratifiedKFold,
    RepeatedStratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score
)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


class GlobalAccuracyTester:
    """
    Comprehensive accuracy testing system for GPS jamming detection models.
    """
    
    def __init__(self, 
                 cv_folds: int = 10,
                 cv_repeats: int = 3,
                 random_state: int = 42,
                 confidence_level: float = 0.95):
        """
        Initialize the global accuracy tester.
        
        Args:
            cv_folds: Number of cross-validation folds
            cv_repeats: Number of CV repetitions for robust evaluation
            random_state: Random state for reproducibility
            confidence_level: Confidence level for statistical tests
        """
        self.cv_folds = cv_folds
        self.cv_repeats = cv_repeats
        self.random_state = random_state
        self.confidence_level = confidence_level
        
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.model_performances = {}
        self.statistical_tests = {}
        self.model_rankings = {}
        
    def load_models_and_data(self, models_dir: str, data_dir: str) -> Tuple[Dict[str, Any], pd.DataFrame, np.ndarray]:
        """Load trained models and test data."""
        self.logger.info("Loading trained models and test data...")
        
        models_path = Path(models_dir)
        data_path = Path(data_dir)
        
        # Load models
        models = {}
        model_files = list(models_path.glob("*_model.pkl"))
        
        for model_file in model_files:
            model_name = model_file.stem.replace("_model", "")
            try:
                # Import the appropriate model class
                from vfsoc_ml.models.isolation_forest import IsolationForestDetector
                from vfsoc_ml.models.advanced_models import (
                    XGBoostDetector, LightGBMDetector, EnhancedRandomForestDetector,
                    NeuralNetworkDetector, EnsembleDetector
                )
                
                # Load model based on type
                if "isolation_forest" in model_name:
                    model = IsolationForestDetector()
                elif "xgboost" in model_name:
                    model = XGBoostDetector()
                elif "lightgbm" in model_name:
                    model = LightGBMDetector()
                elif "random_forest" in model_name:
                    model = EnhancedRandomForestDetector()
                elif "neural_network" in model_name:
                    model = NeuralNetworkDetector()
                else:
                    # Try generic loading
                    model = BaseJammingDetector("Unknown")
                
                model.load_model(model_file)
                models[model_name] = model
                self.logger.info(f"Loaded model: {model_name}")
                
            except Exception as e:
                self.logger.warning(f"Failed to load model {model_name}: {e}")
                continue
        
        # Load test data
        if (data_path / "features.parquet").exists():
            features = pd.read_parquet(data_path / "features.parquet")
            labels = np.load(data_path / "labels.npy")
        else:
            # Fallback to CSV or other formats
            sample_file = data_path / "sample_data.csv"
            if sample_file.exists():
                df = pd.read_csv(sample_file)
                labels = df['label'].values
                features = df.drop('label', axis=1)
            else:
                raise FileNotFoundError(f"No test data found in {data_path}")
        
        self.logger.info(f"Loaded test data: {len(features)} samples, {features.shape[1]} features")
        
        return models, features, labels
    
    def comprehensive_model_evaluation(self, model: BaseJammingDetector, 
                                     features: pd.DataFrame, labels: np.ndarray,
                                     model_name: str) -> Dict[str, Any]:
        """Perform comprehensive evaluation of a single model."""
        self.logger.info(f"Evaluating model: {model_name}")
        
        results = {
            'model_name': model_name,
            'evaluation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Convert labels for evaluation (ensure binary format)
        y_binary = (labels == 1).astype(int)
        
        try:
            # Basic predictions
            predictions = model.predict(features)
            pred_binary = (predictions == 1).astype(int)
            
            # Basic metrics
            basic_metrics = self._calculate_basic_metrics(y_binary, pred_binary)
            results['basic_metrics'] = basic_metrics
            
            # Cross-validation evaluation
            cv_results = self._cross_validation_evaluation(model, features, y_binary)
            results['cross_validation'] = cv_results
            
            # Advanced metrics
            advanced_metrics = self._calculate_advanced_metrics(y_binary, pred_binary, model, features)
            results['advanced_metrics'] = advanced_metrics
            
            # Performance analysis
            performance_analysis = self._performance_analysis(y_binary, pred_binary)
            results['performance_analysis'] = performance_analysis
            
            # Model-specific metrics
            model_specific = self._model_specific_metrics(model, features, y_binary)
            results['model_specific'] = model_specific
            
            self.logger.info(f"✓ {model_name} evaluation completed")
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate {model_name}: {e}")
            results['error'] = str(e)
        
        return results
    
    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate basic classification metrics."""
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_true, y_pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
            'matthews_corrcoef': float(matthews_corrcoef(y_true, y_pred)),
            'cohen_kappa': float(cohen_kappa_score(y_true, y_pred))
        }
    
    def _cross_validation_evaluation(self, model: BaseJammingDetector, 
                                   features: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
        """Perform repeated stratified cross-validation."""
        self.logger.info("Performing cross-validation...")
        
        # Create cross-validation strategy
        cv = RepeatedStratifiedKFold(
            n_splits=self.cv_folds,
            n_repeats=self.cv_repeats,
            random_state=self.random_state
        )
        
        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc',
            'balanced_accuracy': 'balanced_accuracy'
        }
        
        try:
            # Perform cross-validation
            cv_results = cross_validate(
                model.model,  # Use the underlying sklearn model
                features,
                labels,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                return_train_score=True
            )
            
            # Calculate statistics
            cv_stats = {}
            for metric in scoring.keys():
                test_scores = cv_results[f'test_{metric}']
                train_scores = cv_results[f'train_{metric}']
                
                cv_stats[metric] = {
                    'test_mean': float(np.mean(test_scores)),
                    'test_std': float(np.std(test_scores)),
                    'test_min': float(np.min(test_scores)),
                    'test_max': float(np.max(test_scores)),
                    'train_mean': float(np.mean(train_scores)),
                    'train_std': float(np.std(train_scores)),
                    'overfitting': float(np.mean(train_scores) - np.mean(test_scores))
                }
                
                # Confidence intervals
                confidence_interval = stats.t.interval(
                    self.confidence_level,
                    len(test_scores) - 1,
                    loc=np.mean(test_scores),
                    scale=stats.sem(test_scores)
                )
                cv_stats[metric]['confidence_interval'] = [float(confidence_interval[0]), float(confidence_interval[1])]
            
            return cv_stats
            
        except Exception as e:
            self.logger.warning(f"Cross-validation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_advanced_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  model: BaseJammingDetector, features: pd.DataFrame) -> Dict[str, Any]:
        """Calculate advanced performance metrics."""
        metrics = {}
        
        # Confusion matrix analysis
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        metrics['confusion_matrix'] = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
        
        # Derived metrics
        total = len(y_true)
        metrics['error_analysis'] = {
            'total_samples': int(total),
            'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            'false_negative_rate': float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
            'true_positive_rate': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            'true_negative_rate': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'positive_predictive_value': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            'negative_predictive_value': float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        }
        
        # Try to get probability predictions for ROC/PR curves
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(features)[:, 1]  # Probability of positive class
            elif hasattr(model, 'get_anomaly_scores'):
                y_scores = model.get_anomaly_scores(features)
                y_proba = (y_scores - np.min(y_scores)) / (np.max(y_scores) - np.min(y_scores))
            else:
                y_proba = y_pred.astype(float)  # Use predictions as proxy
            
            # ROC curve analysis
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
            metrics['roc_analysis'] = {
                'auc': float(roc_auc_score(y_true, y_proba)),
                'optimal_threshold_idx': int(np.argmax(tpr - fpr)),
                'optimal_threshold': float(roc_thresholds[np.argmax(tpr - fpr)]) if len(roc_thresholds) > 0 else 0.5
            }
            
            # Precision-Recall curve analysis
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
            metrics['pr_analysis'] = {
                'average_precision': float(average_precision_score(y_true, y_proba)),
                'f1_optimal_threshold': float(pr_thresholds[np.argmax(2 * precision * recall / (precision + recall + 1e-10))]) if len(pr_thresholds) > 0 else 0.5
            }
            
        except Exception as e:
            self.logger.warning(f"Could not calculate probability-based metrics: {e}")
            metrics['probability_metrics_error'] = str(e)
        
        return metrics
    
    def _performance_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze model performance patterns."""
        analysis = {}
        
        # Class-wise performance
        unique_labels = np.unique(y_true)
        class_performance = {}
        
        for label in unique_labels:
            mask = y_true == label
            class_predictions = y_pred[mask]
            
            class_performance[f'class_{int(label)}'] = {
                'sample_count': int(np.sum(mask)),
                'accuracy': float(np.mean(class_predictions == label)),
                'misclassification_rate': float(np.mean(class_predictions != label))
            }
        
        analysis['class_wise_performance'] = class_performance
        
        # Overall performance summary
        analysis['performance_summary'] = {
            'total_correct': int(np.sum(y_true == y_pred)),
            'total_incorrect': int(np.sum(y_true != y_pred)),
            'overall_accuracy': float(np.mean(y_true == y_pred)),
            'error_rate': float(np.mean(y_true != y_pred))
        }
        
        return analysis
    
    def _model_specific_metrics(self, model: BaseJammingDetector, 
                              features: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
        """Calculate model-specific metrics."""
        metrics = {
            'model_type': model.model_name,
            'is_trained': model.is_trained,
            'training_time': getattr(model, 'training_time', None),
            'feature_count': features.shape[1]
        }
        
        # Inference time analysis
        try:
            inference_times = []
            for _ in range(10):  # Multiple runs for stable timing
                start_time = time.time()
                _ = model.predict(features[:100])  # Sample for timing
                inference_times.append(time.time() - start_time)
            
            metrics['inference_performance'] = {
                'mean_time_100_samples': float(np.mean(inference_times)),
                'std_time_100_samples': float(np.std(inference_times)),
                'estimated_time_per_sample_ms': float(np.mean(inference_times) * 10)  # Convert to ms per sample
            }
        except Exception as e:
            metrics['inference_performance_error'] = str(e)
        
        # Feature importance (if available)
        try:
            if hasattr(model, 'get_feature_importance'):
                importance = model.get_feature_importance()
                if importance:
                    metrics['feature_importance_available'] = True
                    metrics['top_features'] = dict(list(importance.items())[:10])  # Top 10 features
                else:
                    metrics['feature_importance_available'] = False
            else:
                metrics['feature_importance_available'] = False
        except Exception as e:
            metrics['feature_importance_error'] = str(e)
        
        return metrics
    
    def statistical_significance_testing(self, model_performances: Dict[str, Dict]) -> Dict[str, Any]:
        """Perform statistical significance testing between models."""
        self.logger.info("Performing statistical significance testing...")
        
        results = {}
        
        # Extract F1 scores from cross-validation results
        model_f1_scores = {}
        for model_name, performance in model_performances.items():
            if 'cross_validation' in performance and 'f1' in performance['cross_validation']:
                # Extract individual CV scores (not available in current implementation)
                # For now, use confidence intervals
                f1_mean = performance['cross_validation']['f1']['test_mean']
                f1_std = performance['cross_validation']['f1']['test_std']
                model_f1_scores[model_name] = {
                    'mean': f1_mean,
                    'std': f1_std,
                    'confidence_interval': performance['cross_validation']['f1']['confidence_interval']
                }
        
        # Pairwise comparisons
        model_names = list(model_f1_scores.keys())
        pairwise_comparisons = {}
        
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                comparison_key = f"{model1}_vs_{model2}"
                
                # Simple comparison based on confidence intervals
                ci1 = model_f1_scores[model1]['confidence_interval']
                ci2 = model_f1_scores[model2]['confidence_interval']
                
                # Check if confidence intervals overlap
                overlap = not (ci1[1] < ci2[0] or ci2[1] < ci1[0])
                
                pairwise_comparisons[comparison_key] = {
                    'model1': model1,
                    'model2': model2,
                    'model1_f1_mean': model_f1_scores[model1]['mean'],
                    'model2_f1_mean': model_f1_scores[model2]['mean'],
                    'confidence_intervals_overlap': overlap,
                    'statistically_significant': not overlap,
                    'better_model': model1 if model_f1_scores[model1]['mean'] > model_f1_scores[model2]['mean'] else model2
                }
        
        results['pairwise_comparisons'] = pairwise_comparisons
        
        # Overall ranking
        ranking = sorted(model_f1_scores.items(), key=lambda x: x[1]['mean'], reverse=True)
        results['model_ranking'] = [
            {
                'rank': i + 1,
                'model': model_name,
                'f1_score': scores['mean'],
                'confidence_interval': scores['confidence_interval']
            }
            for i, (model_name, scores) in enumerate(ranking)
        ]
        
        return results
    
    def generate_comprehensive_report(self, models: Dict[str, Any], 
                                    model_performances: Dict[str, Dict],
                                    statistical_tests: Dict[str, Any],
                                    output_dir: str):
        """Generate comprehensive testing report."""
        self.logger.info("Generating comprehensive accuracy testing report...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate summary report
        report = {
            'testing_summary': {
                'testing_timestamp': pd.Timestamp.now().isoformat(),
                'models_tested': len(models),
                'cv_folds': self.cv_folds,
                'cv_repeats': self.cv_repeats,
                'confidence_level': self.confidence_level
            },
            'model_performances': model_performances,
            'statistical_analysis': statistical_tests
        }
        
        # Save detailed JSON report
        report_path = output_path / "global_accuracy_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        self.logger.info(f"Detailed report saved to {report_path}")
        
        # Generate summary CSV
        summary_data = []
        for model_name, performance in model_performances.items():
            if 'basic_metrics' in performance:
                row = {
                    'Model': model_name,
                    'Accuracy': performance['basic_metrics']['accuracy'],
                    'Precision': performance['basic_metrics']['precision'],
                    'Recall': performance['basic_metrics']['recall'],
                    'F1_Score': performance['basic_metrics']['f1_score'],
                    'ROC_AUC': performance['basic_metrics']['roc_auc'],
                    'Matthews_Corrcoef': performance['basic_metrics']['matthews_corrcoef']
                }
                
                # Add CV metrics if available
                if 'cross_validation' in performance and 'f1' in performance['cross_validation']:
                    row['CV_F1_Mean'] = performance['cross_validation']['f1']['test_mean']
                    row['CV_F1_Std'] = performance['cross_validation']['f1']['test_std']
                    row['Overfitting'] = performance['cross_validation']['f1']['overfitting']
                
                summary_data.append(row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values('F1_Score', ascending=False)
            
            summary_csv_path = output_path / "model_comparison_summary.csv"
            summary_df.to_csv(summary_csv_path, index=False)
            self.logger.info(f"Summary CSV saved to {summary_csv_path}")
            
            # Display summary
            self.logger.info("\n=== Model Performance Summary ===")
            self.logger.info(summary_df.to_string(index=False))
        
        # Generate best model recommendation
        if statistical_tests and 'model_ranking' in statistical_tests:
            best_model = statistical_tests['model_ranking'][0]
            self.logger.info(f"\n=== Best Model Recommendation ===")
            self.logger.info(f"Best Model: {best_model['model']}")
            self.logger.info(f"F1 Score: {best_model['f1_score']:.4f}")
            self.logger.info(f"Confidence Interval: [{best_model['confidence_interval'][0]:.4f}, {best_model['confidence_interval'][1]:.4f}]")
    
    def run_global_testing(self, models_dir: str, data_dir: str, output_dir: str):
        """Run comprehensive global accuracy testing."""
        self.logger.info("=== Starting Global Accuracy Testing ===")
        
        # Load models and data
        models, features, labels = self.load_models_and_data(models_dir, data_dir)
        
        if not models:
            self.logger.error("No models found to test!")
            return
        
        # Evaluate each model
        model_performances = {}
        for model_name, model in models.items():
            performance = self.comprehensive_model_evaluation(model, features, labels, model_name)
            model_performances[model_name] = performance
        
        # Statistical significance testing
        statistical_tests = self.statistical_significance_testing(model_performances)
        
        # Generate comprehensive report
        self.generate_comprehensive_report(models, model_performances, statistical_tests, output_dir)
        
        self.logger.info("=== Global Accuracy Testing Complete ===")


def main():
    """Main function for global accuracy testing."""
    parser = argparse.ArgumentParser(description="Global accuracy testing for GPS jamming detection models")
    parser.add_argument("--models", "-m", type=str, required=True,
                       help="Directory containing trained models")
    parser.add_argument("--data", "-d", type=str, required=True,
                       help="Directory containing test data")
    parser.add_argument("--output", "-o", type=str, default="results/accuracy_testing",
                       help="Output directory for testing results")
    parser.add_argument("--cv-folds", type=int, default=10,
                       help="Number of cross-validation folds")
    parser.add_argument("--cv-repeats", type=int, default=3,
                       help="Number of CV repetitions")
    parser.add_argument("--confidence", type=float, default=0.95,
                       help="Confidence level for statistical tests")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("=== Global Accuracy Testing for GPS Jamming Detection ===")
    logger.info(f"Models directory: {args.models}")
    logger.info(f"Test data directory: {args.data}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"CV configuration: {args.cv_folds} folds, {args.cv_repeats} repeats")
    logger.info(f"Confidence level: {args.confidence}")
    
    try:
        # Initialize tester
        tester = GlobalAccuracyTester(
            cv_folds=args.cv_folds,
            cv_repeats=args.cv_repeats,
            confidence_level=args.confidence
        )
        
        # Run global testing
        tester.run_global_testing(args.models, args.data, args.output)
        
        logger.info("Global accuracy testing completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Global accuracy testing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main()) 