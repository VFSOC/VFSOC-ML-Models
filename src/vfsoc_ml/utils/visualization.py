"""
Visualization utilities for anomaly detection analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")


def plot_anomaly_analysis(
    model,
    X_test: pd.DataFrame,
    original_data: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: tuple = (15, 12)
) -> None:
    """
    Create comprehensive anomaly analysis plots.
    
    Args:
        model: Trained isolation forest model
        X_test: Test features
        original_data: Original data with all columns
        save_path: Optional path to save plots
        figsize: Figure size
    """
    # Get predictions and scores
    anomaly_scores = model.decision_function(X_test)
    anomaly_labels = model.predict(X_test)
    
    # Convert labels to binary (1=normal, 0=anomaly)
    binary_labels = (anomaly_labels == 1).astype(int)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Energy Consumption Anomaly Detection Analysis', fontsize=16, fontweight='bold')
    
    # 1. Anomaly Score Distribution
    axes[0, 0].hist(anomaly_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(np.mean(anomaly_scores), color='red', linestyle='--', label=f'Mean: {np.mean(anomaly_scores):.3f}')
    axes[0, 0].set_xlabel('Anomaly Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Anomaly Scores')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Normal vs Anomaly Score Comparison
    normal_scores = anomaly_scores[binary_labels == 1]
    anomaly_scores_subset = anomaly_scores[binary_labels == 0]
    
    axes[0, 1].boxplot([normal_scores, anomaly_scores_subset], 
                       labels=['Normal', 'Anomaly'],
                       patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
    axes[0, 1].set_ylabel('Anomaly Score')
    axes[0, 1].set_title('Score Distribution by Class')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Energy Consumption vs Anomaly Score
    if 'kwhTotal' in original_data.columns:
        scatter_colors = ['red' if label == 0 else 'blue' for label in binary_labels]
        axes[0, 2].scatter(original_data['kwhTotal'].iloc[X_test.index], 
                          anomaly_scores, 
                          c=scatter_colors, alpha=0.6, s=20)
        axes[0, 2].set_xlabel('Energy Consumption (kWh)')
        axes[0, 2].set_ylabel('Anomaly Score')
        axes[0, 2].set_title('Energy vs Anomaly Score')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Normal'),
                          Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Anomaly')]
        axes[0, 2].legend(handles=legend_elements)
    
    # 4. Temporal Pattern Analysis
    if 'hour_of_day' in X_test.columns:
        hourly_anomaly_rate = []
        hours = range(24)
        
        for hour in hours:
            hour_mask = X_test['hour_of_day'] == hour
            if np.sum(hour_mask) > 0:
                hour_anomaly_rate = 1 - np.mean(binary_labels[hour_mask])
            else:
                hour_anomaly_rate = 0
            hourly_anomaly_rate.append(hour_anomaly_rate)
        
        axes[1, 0].bar(hours, hourly_anomaly_rate, alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Hour of Day')
        axes[1, 0].set_ylabel('Anomaly Rate')
        axes[1, 0].set_title('Anomaly Rate by Hour of Day')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xticks(range(0, 24, 4))
    
    # 5. Feature Importance (Top 10)
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    else:
        # Calculate proxy importance for isolation forest
        from ..utils.metrics import calculate_feature_importance_proxy
        feature_importance_dict = calculate_feature_importance_proxy(
            model, X_test.values, X_test.columns.tolist(), n_iterations=5
        )
        feature_importance = list(feature_importance_dict.values())
    
    # Get top 10 features
    feature_names = X_test.columns.tolist()
    top_indices = np.argsort(feature_importance)[-10:]
    top_features = [feature_names[i] for i in top_indices]
    top_importance = [feature_importance[i] for i in top_indices]
    
    axes[1, 1].barh(range(len(top_features)), top_importance, alpha=0.7, color='green')
    axes[1, 1].set_yticks(range(len(top_features)))
    axes[1, 1].set_yticklabels(top_features)
    axes[1, 1].set_xlabel('Importance Score')
    axes[1, 1].set_title('Top 10 Feature Importance')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Anomaly Rate by Energy Category
    if 'kwhTotal' in original_data.columns:
        # Create energy categories
        energy_data = original_data['kwhTotal'].iloc[X_test.index]
        energy_categories = pd.cut(energy_data, 
                                 bins=[0, 5, 15, 30, float('inf')], 
                                 labels=['Low (0-5)', 'Medium (5-15)', 'High (15-30)', 'Very High (30+)'])
        
        category_anomaly_rates = []
        category_labels = []
        
        for category in energy_categories.cat.categories:
            category_mask = energy_categories == category
            if np.sum(category_mask) > 0:
                category_anomaly_rate = 1 - np.mean(binary_labels[category_mask])
                category_anomaly_rates.append(category_anomaly_rate)
                category_labels.append(f"{category}\n(n={np.sum(category_mask)})")
        
        axes[1, 2].bar(range(len(category_labels)), category_anomaly_rates, 
                       alpha=0.7, color='purple')
        axes[1, 2].set_xticks(range(len(category_labels)))
        axes[1, 2].set_xticklabels(category_labels, rotation=45, ha='right')
        axes[1, 2].set_ylabel('Anomaly Rate')
        axes[1, 2].set_title('Anomaly Rate by Energy Category')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / 'anomaly_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Anomaly analysis plot saved to: {save_dir / 'anomaly_analysis.png'}")
    
    plt.show()


def plot_energy_consumption_patterns(
    df: pd.DataFrame,
    anomaly_labels: np.ndarray,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8)
) -> None:
    """
    Plot energy consumption patterns for normal vs anomalous sessions.
    
    Args:
        df: Original dataframe
        anomaly_labels: Binary labels (1=normal, 0=anomaly)
        save_path: Optional path to save plots
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Energy Consumption Patterns: Normal vs Anomalous Sessions', fontsize=14, fontweight='bold')
    
    normal_mask = anomaly_labels == 1
    anomaly_mask = anomaly_labels == 0
    
    # 1. Energy Distribution
    if 'kwhTotal' in df.columns:
        axes[0, 0].hist(df.loc[normal_mask, 'kwhTotal'], bins=30, alpha=0.7, 
                       label='Normal', color='blue', density=True)
        axes[0, 0].hist(df.loc[anomaly_mask, 'kwhTotal'], bins=30, alpha=0.7, 
                       label='Anomaly', color='red', density=True)
        axes[0, 0].set_xlabel('Energy Consumption (kWh)')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Energy Consumption Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Charging Duration Distribution
    if 'chargeTimeHrs' in df.columns:
        axes[0, 1].hist(df.loc[normal_mask, 'chargeTimeHrs'], bins=30, alpha=0.7, 
                       label='Normal', color='blue', density=True)
        axes[0, 1].hist(df.loc[anomaly_mask, 'chargeTimeHrs'], bins=30, alpha=0.7, 
                       label='Anomaly', color='red', density=True)
        axes[0, 1].set_xlabel('Charging Duration (hours)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Charging Duration Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Energy Rate Distribution
    if 'kwhTotal' in df.columns and 'chargeTimeHrs' in df.columns:
        energy_rate = df['kwhTotal'] / (df['chargeTimeHrs'] + 1e-6)
        axes[1, 0].hist(energy_rate[normal_mask], bins=30, alpha=0.7, 
                       label='Normal', color='blue', density=True)
        axes[1, 0].hist(energy_rate[anomaly_mask], bins=30, alpha=0.7, 
                       label='Anomaly', color='red', density=True)
        axes[1, 0].set_xlabel('Energy Rate (kWh/hour)')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Energy Rate Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Energy vs Duration Scatter
    if 'kwhTotal' in df.columns and 'chargeTimeHrs' in df.columns:
        axes[1, 1].scatter(df.loc[normal_mask, 'chargeTimeHrs'], 
                          df.loc[normal_mask, 'kwhTotal'], 
                          alpha=0.6, label='Normal', color='blue', s=20)
        axes[1, 1].scatter(df.loc[anomaly_mask, 'chargeTimeHrs'], 
                          df.loc[anomaly_mask, 'kwhTotal'], 
                          alpha=0.8, label='Anomaly', color='red', s=30)
        axes[1, 1].set_xlabel('Charging Duration (hours)')
        axes[1, 1].set_ylabel('Energy Consumption (kWh)')
        axes[1, 1].set_title('Energy vs Duration')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / 'energy_patterns.png', dpi=300, bbox_inches='tight')
        print(f"Energy patterns plot saved to: {save_dir / 'energy_patterns.png'}")
    
    plt.show()


def plot_station_analysis(
    df: pd.DataFrame,
    anomaly_labels: np.ndarray,
    save_path: Optional[str] = None,
    top_n: int = 10,
    figsize: tuple = (12, 8)
) -> None:
    """
    Plot station-level analysis of anomalies.
    
    Args:
        df: Original dataframe
        anomaly_labels: Binary labels (1=normal, 0=anomaly)
        save_path: Optional path to save plots
        top_n: Number of top stations to show
        figsize: Figure size
    """
    if 'stationId' not in df.columns:
        print("Station analysis requires 'stationId' column")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Station-Level Anomaly Analysis', fontsize=14, fontweight='bold')
    
    # Calculate station-level metrics
    station_stats = df.groupby('stationId').agg({
        'kwhTotal': ['count', 'mean', 'std'],
        'chargeTimeHrs': 'mean' if 'chargeTimeHrs' in df.columns else 'count'
    }).round(3)
    
    # Calculate anomaly rates by station
    station_anomaly_rates = []
    station_ids = []
    
    for station_id in df['stationId'].unique():
        station_mask = df['stationId'] == station_id
        station_anomaly_rate = 1 - np.mean(anomaly_labels[station_mask])
        station_anomaly_rates.append(station_anomaly_rate)
        station_ids.append(station_id)
    
    station_anomaly_df = pd.DataFrame({
        'stationId': station_ids,
        'anomaly_rate': station_anomaly_rates,
        'session_count': [np.sum(df['stationId'] == sid) for sid in station_ids]
    })
    
    # 1. Top stations by anomaly rate
    top_anomaly_stations = station_anomaly_df.nlargest(top_n, 'anomaly_rate')
    axes[0, 0].bar(range(len(top_anomaly_stations)), top_anomaly_stations['anomaly_rate'], 
                   alpha=0.7, color='red')
    axes[0, 0].set_xticks(range(len(top_anomaly_stations)))
    axes[0, 0].set_xticklabels([f"S{sid}" for sid in top_anomaly_stations['stationId']], 
                              rotation=45)
    axes[0, 0].set_ylabel('Anomaly Rate')
    axes[0, 0].set_title(f'Top {top_n} Stations by Anomaly Rate')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Session count vs anomaly rate
    axes[0, 1].scatter(station_anomaly_df['session_count'], 
                      station_anomaly_df['anomaly_rate'], 
                      alpha=0.6, color='purple')
    axes[0, 1].set_xlabel('Number of Sessions')
    axes[0, 1].set_ylabel('Anomaly Rate')
    axes[0, 1].set_title('Session Count vs Anomaly Rate')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Distribution of anomaly rates across stations
    axes[1, 0].hist(station_anomaly_rates, bins=20, alpha=0.7, color='orange')
    axes[1, 0].set_xlabel('Anomaly Rate')
    axes[1, 0].set_ylabel('Number of Stations')
    axes[1, 0].set_title('Distribution of Station Anomaly Rates')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Average energy consumption by station (top stations)
    if 'kwhTotal' in df.columns:
        top_energy_stations = df.groupby('stationId')['kwhTotal'].mean().nlargest(top_n)
        axes[1, 1].bar(range(len(top_energy_stations)), top_energy_stations.values, 
                       alpha=0.7, color='green')
        axes[1, 1].set_xticks(range(len(top_energy_stations)))
        axes[1, 1].set_xticklabels([f"S{sid}" for sid in top_energy_stations.index], 
                                  rotation=45)
        axes[1, 1].set_ylabel('Average Energy (kWh)')
        axes[1, 1].set_title(f'Top {top_n} Stations by Average Energy')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / 'station_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Station analysis plot saved to: {save_dir / 'station_analysis.png'}")
    
    plt.show()


def create_anomaly_report(
    model,
    X_test: pd.DataFrame,
    original_data: pd.DataFrame,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a comprehensive anomaly detection report.
    
    Args:
        model: Trained model
        X_test: Test features
        original_data: Original data
        save_path: Optional path to save report
        
    Returns:
        Dictionary containing report data
    """
    # Generate all plots
    if save_path:
        plot_anomaly_analysis(model, X_test, original_data, save_path)
        
        anomaly_labels = model.predict(X_test)
        binary_labels = (anomaly_labels == 1).astype(int)
        
        plot_energy_consumption_patterns(
            original_data.iloc[X_test.index], binary_labels, save_path
        )
        plot_station_analysis(
            original_data.iloc[X_test.index], binary_labels, save_path
        )
    
    # Calculate summary statistics
    anomaly_scores = model.decision_function(X_test)
    anomaly_labels = model.predict(X_test)
    binary_labels = (anomaly_labels == 1).astype(int)
    
    from ..utils.metrics import calculate_anomaly_metrics, get_anomaly_summary
    
    metrics = calculate_anomaly_metrics(anomaly_scores, predicted_labels=binary_labels)
    summary = get_anomaly_summary(
        original_data.iloc[X_test.index], 
        binary_labels, 
        X_test.columns[:10].tolist()  # Top 10 features
    )
    
    report = {
        'metrics': metrics,
        'summary': summary,
        'model_info': {
            'model_type': type(model).__name__,
            'n_features': X_test.shape[1],
            'n_samples': X_test.shape[0]
        }
    }
    
    return report 