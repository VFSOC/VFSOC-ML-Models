"""
Energy Consumption Feature Engineering

This module creates features for detecting irregular energy consumption patterns
in EV charging stations, focusing on meter tampering, unauthorized power drain,
broken billing logic, and station configuration errors.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class EnergyConsumptionFeatureEngineer:
    """
    Feature engineer for EV charging station energy consumption data.
    
    Creates core features for anomaly detection:
    - energy: Energy delivered in the session (kWh)
    - billing_per_kWh: Derived feature (billing ÷ energy)
    - vehicle_mean_energy: Historical average per vehicle
    - z_score_energy: (energy - mean) / std_dev
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature engineer.
        
        Args:
            config: Configuration dictionary containing feature parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.feature_config = config['features']
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features for energy consumption anomaly detection.
        
        Args:
            df: Preprocessed charging session data
            
        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Starting feature engineering for irregular energy consumption detection...")
        
        # Start with a copy of the data
        features_df = df.copy()
        
        # Create core energy features (primary focus)
        features_df = self._create_core_energy_features(features_df)
        
        # Create vehicle-specific features
        features_df = self._create_vehicle_features(features_df)
        
        # Create station context features
        features_df = self._create_station_features(features_df)
        
        # Create billing and cost features
        features_df = self._create_billing_features(features_df)
        
        # Create additional energy features
        features_df = self._create_additional_energy_features(features_df)
        
        # Select only numeric features for ML
        numeric_features = self._select_numeric_features(features_df)
        
        self.logger.info(f"Feature engineering complete. Created {len(numeric_features.columns)} features")
        
        return numeric_features
    
    def _create_core_energy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create the core energy features as specified in the ML approach."""
        self.logger.info("Creating core energy features...")
        
        # 1. Energy - Primary feature (energy delivered in session)
        if 'kwhTotal' in df.columns:
            df['energy'] = df['kwhTotal']
        elif 'kWhDelivered' in df.columns:
            df['energy'] = df['kWhDelivered']
        else:
            self.logger.warning("No energy column found, using placeholder")
            df['energy'] = 0.0
        
        # 2. Billing per kWh (derived feature: billing ÷ energy)
        if 'dollars' in df.columns and 'energy' in df.columns:
            # Avoid division by zero
            df['billing_per_kWh'] = df['dollars'] / (df['energy'] + 1e-6)
            # Cap unrealistic values
            df['billing_per_kWh'] = df['billing_per_kWh'].clip(0, 2.0)  # Max $2/kWh
        else:
            # Use default rate if no billing data
            df['billing_per_kWh'] = 0.15  # Default $0.15/kWh
        
        # 3. Vehicle mean energy (historical average per vehicle)
        if 'userId' in df.columns:  # Using userId as vehicle_id proxy
            vehicle_mean_energy = df.groupby('userId')['energy'].transform('mean')
            df['vehicle_mean_energy'] = vehicle_mean_energy
        else:
            df['vehicle_mean_energy'] = df['energy'].mean()
        
        # 4. Z-score energy ((energy - mean) / std_dev per vehicle)
        if 'userId' in df.columns:
            vehicle_energy_mean = df.groupby('userId')['energy'].transform('mean')
            vehicle_energy_std = df.groupby('userId')['energy'].transform('std')
            
            # Avoid division by zero for vehicles with only one session
            vehicle_energy_std = vehicle_energy_std.fillna(df['energy'].std())
            vehicle_energy_std = vehicle_energy_std.replace(0, df['energy'].std())
            
            df['z_score_energy'] = (df['energy'] - vehicle_energy_mean) / vehicle_energy_std
        else:
            # Global z-score if no vehicle information
            df['z_score_energy'] = (df['energy'] - df['energy'].mean()) / df['energy'].std()
        
        # Handle any NaN values
        df['z_score_energy'] = df['z_score_energy'].fillna(0)
        
        return df
    
    def _create_vehicle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create vehicle behavior features."""
        self.logger.info("Creating vehicle behavior features...")
        
        if 'userId' in df.columns:
            # Vehicle session count
            vehicle_counts = df['userId'].value_counts()
            df['vehicle_session_count'] = df['userId'].map(vehicle_counts)
            
            # Vehicle average energy (same as vehicle_mean_energy but for consistency)
            df['vehicle_avg_energy'] = df['vehicle_mean_energy']
            
            # Vehicle energy standard deviation
            vehicle_energy_std = df.groupby('userId')['energy'].transform('std')
            df['vehicle_energy_std'] = vehicle_energy_std.fillna(df['energy'].std())
            
            # Energy deviation from vehicle average
            df['energy_deviation'] = abs(df['energy'] - df['vehicle_avg_energy'])
            
            # Vehicle energy percentile (where this session ranks within vehicle history)
            df['vehicle_energy_percentile'] = df.groupby('userId')['energy'].rank(pct=True)
            
            # Normalized energy per vehicle (current / average)
            df['normalized_energy_per_vehicle'] = df['energy'] / (df['vehicle_avg_energy'] + 1e-6)
        else:
            # Default values if no vehicle information
            df['vehicle_session_count'] = 1
            df['vehicle_avg_energy'] = df['energy'].mean()
            df['vehicle_energy_std'] = df['energy'].std()
            df['energy_deviation'] = abs(df['energy'] - df['energy'].mean())
            df['vehicle_energy_percentile'] = 0.5
            df['normalized_energy_per_vehicle'] = 1.0
        
        return df
    
    def _create_station_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create station context features."""
        self.logger.info("Creating station context features...")
        
        if 'stationId' in df.columns:
            # Station average energy
            station_avg_energy = df.groupby('stationId')['energy'].transform('mean')
            df['station_avg_energy'] = station_avg_energy
            
            # Station energy variance
            station_energy_var = df.groupby('stationId')['energy'].transform('var')
            df['station_energy_variance'] = station_energy_var.fillna(0)
            
            # Deviation from station average
            df['deviation_from_station_avg'] = abs(df['energy'] - df['station_avg_energy'])
            
            # Station energy z-score
            station_energy_std = df.groupby('stationId')['energy'].transform('std')
            station_energy_std = station_energy_std.fillna(df['energy'].std())
            station_energy_std = station_energy_std.replace(0, df['energy'].std())
            
            df['station_energy_zscore'] = (df['energy'] - df['station_avg_energy']) / station_energy_std
        else:
            # Default values if no station information
            df['station_avg_energy'] = df['energy'].mean()
            df['station_energy_variance'] = df['energy'].var()
            df['deviation_from_station_avg'] = abs(df['energy'] - df['energy'].mean())
            df['station_energy_zscore'] = df['z_score_energy']
        
        return df
    
    def _create_billing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create billing and cost features."""
        self.logger.info("Creating billing and cost features...")
        
        if 'dollars' in df.columns:
            # Total cost
            df['dollars_total'] = df['dollars']
            
            # Cost per kWh (same as billing_per_kWh but for consistency)
            df['cost_per_kwh'] = df['billing_per_kWh']
            
            # Expected cost per kWh (could be based on station, time, etc.)
            # For now, use median rate as expected
            if 'stationId' in df.columns:
                station_median_rate = df.groupby('stationId')['billing_per_kWh'].transform('median')
                df['expected_cost_per_kwh'] = station_median_rate
            else:
                df['expected_cost_per_kwh'] = df['billing_per_kWh'].median()
            
            # Billing anomaly score (deviation from expected rate)
            df['billing_anomaly_score'] = abs(df['cost_per_kwh'] - df['expected_cost_per_kwh'])
            
            # Billing z-score
            billing_mean = df['billing_per_kWh'].mean()
            billing_std = df['billing_per_kWh'].std()
            df['billing_zscore'] = (df['billing_per_kWh'] - billing_mean) / (billing_std + 1e-6)
        else:
            # Default values if no billing information
            df['dollars_total'] = df['energy'] * 0.15  # Assume $0.15/kWh
            df['cost_per_kwh'] = 0.15
            df['expected_cost_per_kwh'] = 0.15
            df['billing_anomaly_score'] = 0.0
            df['billing_zscore'] = 0.0
        
        return df
    
    def _create_additional_energy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional energy-specific features."""
        self.logger.info("Creating additional energy features...")
        
        # Energy rate (kWh per hour)
        if 'chargeTimeHrs' in df.columns:
            df['energy_rate'] = df['energy'] / (df['chargeTimeHrs'] + 1e-6)
        elif 'chargingDuration' in df.columns:
            df['energy_rate'] = df['energy'] / (df['chargingDuration'] + 1e-6)
        else:
            df['energy_rate'] = df['energy']  # Default if no duration
        
        # Log-transformed energy (to handle skewness)
        df['log_energy'] = np.log1p(df['energy'])
        
        # Energy categories based on typical ranges
        df['energy_category'] = pd.cut(
            df['energy'], 
            bins=[0, 5, 12, 30, 45, 80, float('inf')], 
            labels=[1, 2, 3, 4, 5, 6]  # Phantom, Very Low, Low, Normal, High, Very High
        ).astype(float)
        
        # Energy efficiency score (relative to vehicle average)
        df['energy_efficiency_score'] = df['energy'] / (df['vehicle_avg_energy'] + 1e-6)
        
        # Absolute energy deviation from global mean
        global_energy_mean = df['energy'].mean()
        df['global_energy_deviation'] = abs(df['energy'] - global_energy_mean)
        
        # Energy range classification
        df['is_phantom_charge'] = (df['energy'] < 5.0).astype(int)  # <5 kWh
        df['is_low_energy'] = ((df['energy'] >= 5.0) & (df['energy'] < 12.0)).astype(int)
        df['is_normal_energy'] = ((df['energy'] >= 12.0) & (df['energy'] <= 45.0)).astype(int)
        df['is_high_energy'] = (df['energy'] > 45.0).astype(int)
        df['is_very_high_energy'] = (df['energy'] > 80.0).astype(int)  # >80 kWh
        
        return df
    
    def _select_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select only numeric features for machine learning."""
        # Get numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove ID columns and target-like columns
        exclude_columns = [
            'sessionId', 'userId', 'stationId', 'locationId', 'managerVehicle',
            'facilityType', 'reportedZip', 'startTime', 'endTime', 'created', 'ended'
        ]
        
        # Keep only relevant numeric features
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        # Handle any remaining NaN values
        features_df = df[feature_columns].fillna(0)
        
        # Remove any infinite values
        features_df = features_df.replace([np.inf, -np.inf], 0)
        
        self.logger.info(f"Selected {len(feature_columns)} numeric features")
        self.logger.debug(f"Feature columns: {feature_columns}")
        
        return features_df
    
    def get_core_features(self) -> List[str]:
        """Get the list of core features for anomaly detection."""
        return [
            'energy',
            'billing_per_kWh',
            'vehicle_mean_energy', 
            'z_score_energy'
        ]
    
    def get_feature_importance_description(self) -> Dict[str, str]:
        """Get descriptions of all features for interpretability."""
        descriptions = {
            # Core features
            'energy': 'Energy delivered in the charging session (kWh)',
            'billing_per_kWh': 'Derived feature: billing amount divided by energy',
            'vehicle_mean_energy': 'Historical average energy consumption per vehicle',
            'z_score_energy': 'Standardized energy score: (energy - vehicle_mean) / vehicle_std',
            
            # Vehicle features
            'vehicle_session_count': 'Total number of sessions for this vehicle',
            'vehicle_avg_energy': 'Vehicle average energy consumption',
            'vehicle_energy_std': 'Vehicle energy consumption standard deviation',
            'energy_deviation': 'Absolute deviation from vehicle average energy',
            'vehicle_energy_percentile': 'Session percentile within vehicle history',
            'normalized_energy_per_vehicle': 'Current energy / vehicle average energy',
            
            # Station features
            'station_avg_energy': 'Average energy consumption at this station',
            'station_energy_variance': 'Energy consumption variance at station',
            'deviation_from_station_avg': 'Absolute deviation from station average',
            'station_energy_zscore': 'Z-score relative to station energy distribution',
            
            # Billing features
            'dollars_total': 'Total cost of the charging session',
            'cost_per_kwh': 'Cost per kWh (dollars / energy)',
            'expected_cost_per_kwh': 'Expected cost per kWh for this station',
            'billing_anomaly_score': 'Deviation from expected billing rate',
            'billing_zscore': 'Z-score of billing rate',
            
            # Additional energy features
            'energy_rate': 'Energy delivery rate (kWh per hour)',
            'log_energy': 'Log-transformed energy consumption',
            'energy_category': 'Energy consumption category (1=Phantom to 6=Very High)',
            'energy_efficiency_score': 'Energy relative to vehicle average',
            'global_energy_deviation': 'Deviation from global energy average',
            'is_phantom_charge': 'Whether session is phantom charge (<5 kWh)',
            'is_low_energy': 'Whether session is low energy (5-12 kWh)',
            'is_normal_energy': 'Whether session is normal energy (12-45 kWh)',
            'is_high_energy': 'Whether session is high energy (>45 kWh)',
            'is_very_high_energy': 'Whether session is very high energy (>80 kWh)'
        }
        
        return descriptions
    
    def validate_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate the created features for quality checks."""
        validation_results = {
            'total_features': len(df.columns),
            'core_features_present': [],
            'missing_core_features': [],
            'feature_quality': {}
        }
        
        # Check core features
        core_features = self.get_core_features()
        for feature in core_features:
            if feature in df.columns:
                validation_results['core_features_present'].append(feature)
            else:
                validation_results['missing_core_features'].append(feature)
        
        # Feature quality checks
        for col in df.columns:
            quality_metrics = {
                'missing_rate': df[col].isnull().sum() / len(df),
                'zero_rate': (df[col] == 0).sum() / len(df),
                'infinite_rate': np.isinf(df[col]).sum() / len(df),
                'unique_values': df[col].nunique(),
                'data_type': str(df[col].dtype)
            }
            validation_results['feature_quality'][col] = quality_metrics
        
        return validation_results 