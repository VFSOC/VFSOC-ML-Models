"""
Enhanced Feature Extractor for GPS Jamming Detection.

This module provides advanced feature engineering techniques specifically designed
for detecting GPS jamming attacks in vehicle telemetry data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy import signal, stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)


class EnhancedFeatureExtractor:
    """
    Enhanced feature extractor for GPS jamming detection.
    
    This class implements sophisticated signal processing and statistical
    feature extraction techniques specifically designed for GPS jamming detection.
    """
    
    def __init__(self, 
                 window_size: int = 10,
                 statistical_features: bool = True,
                 signal_processing_features: bool = True,
                 temporal_features: bool = True,
                 contextual_features: bool = True,
                 dimensionality_reduction: bool = False,
                 pca_components: int = 15):
        """
        Initialize the enhanced feature extractor.
        
        Args:
            window_size: Window size for temporal features
            statistical_features: Whether to extract statistical features
            signal_processing_features: Whether to extract signal processing features
            temporal_features: Whether to extract temporal pattern features
            contextual_features: Whether to extract contextual features
            dimensionality_reduction: Whether to apply PCA
            pca_components: Number of PCA components
        """
        self.window_size = window_size
        self.statistical_features = statistical_features
        self.signal_processing_features = signal_processing_features
        self.temporal_features = temporal_features
        self.contextual_features = contextual_features
        self.dimensionality_reduction = dimensionality_reduction
        self.pca_components = pca_components
        
        # Fitted transformers
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=pca_components) if dimensionality_reduction else None
        self.is_fitted = False
        
        # Feature names for interpretability
        self.feature_names = []
        
    def fit_transform(self, vehilogs: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Fit the feature extractor and transform the data.
        
        Args:
            vehilogs: List of vehicle log dictionaries
            
        Returns:
            Tuple of (features_dataframe, feature_names)
        """
        # Extract raw features
        features_df = self._extract_all_features(vehilogs)
        
        # Fit and transform
        features_scaled = self.scaler.fit_transform(features_df)
        features_df = pd.DataFrame(features_scaled, columns=features_df.columns)
        
        # Apply PCA if enabled
        if self.dimensionality_reduction and self.pca is not None:
            features_pca = self.pca.fit_transform(features_df)
            pca_columns = [f'PCA_{i+1}' for i in range(self.pca_components)]
            features_df = pd.DataFrame(features_pca, columns=pca_columns)
        
        self.feature_names = features_df.columns.tolist()
        self.is_fitted = True
        
        logger.info(f"Extracted {features_df.shape[1]} features from {len(vehilogs)} vehicle logs")
        
        return features_df, self.feature_names
    
    def transform(self, vehilogs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Transform new data using fitted extractors.
        
        Args:
            vehilogs: List of vehicle log dictionaries
            
        Returns:
            Features dataframe
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before transform")
        
        # Extract raw features
        features_df = self._extract_all_features(vehilogs)
        
        # Transform using fitted scaler
        features_scaled = self.scaler.transform(features_df)
        features_df = pd.DataFrame(features_scaled, columns=features_df.columns)
        
        # Apply PCA if enabled
        if self.dimensionality_reduction and self.pca is not None:
            features_pca = self.pca.transform(features_df)
            pca_columns = [f'PCA_{i+1}' for i in range(self.pca_components)]
            features_df = pd.DataFrame(features_pca, columns=pca_columns)
        
        return features_df
    
    def _extract_all_features(self, vehilogs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract all configured features from vehicle logs."""
        all_features = []
        
        for vehilog in vehilogs:
            features = {}
            
            # Basic features from existing system
            basic_features = self._extract_basic_features(vehilog)
            features.update(basic_features)
            
            # Statistical features
            if self.statistical_features:
                stat_features = self._extract_statistical_features(vehilog)
                features.update(stat_features)
            
            # Signal processing features
            if self.signal_processing_features:
                signal_features = self._extract_signal_processing_features(vehilog)
                features.update(signal_features)
            
            # Temporal features
            if self.temporal_features:
                temporal_features = self._extract_temporal_features(vehilog)
                features.update(temporal_features)
            
            # Contextual features
            if self.contextual_features:
                context_features = self._extract_contextual_features(vehilog)
                features.update(context_features)
            
            all_features.append(features)
        
        # Convert to DataFrame and handle missing values
        features_df = pd.DataFrame(all_features)
        features_df = features_df.fillna(0)
        
        return features_df
    
    def _extract_basic_features(self, vehilog: Dict[str, Any]) -> Dict[str, float]:
        """Extract basic GPS and vehicle features."""
        features = {}
        
        # GPS signal features
        features['latitude'] = float(vehilog.get('latitude', 0))
        features['longitude'] = float(vehilog.get('longitude', 0))
        features['speed'] = float(vehilog.get('speed', 0))
        features['bearing'] = float(vehilog.get('bearing', 0))
        features['altitude'] = float(vehilog.get('altitude', 0))
        
        # Device and connectivity features
        features['device_connection_status'] = 1 if vehilog.get('device_connection_status') == 'connected' else 0
        features['ignition_status'] = 1 if vehilog.get('ignition_status') == 'on' else 0
        features['gps_fix_quality'] = float(vehilog.get('gps_fix_quality', 0))
        features['satellite_count'] = float(vehilog.get('satellite_count', 0))
        
        # Engine and vehicle state
        features['engine_hours'] = float(vehilog.get('engine_hours', 0))
        features['odometer'] = float(vehilog.get('odometer', 0))
        features['fuel_level'] = float(vehilog.get('fuel_level', 0))
        
        return features
    
    def _extract_statistical_features(self, vehilog: Dict[str, Any]) -> Dict[str, float]:
        """Extract advanced statistical features for anomaly detection."""
        features = {}
        
        # Get GPS-related values for statistical analysis
        gps_values = [
            vehilog.get('latitude', 0),
            vehilog.get('longitude', 0),
            vehilog.get('speed', 0),
            vehilog.get('bearing', 0),
            vehilog.get('altitude', 0),
            vehilog.get('gps_fix_quality', 0),
            vehilog.get('satellite_count', 0)
        ]
        
        gps_array = np.array(gps_values)
        
        # Statistical moments
        features['gps_mean'] = np.mean(gps_array)
        features['gps_std'] = np.std(gps_array)
        features['gps_var'] = np.var(gps_array)
        features['gps_skewness'] = stats.skew(gps_array) if len(gps_array) > 2 else 0
        features['gps_kurtosis'] = stats.kurtosis(gps_array) if len(gps_array) > 2 else 0
        
        # Percentiles
        features['gps_q25'] = np.percentile(gps_array, 25)
        features['gps_q50'] = np.percentile(gps_array, 50)
        features['gps_q75'] = np.percentile(gps_array, 75)
        features['gps_iqr'] = features['gps_q75'] - features['gps_q25']
        
        # Signal quality variations
        satellite_count = float(vehilog.get('satellite_count', 0))
        gps_quality = float(vehilog.get('gps_fix_quality', 0))
        
        # Quality indicators
        features['signal_strength_score'] = satellite_count * gps_quality if gps_quality > 0 else 0
        features['signal_consistency'] = 1 if satellite_count >= 4 and gps_quality > 0.5 else 0
        features['potential_jamming_indicator'] = 1 if satellite_count < 3 or gps_quality < 0.3 else 0
        
        return features
    
    def _extract_signal_processing_features(self, vehilog: Dict[str, Any]) -> Dict[str, float]:
        """Extract signal processing features for GPS jamming detection."""
        features = {}
        
        # Create a synthetic signal from GPS parameters for analysis
        gps_signal = np.array([
            vehilog.get('latitude', 0) * 1000,  # Scale for analysis
            vehilog.get('longitude', 0) * 1000,
            vehilog.get('speed', 0),
            vehilog.get('bearing', 0),
            vehilog.get('altitude', 0),
            vehilog.get('gps_fix_quality', 0) * 100,
            vehilog.get('satellite_count', 0) * 10
        ])
        
        # Signal energy and power
        features['signal_energy'] = np.sum(gps_signal ** 2)
        features['signal_power'] = np.mean(gps_signal ** 2)
        features['signal_rms'] = np.sqrt(np.mean(gps_signal ** 2))
        
        # Frequency domain features (simplified)
        if len(gps_signal) > 1:
            fft_signal = np.fft.fft(gps_signal)
            features['spectral_centroid'] = np.sum(np.abs(fft_signal)) / len(fft_signal)
            features['spectral_rolloff'] = np.percentile(np.abs(fft_signal), 85)
            features['spectral_flatness'] = stats.gmean(np.abs(fft_signal) + 1e-10) / np.mean(np.abs(fft_signal) + 1e-10)
        else:
            features['spectral_centroid'] = 0
            features['spectral_rolloff'] = 0
            features['spectral_flatness'] = 0
        
        # Zero crossing rate (signal stability)
        zero_crossings = np.sum(np.diff(np.sign(gps_signal)) != 0)
        features['zero_crossing_rate'] = zero_crossings / len(gps_signal) if len(gps_signal) > 1 else 0
        
        # Signal to noise ratio estimation
        signal_mean = np.mean(gps_signal)
        signal_std = np.std(gps_signal)
        features['estimated_snr'] = signal_mean / (signal_std + 1e-10)
        
        return features
    
    def _extract_temporal_features(self, vehilog: Dict[str, Any]) -> Dict[str, float]:
        """Extract temporal pattern features."""
        features = {}
        
        # Time-based features
        timestamp = vehilog.get('timestamp', '')
        if timestamp:
            try:
                dt = pd.to_datetime(timestamp)
                features['hour_of_day'] = dt.hour
                features['day_of_week'] = dt.dayofweek
                features['is_weekend'] = 1 if dt.dayofweek >= 5 else 0
                features['is_business_hours'] = 1 if 8 <= dt.hour <= 17 else 0
                features['is_night_time'] = 1 if dt.hour < 6 or dt.hour > 22 else 0
            except:
                features['hour_of_day'] = 0
                features['day_of_week'] = 0
                features['is_weekend'] = 0
                features['is_business_hours'] = 0
                features['is_night_time'] = 0
        else:
            features['hour_of_day'] = 0
            features['day_of_week'] = 0
            features['is_weekend'] = 0
            features['is_business_hours'] = 0
            features['is_night_time'] = 0
        
        # Movement pattern features
        speed = float(vehilog.get('speed', 0))
        features['is_stationary'] = 1 if speed < 1.0 else 0
        features['is_high_speed'] = 1 if speed > 80.0 else 0
        features['speed_category'] = 0 if speed < 10 else (1 if speed < 50 else 2)
        
        return features
    
    def _extract_contextual_features(self, vehilog: Dict[str, Any]) -> Dict[str, float]:
        """Extract contextual features for enhanced detection."""
        features = {}
        
        # Location context
        latitude = float(vehilog.get('latitude', 0))
        longitude = float(vehilog.get('longitude', 0))
        
        # Geographic context (simplified - could be enhanced with real geographic data)
        features['lat_zone'] = int(abs(latitude) // 10)  # Rough latitude zone
        features['lon_zone'] = int(abs(longitude) // 10)  # Rough longitude zone
        features['is_urban_area'] = 1 if abs(latitude) > 40 and abs(longitude) > 70 else 0  # Simplified
        
        # Vehicle operational context
        ignition_on = 1 if vehilog.get('ignition_status') == 'on' else 0
        device_connected = 1 if vehilog.get('device_connection_status') == 'connected' else 0
        
        features['operational_state'] = ignition_on * device_connected
        features['anomalous_state'] = 1 if ignition_on != device_connected else 0
        
        # Driver context
        features['driver_authenticated'] = 1 if vehilog.get('driver_authenticated', False) else 0
        features['driver_present'] = 1 if vehilog.get('driver_present', False) else 0
        features['security_alert'] = 1 if vehilog.get('security_alert', False) else 0
        
        # Multi-feature interactions
        satellite_count = float(vehilog.get('satellite_count', 0))
        gps_quality = float(vehilog.get('gps_fix_quality', 0))
        
        features['gps_satellite_quality_interaction'] = satellite_count * gps_quality
        features['speed_gps_quality_ratio'] = speed / (gps_quality + 1e-10) if 'speed' in locals() else 0
        
        # Risk assessment features
        features['high_risk_scenario'] = 1 if (
            satellite_count < 4 and 
            gps_quality < 0.5 and 
            not vehilog.get('driver_authenticated', False)
        ) else 0
        
        return features
    
    def get_feature_importance_analysis(self, features_df: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
        """
        Analyze feature importance for jamming detection.
        
        Args:
            features_df: Features dataframe
            labels: Target labels
            
        Returns:
            Dictionary of feature importance scores
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import mutual_info_classif
        
        # Random Forest feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(features_df, labels)
        rf_importance = dict(zip(features_df.columns, rf.feature_importances_))
        
        # Mutual information
        mi_scores = mutual_info_classif(features_df, labels, random_state=42)
        mi_importance = dict(zip(features_df.columns, mi_scores))
        
        # Combine scores
        combined_importance = {}
        for feature in features_df.columns:
            combined_importance[feature] = (rf_importance[feature] + mi_importance[feature]) / 2
        
        # Sort by importance
        return dict(sorted(combined_importance.items(), key=lambda x: x[1], reverse=True))
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all extracted features."""
        return {
            # Basic GPS features
            'latitude': 'Vehicle latitude coordinate',
            'longitude': 'Vehicle longitude coordinate',
            'speed': 'Vehicle speed in km/h',
            'bearing': 'Vehicle heading/bearing in degrees',
            'altitude': 'Vehicle altitude in meters',
            'gps_fix_quality': 'GPS fix quality indicator (0-1)',
            'satellite_count': 'Number of GPS satellites in view',
            
            # Statistical features
            'gps_mean': 'Mean of GPS parameter values',
            'gps_std': 'Standard deviation of GPS parameters',
            'gps_var': 'Variance of GPS parameters',
            'gps_skewness': 'Skewness of GPS parameter distribution',
            'gps_kurtosis': 'Kurtosis of GPS parameter distribution',
            'signal_strength_score': 'Combined GPS signal strength score',
            'signal_consistency': 'GPS signal consistency indicator',
            'potential_jamming_indicator': 'Basic jamming detection indicator',
            
            # Signal processing features
            'signal_energy': 'Total energy of GPS signal',
            'signal_power': 'Average power of GPS signal',
            'signal_rms': 'RMS value of GPS signal',
            'spectral_centroid': 'Spectral centroid of GPS signal',
            'spectral_rolloff': 'Spectral rolloff of GPS signal',
            'zero_crossing_rate': 'Zero crossing rate (signal stability)',
            'estimated_snr': 'Estimated signal-to-noise ratio',
            
            # Temporal features
            'hour_of_day': 'Hour of day (0-23)',
            'day_of_week': 'Day of week (0-6)',
            'is_weekend': 'Weekend indicator',
            'is_business_hours': 'Business hours indicator',
            'is_night_time': 'Night time indicator',
            'is_stationary': 'Vehicle stationary indicator',
            'is_high_speed': 'High speed indicator',
            
            # Contextual features
            'operational_state': 'Combined operational state',
            'anomalous_state': 'Anomalous operational state indicator',
            'high_risk_scenario': 'High risk jamming scenario indicator',
            'gps_satellite_quality_interaction': 'GPS satellite-quality interaction term'
        } 