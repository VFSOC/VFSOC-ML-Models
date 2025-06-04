"""
Synthetic data generator for GPS jamming detection.

This module integrates with the VFSOC Geotab connector to generate realistic
synthetic GPS jamming data for training and testing machine learning models.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

# Add the VFSOC log generation path to system path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent / "VFSOC-log_generation"))

try:
    from log_simulator.connectors.geotab_connector import GeotabConnector
except ImportError:
    # Fallback if import fails
    GeotabConnector = None

logger = logging.getLogger(__name__)


@dataclass
class DataGenerationConfig:
    """Configuration for synthetic data generation."""
    total_logs: int = 3150
    jamming_ratio: float = 0.05  # 5% jamming events
    normal_gps_ratio: float = 0.60  # 60% normal GPS logs
    device_disconnect_ratio: float = 0.05
    ignition_failure_ratio: float = 0.05
    geofence_breach_ratio: float = 0.05
    miscellaneous_ratio: float = 0.20
    
    # Time span configuration
    simulation_hours: float = 10.5
    start_time: Optional[datetime] = None
    
    # Feature engineering parameters
    add_noise: bool = True
    noise_level: float = 0.1
    
    # Output configuration
    output_format: str = "dataframe"  # "dataframe", "csv", "parquet"
    include_labels: bool = True


class SyntheticDataGenerator:
    """
    Generates synthetic GPS jamming detection data using the Geotab connector.
    
    This class creates realistic vehicle telemetry data with embedded jamming
    patterns for training machine learning models.
    """
    
    def __init__(self, config: Optional[DataGenerationConfig] = None):
        """
        Initialize the synthetic data generator.
        
        Args:
            config: Configuration for data generation
        """
        self.config = config or DataGenerationConfig()
        self.geotab_connector = None
        self._setup_connector()
        
        # Generated data storage
        self.raw_logs = []
        self.processed_data = None
        self.labels = None
        
    def _setup_connector(self) -> None:
        """Setup the Geotab connector for data generation."""
        if GeotabConnector is None:
            logger.error("Geotab connector not available. Check VFSOC-log_generation path.")
            raise ImportError("Cannot import GeotabConnector")
        
        try:
            self.geotab_connector = GeotabConnector()
            logger.info("Geotab connector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Geotab connector: {e}")
            raise
    
    def generate_synthetic_data(self, save_path: Optional[str] = None) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate synthetic GPS jamming detection dataset.
        
        Args:
            save_path: Optional path to save the generated data
            
        Returns:
            Tuple of (features_df, labels_array)
        """
        logger.info(f"Generating {self.config.total_logs} synthetic logs...")
        
        # Generate raw logs using Geotab connector
        self._generate_raw_logs()
        
        # Convert to Vehilog format
        vehilogs = self._convert_to_vehilog_format()
        
        # Extract features and labels
        features_df, labels = self._extract_features_and_labels(vehilogs)
        
        # Add synthetic noise if configured
        if self.config.add_noise:
            features_df = self._add_noise(features_df)
        
        # Store processed data
        self.processed_data = features_df
        self.labels = labels
        
        logger.info(f"Generated dataset with {len(features_df)} samples and {features_df.shape[1]} features")
        logger.info(f"Class distribution - Normal: {np.sum(labels == 1)}, Jamming: {np.sum(labels == -1)}")
        
        # Save if requested
        if save_path:
            self._save_data(features_df, labels, save_path)
        
        return features_df, labels
    
    def _generate_raw_logs(self) -> None:
        """Generate raw logs using the Geotab connector."""
        logger.info("Generating raw logs using Geotab connector...")
        
        # Generate all logs at once
        self.raw_logs = self.geotab_connector.generate_all_logs()
        
        logger.info(f"Generated {len(self.raw_logs)} raw logs")
        
        # Log generation summary
        summary = self.geotab_connector.get_log_summary()
        logger.info(f"Log generation summary: {summary}")
    
    def _convert_to_vehilog_format(self) -> List[Dict[str, Any]]:
        """Convert raw logs to Vehilog format."""
        logger.info("Converting raw logs to Vehilog format...")
        
        vehilogs = []
        for raw_log in self.raw_logs:
            try:
                vehilog = self.geotab_connector.convert_to_vehilog(raw_log)
                vehilogs.append(vehilog)
            except Exception as e:
                logger.warning(f"Failed to convert log: {e}")
                continue
        
        logger.info(f"Converted {len(vehilogs)} logs to Vehilog format")
        return vehilogs
    
    def _extract_features_and_labels(self, vehilogs: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Extract features and labels from Vehilog data.
        
        Args:
            vehilogs: List of Vehilog dictionaries
            
        Returns:
            Tuple of (features_df, labels_array)
        """
        logger.info("Extracting features and labels...")
        
        features_list = []
        labels_list = []
        
        for vehilog in vehilogs:
            # Extract features
            features = self._extract_single_log_features(vehilog)
            
            # Determine label (1 = normal, -1 = jamming/anomaly)
            label = self._determine_label(vehilog)
            
            features_list.append(features)
            labels_list.append(label)
        
        # Convert to DataFrame and numpy array
        features_df = pd.DataFrame(features_list)
        labels_array = np.array(labels_list)
        
        # Handle missing values
        features_df = features_df.fillna(0)
        
        return features_df, labels_array
    
    def _extract_single_log_features(self, vehilog: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract features from a single Vehilog entry.
        
        Args:
            vehilog: Single Vehilog dictionary
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        additional_data = vehilog.get('additional_data', {})
        
        # Basic features
        features['timestamp_hour'] = vehilog['timestamp'].hour if vehilog.get('timestamp') else 12
        features['is_off_hours'] = 1.0 if features['timestamp_hour'] >= 22 or features['timestamp_hour'] <= 6 else 0.0
        features['is_weekend'] = 1.0 if (vehilog.get('timestamp') and vehilog['timestamp'].weekday() >= 5) else 0.0
        
        # GPS signal features
        features['gps_signal_strength'] = float(additional_data.get('signal_strength', 65.0))
        features['is_signal_loss'] = 1.0 if features['gps_signal_strength'] == 0.0 else 0.0
        features['speed'] = float(additional_data.get('speed', 0.0))
        features['bearing'] = float(additional_data.get('bearing', 0.0))
        
        # Event type features
        event_type = vehilog.get('event_type', '')
        features['is_gps_jamming_event'] = 1.0 if 'GpsJamming' in event_type else 0.0
        features['is_gps_event'] = 1.0 if 'GPS' in event_type else 0.0
        features['is_device_disconnect'] = 1.0 if event_type == 'device_disconnect' else 0.0
        features['is_ignition_failure'] = 1.0 if event_type == 'ignition_on' and additional_data.get('status') == 'failure' else 0.0
        features['is_geofence_breach'] = 1.0 if event_type == 'geofence_exit' else 0.0
        
        # Severity features
        severity = vehilog.get('severity', 'info')
        features['severity_warning'] = 1.0 if severity == 'warning' else 0.0
        features['severity_error'] = 1.0 if severity == 'error' else 0.0
        
        # Location features
        location = vehilog.get('location', 'unknown')
        features['has_location'] = 1.0 if location != 'unknown' else 0.0
        
        # Trip-based features
        features['trip_id_numeric'] = float(hash(additional_data.get('trip_id', 'unknown')) % 10000)
        features['jamming_event_number'] = float(additional_data.get('jamming_event_number', 0))
        
        # Driver context
        features['has_driver'] = 1.0 if vehilog.get('driver_id') is not None else 0.0
        
        # Vehicle context
        vehicle_id = vehilog.get('vehicle_id', 'unknown')
        features['vehicle_id_numeric'] = float(hash(vehicle_id) % 1000)
        
        # Additional signal processing features
        features['signal_strength_normalized'] = features['gps_signal_strength'] / 100.0
        features['speed_normalized'] = min(features['speed'] / 120.0, 1.0)  # Normalize to 120 km/h max
        features['bearing_sin'] = np.sin(np.radians(features['bearing']))
        features['bearing_cos'] = np.cos(np.radians(features['bearing']))
        
        # Complex features
        features['signal_quality_score'] = self._calculate_signal_quality_score(features)
        features['anomaly_risk_score'] = self._calculate_anomaly_risk_score(features)
        
        return features
    
    def _calculate_signal_quality_score(self, features: Dict[str, float]) -> float:
        """Calculate a composite signal quality score."""
        base_score = features['gps_signal_strength'] / 100.0
        
        # Penalize signal loss
        if features['is_signal_loss']:
            base_score = 0.0
        
        # Consider location availability
        if not features['has_location']:
            base_score *= 0.5
        
        return base_score
    
    def _calculate_anomaly_risk_score(self, features: Dict[str, float]) -> float:
        """Calculate a composite anomaly risk score."""
        risk_score = 0.0
        
        # GPS jamming indicators
        if features['is_gps_jamming_event']:
            risk_score += 0.4
        
        if features['is_signal_loss']:
            risk_score += 0.3
        
        # Device issues
        if features['is_device_disconnect']:
            risk_score += 0.3
        
        # Unauthorized activity
        if features['is_off_hours'] and not features['has_driver']:
            risk_score += 0.2
        
        # Security events
        if features['is_geofence_breach'] and features['is_off_hours']:
            risk_score += 0.4
        
        return min(risk_score, 1.0)
    
    def _determine_label(self, vehilog: Dict[str, Any]) -> int:
        """
        Determine the label for a Vehilog entry.
        
        Args:
            vehilog: Vehilog dictionary
            
        Returns:
            Label (1 = normal, -1 = jamming/anomaly)
        """
        event_type = vehilog.get('event_type', '')
        additional_data = vehilog.get('additional_data', {})
        
        # GPS jamming events
        if 'GpsJamming' in event_type:
            return -1
        
        # Device disconnect during operation
        if event_type == 'device_disconnect':
            ignition_status = additional_data.get('ignition_status')
            if ignition_status == 'on':
                return -1
        
        # Multiple ignition failures (unauthorized access)
        if event_type == 'ignition_on' and additional_data.get('status') == 'failure':
            if not vehilog.get('driver_id'):
                return -1
        
        # Off-hours geofence breach (potential theft)
        if event_type == 'geofence_exit':
            if additional_data.get('off_hours_flag') and not vehilog.get('driver_id'):
                return -1
        
        # Everything else is normal
        return 1
    
    def _add_noise(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add synthetic noise to features for robustness.
        
        Args:
            features_df: Features DataFrame
            
        Returns:
            DataFrame with added noise
        """
        logger.info(f"Adding noise with level {self.config.noise_level}")
        
        # Select numeric columns for noise addition
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col not in ['is_signal_loss', 'is_gps_jamming_event', 'is_device_disconnect',
                          'is_ignition_failure', 'is_geofence_breach', 'is_off_hours', 'is_weekend']:
                # Skip binary features
                noise = np.random.normal(0, self.config.noise_level * features_df[col].std(),
                                        size=len(features_df))
                features_df[col] = features_df[col] + noise
        
        return features_df
    
    def _save_data(self, features_df: pd.DataFrame, labels: np.ndarray, save_path: str) -> None:
        """
        Save the generated data to disk.
        
        Args:
            features_df: Features DataFrame
            labels: Labels array
            save_path: Path to save the data
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save features
        if self.config.output_format == "csv":
            features_df.to_csv(save_path / "features.csv", index=False)
            np.savetxt(save_path / "labels.csv", labels, delimiter=",", fmt="%d")
        elif self.config.output_format == "parquet":
            features_df.to_parquet(save_path / "features.parquet", index=False)
            pd.DataFrame({"labels": labels}).to_parquet(save_path / "labels.parquet", index=False)
        else:
            # Default: pickle format
            features_df.to_pickle(save_path / "features.pkl")
            np.save(save_path / "labels.npy", labels)
        
        # Save metadata
        metadata = {
            "generation_config": {
                "total_logs": self.config.total_logs,
                "jamming_ratio": self.config.jamming_ratio,
                "simulation_hours": self.config.simulation_hours,
                "noise_level": self.config.noise_level if self.config.add_noise else 0.0
            },
            "dataset_info": {
                "n_samples": len(features_df),
                "n_features": features_df.shape[1],
                "n_normal": int(np.sum(labels == 1)),
                "n_jamming": int(np.sum(labels == -1)),
                "feature_names": features_df.columns.tolist()
            },
            "generation_timestamp": datetime.now().isoformat()
        }
        
        with open(save_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Data saved to {save_path}")
    
    def generate_time_series_data(self, n_sequences: int = 100, 
                                 sequence_length: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate time series sequences for LSTM training.
        
        Args:
            n_sequences: Number of sequences to generate
            sequence_length: Length of each sequence
            
        Returns:
            Tuple of (sequences, labels)
        """
        logger.info(f"Generating {n_sequences} time series sequences of length {sequence_length}")
        
        if self.processed_data is None:
            # Generate data first
            self.generate_synthetic_data()
        
        features_array = self.processed_data.values
        
        sequences = []
        sequence_labels = []
        
        for _ in range(n_sequences):
            # Randomly sample a starting point
            start_idx = np.random.randint(0, len(features_array) - sequence_length)
            sequence = features_array[start_idx:start_idx + sequence_length]
            
            # Label is 1 if any point in sequence is jamming, 0 otherwise
            sequence_label = 1 if np.any(self.labels[start_idx:start_idx + sequence_length] == -1) else 0
            
            sequences.append(sequence)
            sequence_labels.append(sequence_label)
        
        return np.array(sequences), np.array(sequence_labels)
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all extracted features.
        
        Returns:
            Dictionary mapping feature names to descriptions
        """
        return {
            'timestamp_hour': 'Hour of the day (0-23)',
            'is_off_hours': 'Whether event occurred during off-hours (22:00-06:00)',
            'is_weekend': 'Whether event occurred on weekend',
            'gps_signal_strength': 'GPS signal strength (0-100)',
            'is_signal_loss': 'Whether GPS signal is lost (strength = 0)',
            'speed': 'Vehicle speed in km/h',
            'bearing': 'Vehicle bearing in degrees',
            'is_gps_jamming_event': 'Whether event is GPS jamming',
            'is_gps_event': 'Whether event is GPS-related',
            'is_device_disconnect': 'Whether device was disconnected',
            'is_ignition_failure': 'Whether ignition failed',
            'is_geofence_breach': 'Whether geofence was breached',
            'severity_warning': 'Whether event severity is warning',
            'severity_error': 'Whether event severity is error',
            'has_location': 'Whether location information is available',
            'trip_id_numeric': 'Numeric representation of trip ID',
            'jamming_event_number': 'Sequential number of jamming event in trip',
            'has_driver': 'Whether driver is authenticated',
            'vehicle_id_numeric': 'Numeric representation of vehicle ID',
            'signal_strength_normalized': 'Normalized GPS signal strength (0-1)',
            'speed_normalized': 'Normalized vehicle speed (0-1)',
            'bearing_sin': 'Sine of bearing angle',
            'bearing_cos': 'Cosine of bearing angle',
            'signal_quality_score': 'Composite signal quality score',
            'anomaly_risk_score': 'Composite anomaly risk score'
        }
    
    def get_generation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the data generation process.
        
        Returns:
            Summary dictionary
        """
        if self.processed_data is None:
            return {"status": "No data generated"}
        
        return {
            "total_samples": len(self.processed_data),
            "total_features": self.processed_data.shape[1],
            "normal_samples": int(np.sum(self.labels == 1)),
            "jamming_samples": int(np.sum(self.labels == -1)),
            "jamming_percentage": float(np.mean(self.labels == -1) * 100),
            "feature_names": self.processed_data.columns.tolist(),
            "config": self.config.__dict__
        } 