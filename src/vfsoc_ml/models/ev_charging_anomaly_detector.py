"""
EV Charging Energy Consumption Anomaly Detector
Simple but effective anomaly detection for fleet charging irregularities
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pickle
import os


class EVChargingAnomalyDetector:
    """
    Fleet-focused EV charging anomaly detector for irregular energy consumption.
    
    Detects:
    - Abnormally high energy consumption (>80 kWh)
    - Abnormally low energy consumption (<5 kWh)  
    - Irregular billing per kWh ratios
    - Vehicle-specific consumption anomalies
    """
    
    def __init__(self):
        self.model = IsolationForest(
            contamination=0.07,  # Expected 5-8% anomaly rate
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.vehicle_profiles = {}  # Historical energy patterns per vehicle
        self.is_trained = False
        
        # Fleet configuration
        self.normal_energy_range = (12, 45)  # Normal kWh range for fleet vehicles
        self.anomaly_thresholds = {
            'low_energy': 5.0,   # Below 5 kWh is suspicious
            'high_energy': 80.0,  # Above 80 kWh is suspicious
            'billing_ratio_min': 0.15,  # Min $/kWh
            'billing_ratio_max': 0.50   # Max $/kWh
        }
    
    def generate_training_data(self, n_samples: int = 3100) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate realistic fleet charging data with time windows."""
        np.random.seed(42)
        
        # Fleet configuration
        vehicles = [f"VH_{i:03d}" for i in range(1, 11)]  # 10 vehicles
        stations = [f"CHG_{i:02d}" for i in range(1, 6)]   # 5 stations
        
        # Vehicle types with different energy patterns
        vehicle_types = {
            f"VH_{i:03d}": {
                'type': 'city' if i <= 5 else 'long_range',
                'base_energy': 25 if i <= 5 else 35,
                'energy_std': 8 if i <= 5 else 12
            } for i in range(1, 11)
        }
        
        data = []
        labels = []
        
        # Generate sessions over 30 days
        start_date = datetime(2024, 1, 1)
        
        for day in range(30):
            current_date = start_date + timedelta(days=day)
            
            # Generate 100-105 sessions per day
            daily_sessions = np.random.randint(100, 106)
            
            for session in range(daily_sessions):
                # Random time during day (more sessions during business hours)
                hour = np.random.choice(range(24), p=self._get_hourly_distribution())
                minute = np.random.randint(0, 60)
                timestamp = current_date.replace(hour=hour, minute=minute)
                
                vehicle_id = np.random.choice(vehicles)
                station_id = np.random.choice(stations)
                
                # Get vehicle profile
                v_profile = vehicle_types[vehicle_id]
                
                # Normal energy consumption with time-based variation
                base_energy = v_profile['base_energy']
                energy_std = v_profile['energy_std']
                
                # Add time-based variation (higher consumption during cold hours)
                time_factor = 1.0
                if hour < 6 or hour > 20:  # Early morning/evening
                    time_factor = 1.15
                elif 10 <= hour <= 16:  # Midday
                    time_factor = 0.95
                
                energy = np.random.normal(base_energy * time_factor, energy_std)
                energy = max(8, energy)  # Minimum viable energy
                
                # Normal billing rate with small variation
                base_rate = 0.25  # $0.25 per kWh
                rate_variation = np.random.normal(0, 0.03)
                billing_rate = base_rate + rate_variation
                billing = energy * billing_rate
                
                # Session duration (realistic charging time)
                power_kw = np.random.uniform(22, 150)  # Charging power
                duration_minutes = max(10, (energy / power_kw) * 60 + np.random.normal(0, 10))
                
                is_anomaly = 0
                
                # Inject anomalies (5-8% of sessions)
                if np.random.random() < 0.07:
                    is_anomaly = 1
                    anomaly_type = np.random.choice(['low_energy', 'high_energy', 'billing_fraud'])
                    
                    if anomaly_type == 'low_energy':
                        # Phantom charges, connection failures
                        energy = np.random.uniform(0.5, 4.5)
                        duration_minutes = np.random.uniform(2, 15)
                        
                    elif anomaly_type == 'high_energy':
                        # Over-delivery, meter tampering
                        energy = np.random.uniform(80, 120)
                        duration_minutes = np.random.uniform(180, 400)
                        
                    elif anomaly_type == 'billing_fraud':
                        # Billing manipulation
                        billing = energy * np.random.uniform(0.05, 0.12)  # Very low rate
                
                data.append({
                    'timestamp': timestamp.isoformat(),
                    'vehicle_id': vehicle_id,
                    'station_id': station_id,
                    'energy': energy,
                    'billing': billing,
                    'duration_minutes': duration_minutes,
                    'power_kw': power_kw,
                    'hour_of_day': hour,
                    'day_of_week': timestamp.weekday()
                })
                labels.append(is_anomaly)
        
        df = pd.DataFrame(data)
        return self._extract_features(df), np.array(labels)
    
    def _get_hourly_distribution(self) -> np.ndarray:
        """Get realistic hourly charging distribution."""
        # Higher probability during business hours and evening commute
        probs = np.array([
            0.02, 0.01, 0.01, 0.01, 0.02, 0.03,  # 0-5 AM
            0.05, 0.08, 0.10, 0.12, 0.10, 0.08,  # 6-11 AM
            0.06, 0.08, 0.10, 0.12, 0.15, 0.18,  # 12-5 PM
            0.15, 0.12, 0.08, 0.05, 0.03, 0.02   # 6-11 PM
        ])
        return probs / probs.sum()
    
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for anomaly detection."""
        features = df.copy()
        
        # Calculate billing per kWh
        features['billing_per_kwh'] = features['billing'] / features['energy']
        features['billing_per_kwh'] = features['billing_per_kwh'].fillna(0)
        
        # Calculate vehicle historical averages (using all data for training)
        vehicle_stats = features.groupby('vehicle_id')['energy'].agg(['mean', 'std']).reset_index()
        vehicle_stats.columns = ['vehicle_id', 'vehicle_mean_energy', 'vehicle_std_energy']
        features = features.merge(vehicle_stats, on='vehicle_id', how='left')
        
        # Calculate z-score for each vehicle
        features['z_score_energy'] = (features['energy'] - features['vehicle_mean_energy']) / features['vehicle_std_energy']
        features['z_score_energy'] = features['z_score_energy'].fillna(0)
        
        # Energy efficiency (kWh per minute)
        features['energy_per_minute'] = features['energy'] / features['duration_minutes']
        
        # Power efficiency
        features['power_efficiency'] = features['energy'] / (features['power_kw'] * features['duration_minutes'] / 60)
        
        # Time-based features
        features['is_night'] = ((features['hour_of_day'] < 6) | (features['hour_of_day'] > 22)).astype(int)
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        
        # Select final features for model
        feature_columns = [
            'energy', 'billing_per_kwh', 'vehicle_mean_energy', 'z_score_energy',
            'energy_per_minute', 'power_efficiency', 'duration_minutes',
            'hour_of_day', 'is_night', 'is_weekend'
        ]
        
        return features[feature_columns].fillna(0)
    
    def train(self, features_df: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
        """Train the anomaly detection model."""
        # Scale features
        features_scaled = self.scaler.fit_transform(features_df)
        
        # Train IsolationForest on all data (unsupervised)
        self.model.fit(features_scaled)
        
        # Evaluate on the same data (for monitoring purposes)
        predictions = self.model.predict(features_scaled)
        anomaly_scores = self.model.decision_function(features_scaled)
        
        # Convert predictions (-1 for anomaly, 1 for normal) to (1 for anomaly, 0 for normal)
        predictions_binary = (predictions == -1).astype(int)
        
        # Calculate metrics
        accuracy = np.mean(predictions_binary == labels)
        precision = np.sum((predictions_binary == 1) & (labels == 1)) / max(1, np.sum(predictions_binary == 1))
        recall = np.sum((predictions_binary == 1) & (labels == 1)) / max(1, np.sum(labels == 1))
        
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'total_samples': len(features_df),
            'detected_anomalies': np.sum(predictions_binary),
            'actual_anomalies': np.sum(labels)
        }
    
    def predict(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies and return scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        features_scaled = self.scaler.transform(features_df)
        predictions = self.model.predict(features_scaled)
        scores = self.model.decision_function(features_scaled)
        
        # Convert to binary (1 for anomaly, 0 for normal)
        predictions_binary = (predictions == -1).astype(int)
        
        # Convert scores to [0,1] range (higher = more anomalous)
        normalized_scores = (0.5 - scores) / (0.5 - scores.min()) if scores.min() < 0.5 else np.ones_like(scores)
        normalized_scores = np.clip(normalized_scores, 0, 1)
        
        return predictions_binary, normalized_scores
    
    def analyze_charging_session(self, session_data: str) -> Dict[str, Any]:
        """Analyze a single charging session for irregularities."""
        try:
            data = json.loads(session_data)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format"}
        
        # Convert to DataFrame for feature extraction
        df = pd.DataFrame([data])
        
        # Extract features
        features_df = self._extract_features(df)
        
        # Predict
        predictions, scores = self.predict(features_df)
        
        is_anomaly = predictions[0] == 1
        anomaly_score = scores[0]
        
        # Determine anomaly type and severity
        energy = data.get('energy', 0)
        billing_per_kwh = data.get('billing', 0) / max(0.1, energy)
        
        anomaly_type = "normal"
        severity = "low"
        expected_range = f"{self.normal_energy_range[0]}-{self.normal_energy_range[1]} kWh"
        
        if is_anomaly:
            if energy < self.anomaly_thresholds['low_energy']:
                anomaly_type = "low_energy_consumption"
                severity = "high" if energy < 2 else "medium"
            elif energy > self.anomaly_thresholds['high_energy']:
                anomaly_type = "high_energy_consumption"  
                severity = "high" if energy > 100 else "medium"
            elif billing_per_kwh < self.anomaly_thresholds['billing_ratio_min']:
                anomaly_type = "billing_irregularity"
                severity = "high"
            else:
                anomaly_type = "irregular_pattern"
                severity = "medium"
        
        return {
            "alert_type": "IrregularEnergyConsumption" if is_anomaly else "Normal",
            "vehicle_id": data.get('vehicle_id', 'unknown'),
            "station_id": data.get('station_id', 'unknown'),
            "timestamp": data.get('timestamp', datetime.now().isoformat()),
            "energy": energy,
            "expected_range": expected_range,
            "anomaly_score": float(anomaly_score),
            "severity": severity,
            "anomaly_type": anomaly_type,
            "billing_per_kwh": round(billing_per_kwh, 3),
            "is_anomaly": is_anomaly
        }
    
    def analyze_time_window(self, sessions: List[str], window_hours: int = 24) -> Dict[str, Any]:
        """Analyze multiple sessions within a time window."""
        results = []
        anomaly_count = 0
        total_sessions = len(sessions)
        
        # Analyze each session
        for session in sessions:
            result = self.analyze_charging_session(session)
            results.append(result)
            if result.get('is_anomaly', False):
                anomaly_count += 1
        
        # Aggregate statistics
        energies = [r.get('energy', 0) for r in results]
        anomaly_scores = [r.get('anomaly_score', 0) for r in results]
        
        # Group by anomaly types
        anomaly_types = {}
        for result in results:
            if result.get('is_anomaly', False):
                atype = result.get('anomaly_type', 'unknown')
                anomaly_types[atype] = anomaly_types.get(atype, 0) + 1
        
        return {
            "analysis_timestamp": datetime.now().isoformat(),
            "time_window_hours": window_hours,
            "total_sessions": total_sessions,
            "anomalies_detected": anomaly_count,
            "anomaly_rate": round(anomaly_count / max(1, total_sessions), 3),
            "average_energy": round(np.mean(energies), 2),
            "energy_std": round(np.std(energies), 2),
            "max_anomaly_score": round(max(anomaly_scores), 3),
            "anomaly_types": anomaly_types,
            "session_results": results
        }
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'vehicle_profiles': self.vehicle_profiles,
            'is_trained': self.is_trained,
            'normal_energy_range': self.normal_energy_range,
            'anomaly_thresholds': self.anomaly_thresholds
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.vehicle_profiles = model_data['vehicle_profiles']
        self.is_trained = model_data['is_trained']
        self.normal_energy_range = model_data['normal_energy_range']
        self.anomaly_thresholds = model_data['anomaly_thresholds'] 