#!/usr/bin/env python3
"""
Synthetic Data Generation for Irregular Energy Consumption Detection

This script generates synthetic EV charging data according to the simulation strategy:
- Total Logs: ~3100 charging sessions
- 5-10 vehicles across 5 stations
- Normal sessions: 12-45 kWh with ±10% variability
- Anomaly injection: 5-8% rate with specific patterns
  - Low energy: <5 kWh (phantom charges, failures)
  - High energy: >80 kWh (over-delivery, false logs)
  - Billing anomalies: Distorted billing calculations
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple
import argparse

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from vfsoc_ml.utils.logger import setup_logger


class SyntheticEnergyDataGenerator:
    """
    Generator for synthetic EV charging session data following the simulation strategy.
    
    Generates realistic charging sessions with controlled anomaly injection
    for testing irregular energy consumption detection models.
    """
    
    def __init__(self, config_path: str):
        """Initialize the synthetic data generator."""
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = setup_logger("synthetic_data_generator")
        
        # Simulation parameters
        self.sim_config = self.config.get('simulation', {})
        self.vehicle_types = self.config.get('vehicle_types', {})
        
        # Random seed for reproducibility
        np.random.seed(42)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file."""
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def generate_synthetic_data(self, output_path: str = None) -> pd.DataFrame:
        """
        Generate synthetic charging session data according to simulation strategy.
        
        Args:
            output_path: Optional path to save the generated data
            
        Returns:
            DataFrame containing synthetic charging sessions
        """
        self.logger.info("Starting synthetic data generation...")
        
        # Parameters from simulation config
        total_logs = self.sim_config.get('total_logs', 3100)
        num_vehicles = self.sim_config.get('vehicles', {}).get('count', 8)
        num_stations = self.sim_config.get('stations', {}).get('count', 5)
        anomaly_rate = self.sim_config.get('anomalies', {}).get('injection_rate', 0.06)
        
        self.logger.info(f"Generating {total_logs} sessions for {num_vehicles} vehicles at {num_stations} stations")
        self.logger.info(f"Target anomaly rate: {anomaly_rate:.1%}")
        
        # Generate normal sessions
        normal_sessions = self._generate_normal_sessions(
            total_logs, num_vehicles, num_stations, anomaly_rate
        )
        
        # Inject anomalies
        all_sessions = self._inject_anomalies(normal_sessions, anomaly_rate)
        
        # Add metadata and timestamps
        final_data = self._add_metadata(all_sessions)
        
        # Validate generated data
        self._validate_data(final_data)
        
        # Save if output path provided
        if output_path:
            self._save_data(final_data, output_path)
        
        self.logger.info(f"Generated {len(final_data)} total sessions with {final_data['is_anomaly'].sum()} anomalies")
        
        return final_data
    
    def _generate_normal_sessions(self, total_logs: int, num_vehicles: int, 
                                 num_stations: int, anomaly_rate: float) -> pd.DataFrame:
        """Generate normal charging sessions."""
        self.logger.info("Generating normal charging sessions...")
        
        # Calculate number of normal sessions (excluding planned anomalies)
        num_normal = int(total_logs * (1 - anomaly_rate))
        
        sessions = []
        session_range = self.sim_config.get('vehicles', {}).get('sessions_per_vehicle', [300, 600])
        
        # Generate sessions for each vehicle
        for vehicle_id in range(1, num_vehicles + 1):
            # Determine vehicle type (mix of city and long-range EVs)
            if vehicle_id <= num_vehicles // 2:
                vehicle_type = 'city_ev'
            else:
                vehicle_type = 'long_range_ev'
            
            # Number of sessions for this vehicle
            vehicle_sessions = np.random.randint(session_range[0], session_range[1] + 1)
            vehicle_sessions = min(vehicle_sessions, num_normal // num_vehicles + 100)
            
            # Generate sessions for this vehicle
            vehicle_data = self._generate_vehicle_sessions(
                vehicle_id, vehicle_type, vehicle_sessions, num_stations
            )
            
            sessions.extend(vehicle_data)
        
        # Trim to exact number needed
        sessions = sessions[:num_normal]
        
        # Convert to DataFrame
        df = pd.DataFrame(sessions)
        df['is_anomaly'] = False
        
        self.logger.info(f"Generated {len(df)} normal sessions")
        
        return df
    
    def _generate_vehicle_sessions(self, vehicle_id: int, vehicle_type: str, 
                                  num_sessions: int, num_stations: int) -> List[Dict[str, Any]]:
        """Generate sessions for a specific vehicle."""
        sessions = []
        
        # Get vehicle type configuration
        vehicle_config = self.vehicle_types.get(vehicle_type, {})
        energy_range = vehicle_config.get('typical_range', [12, 45])
        variability = self.sim_config.get('vehicles', {}).get('vehicle_types', {}).get(vehicle_type, {}).get('variability', 0.10)
        
        # Vehicle's base energy consumption (consistent per vehicle)
        base_energy = np.random.uniform(energy_range[0], energy_range[1])
        
        for session_idx in range(num_sessions):
            # Add variability to energy consumption (±10%)
            energy_variation = np.random.normal(0, variability * base_energy)
            energy = max(base_energy + energy_variation, 1.0)  # Minimum 1 kWh
            energy = min(energy, energy_range[1] * 1.2)  # Cap at 120% of max
            
            # Charging duration (varies with energy)
            base_duration = energy / np.random.uniform(3, 15)  # 3-15 kW charging rate
            duration = max(base_duration, 0.25)  # Minimum 15 minutes
            
            # Station assignment (vehicles may prefer certain stations)
            preferred_stations = np.random.choice(range(1, num_stations + 1), 
                                                size=min(3, num_stations), replace=False)
            station_id = np.random.choice(preferred_stations)
            
            # Billing calculation
            base_rate = np.random.uniform(0.12, 0.18)  # $0.12-$0.18 per kWh
            billing = energy * base_rate
            
            session = {
                'sessionId': f"SESSION_{vehicle_id:03d}_{session_idx:04d}",
                'userId': f"VH_{vehicle_id:03d}",
                'stationId': f"CHG_{station_id:02d}",
                'kwhTotal': round(energy, 2),
                'chargeTimeHrs': round(duration, 2),
                'dollars': round(billing, 2),
                'vehicle_type': vehicle_type
            }
            
            sessions.append(session)
        
        return sessions
    
    def _inject_anomalies(self, normal_sessions: pd.DataFrame, anomaly_rate: float) -> pd.DataFrame:
        """Inject anomalies according to the specified patterns."""
        self.logger.info("Injecting anomalies...")
        
        # Calculate number of anomalies to inject
        num_anomalies = int(len(normal_sessions) * anomaly_rate / (1 - anomaly_rate))
        
        # Get anomaly type configuration
        anomaly_config = self.sim_config.get('anomalies', {}).get('types', {})
        
        # Generate anomalous sessions
        anomalous_sessions = []
        
        # Low energy anomalies (<5 kWh - phantom charges, failures)
        low_energy_count = int(num_anomalies * anomaly_config.get('low_energy', {}).get('proportion', 0.4))
        low_energy_sessions = self._generate_low_energy_anomalies(low_energy_count, normal_sessions)
        anomalous_sessions.extend(low_energy_sessions)
        
        # High energy anomalies (>80 kWh - over-delivery, false logs)
        high_energy_count = int(num_anomalies * anomaly_config.get('high_energy', {}).get('proportion', 0.4))
        high_energy_sessions = self._generate_high_energy_anomalies(high_energy_count, normal_sessions)
        anomalous_sessions.extend(high_energy_sessions)
        
        # Billing mismatch anomalies
        billing_count = num_anomalies - len(anomalous_sessions)
        billing_sessions = self._generate_billing_anomalies(billing_count, normal_sessions)
        anomalous_sessions.extend(billing_sessions)
        
        # Convert anomalies to DataFrame
        anomaly_df = pd.DataFrame(anomalous_sessions)
        anomaly_df['is_anomaly'] = True
        
        # Combine normal and anomalous sessions
        all_sessions = pd.concat([normal_sessions, anomaly_df], ignore_index=True)
        
        # Shuffle the data
        all_sessions = all_sessions.sample(frac=1, random_state=42).reset_index(drop=True)
        
        self.logger.info(f"Injected {len(anomaly_df)} anomalies ({len(anomaly_df)/len(all_sessions):.1%} rate)")
        
        return all_sessions
    
    def _generate_low_energy_anomalies(self, count: int, normal_sessions: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate low energy anomalies (<5 kWh)."""
        anomalies = []
        
        for i in range(count):
            # Base session from normal data
            base_session = normal_sessions.sample(1).iloc[0].to_dict()
            
            # Modify to be low energy anomaly
            base_session['sessionId'] = f"ANOM_LOW_{i:04d}"
            base_session['kwhTotal'] = round(np.random.uniform(0.1, 4.9), 2)  # <5 kWh
            base_session['chargeTimeHrs'] = round(np.random.uniform(0.1, 2.0), 2)  # Short duration
            
            # Billing might still be normal rate (phantom charges) or very low
            if np.random.random() < 0.5:
                # Phantom charge - normal billing despite low energy
                base_session['dollars'] = round(base_session['kwhTotal'] * np.random.uniform(0.12, 0.18), 2)
            else:
                # Equipment failure - very low or zero billing
                base_session['dollars'] = round(np.random.uniform(0, 1.0), 2)
            
            anomalies.append(base_session)
        
        return anomalies
    
    def _generate_high_energy_anomalies(self, count: int, normal_sessions: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate high energy anomalies (>80 kWh)."""
        anomalies = []
        
        for i in range(count):
            # Base session from normal data
            base_session = normal_sessions.sample(1).iloc[0].to_dict()
            
            # Modify to be high energy anomaly
            base_session['sessionId'] = f"ANOM_HIGH_{i:04d}"
            base_session['kwhTotal'] = round(np.random.uniform(80.1, 150.0), 2)  # >80 kWh
            base_session['chargeTimeHrs'] = round(np.random.uniform(4.0, 12.0), 2)  # Long duration
            
            # Billing anomalies
            if np.random.random() < 0.3:
                # Over-billing
                base_session['dollars'] = round(base_session['kwhTotal'] * np.random.uniform(0.20, 0.50), 2)
            else:
                # Normal billing despite high energy
                base_session['dollars'] = round(base_session['kwhTotal'] * np.random.uniform(0.12, 0.18), 2)
            
            anomalies.append(base_session)
        
        return anomalies
    
    def _generate_billing_anomalies(self, count: int, normal_sessions: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate billing mismatch anomalies."""
        anomalies = []
        
        for i in range(count):
            # Base session from normal data
            base_session = normal_sessions.sample(1).iloc[0].to_dict()
            
            # Keep energy normal but distort billing
            base_session['sessionId'] = f"ANOM_BILL_{i:04d}"
            
            # Billing anomalies
            if np.random.random() < 0.5:
                # Extremely high rate (>$0.50/kWh)
                rate = np.random.uniform(0.50, 2.00)
                base_session['dollars'] = round(base_session['kwhTotal'] * rate, 2)
            else:
                # Extremely low rate (<$0.05/kWh) or free
                rate = np.random.uniform(0.00, 0.05)
                base_session['dollars'] = round(base_session['kwhTotal'] * rate, 2)
            
            anomalies.append(base_session)
        
        return anomalies
    
    def _add_metadata(self, sessions_df: pd.DataFrame) -> pd.DataFrame:
        """Add timestamps and other metadata to sessions."""
        self.logger.info("Adding metadata and timestamps...")
        
        # Generate timestamps over the past 6 months
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        # Add timestamps
        timestamps = []
        for _ in range(len(sessions_df)):
            # Random timestamp in the range
            random_timestamp = start_date + timedelta(
                seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))
            )
            timestamps.append(random_timestamp)
        
        sessions_df['startTime'] = timestamps
        
        # Calculate end times
        sessions_df['endTime'] = sessions_df.apply(
            lambda row: row['startTime'] + timedelta(hours=row['chargeTimeHrs']), 
            axis=1
        )
        
        # Add location and other metadata
        sessions_df['locationId'] = sessions_df['stationId'].map(
            lambda x: f"LOC_{int(x.split('_')[1]):02d}"
        )
        
        # Add weekday information
        sessions_df['weekday'] = sessions_df['startTime'].dt.dayofweek
        sessions_df['is_weekend'] = (sessions_df['weekday'] >= 5).astype(int)
        
        # Add some random noise to make it more realistic
        sessions_df['facilityType'] = np.random.choice(['public', 'workplace', 'residential'], 
                                                      size=len(sessions_df), 
                                                      p=[0.6, 0.3, 0.1])
        
        return sessions_df
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate the generated synthetic data."""
        self.logger.info("Validating generated data...")
        
        # Check data quality
        anomaly_rate = data['is_anomaly'].mean()
        energy_range = (data['kwhTotal'].min(), data['kwhTotal'].max())
        
        # Validation checks
        checks = {
            'Total sessions': len(data),
            'Anomaly rate': f"{anomaly_rate:.1%}",
            'Energy range': f"{energy_range[0]:.1f} - {energy_range[1]:.1f} kWh",
            'Unique vehicles': data['userId'].nunique(),
            'Unique stations': data['stationId'].nunique(),
            'Low energy anomalies': len(data[(data['is_anomaly']) & (data['kwhTotal'] < 5)]),
            'High energy anomalies': len(data[(data['is_anomaly']) & (data['kwhTotal'] > 80)]),
            'Billing anomalies': len(data[data['is_anomaly']]) - 
                               len(data[(data['is_anomaly']) & ((data['kwhTotal'] < 5) | (data['kwhTotal'] > 80))])
        }
        
        self.logger.info("Data validation results:")
        for check, value in checks.items():
            self.logger.info(f"  {check}: {value}")
        
        # Quality checks
        assert 0.05 <= anomaly_rate <= 0.08, f"Anomaly rate {anomaly_rate:.1%} outside target range (5-8%)"
        assert data['userId'].nunique() >= 5, "Should have at least 5 vehicles"
        assert data['stationId'].nunique() == 5, "Should have exactly 5 stations"
        
        self.logger.info("Data validation passed!")
    
    def _save_data(self, data: pd.DataFrame, output_path: str) -> None:
        """Save the generated data to file."""
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save data
        data.to_csv(output_path, index=False)
        self.logger.info(f"Synthetic data saved to: {output_path}")
        
        # Save summary statistics
        summary_path = output_path.replace('.csv', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Synthetic EV Charging Data Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total sessions: {len(data)}\n")
            f.write(f"Anomaly rate: {data['is_anomaly'].mean():.1%}\n")
            f.write(f"Energy range: {data['kwhTotal'].min():.1f} - {data['kwhTotal'].max():.1f} kWh\n")
            f.write(f"Unique vehicles: {data['userId'].nunique()}\n")
            f.write(f"Unique stations: {data['stationId'].nunique()}\n")
            f.write(f"Date range: {data['startTime'].min()} to {data['startTime'].max()}\n")
            f.write("\nAnomaly breakdown:\n")
            f.write(f"  Low energy (<5 kWh): {len(data[(data['is_anomaly']) & (data['kwhTotal'] < 5)])}\n")
            f.write(f"  High energy (>80 kWh): {len(data[(data['is_anomaly']) & (data['kwhTotal'] > 80)])}\n")
            f.write(f"  Billing anomalies: {len(data[data['is_anomaly']]) - len(data[(data['is_anomaly']) & ((data['kwhTotal'] < 5) | (data['kwhTotal'] > 80))])}\n")


def main():
    """Main function for synthetic data generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic EV charging data")
    parser.add_argument('--config', 
                       default='config/energy_consumption_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', 
                       default='data/raw/synthetic_energy_data.csv',
                       help='Output path for synthetic data')
    
    args = parser.parse_args()
    
    # Configuration path
    config_path = Path(__file__).parent.parent / args.config
    
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Output path
    output_path = Path(__file__).parent.parent / args.output
    
    try:
        # Initialize generator
        generator = SyntheticEnergyDataGenerator(str(config_path))
        
        # Generate data
        synthetic_data = generator.generate_synthetic_data(str(output_path))
        
        print("\n" + "="*50)
        print("SYNTHETIC DATA GENERATION COMPLETED")
        print("="*50)
        print(f"Generated {len(synthetic_data)} charging sessions")
        print(f"Anomaly rate: {synthetic_data['is_anomaly'].mean():.1%}")
        print(f"Output saved to: {output_path.absolute()}")
        print(f"Summary saved to: {str(output_path).replace('.csv', '_summary.txt')}")
        
    except Exception as e:
        print(f"Data generation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 