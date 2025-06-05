#!/usr/bin/env python3
"""
EV Charging Anomaly Detection - Simple Training Script
Fleet-focused irregular energy consumption detection
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vfsoc_ml.models.ev_charging_anomaly_detector import EVChargingAnomalyDetector


def main():
    """Train anomaly detection model for irregular energy consumption."""
    print("EV Charging Energy Consumption Anomaly Detection - Training")
    print("=" * 60)
    print("Fleet Management - Irregular Energy Consumption Detection")
    print("Model: IsolationForest (Unsupervised Anomaly Detection)")
    print("=" * 60)
    
    # Initialize detector
    detector = EVChargingAnomalyDetector()
    
    # Display capabilities
    print("\nDetection Capabilities:")
    print("* Low energy consumption (<5 kWh) - phantom charges, failures")
    print("* High energy consumption (>80 kWh) - meter tampering, over-delivery")
    print("* Billing irregularities - rate manipulation")
    print("* Vehicle-specific consumption anomalies")
    print("* Time-window based pattern analysis")
    
    # Generate training data
    print(f"\nGenerating fleet charging data...")
    print("Configuration:")
    print("  - 10 fleet vehicles (5 city, 5 long-range)")
    print("  - 5 charging stations")
    print("  - 30 days of charging data")
    print("  - ~3100 total charging sessions")
    print("  - 5-8% anomaly injection rate")
    
    features_df, labels = detector.generate_training_data(n_samples=3100)
    
    print(f"\nDataset Statistics:")
    print(f"  Total sessions: {len(features_df):,}")
    print(f"  Anomaly rate: {labels.mean():.1%} ({(labels == 1).sum():,} anomalies)")
    print(f"  Normal sessions: {(labels == 0).sum():,}")
    print(f"  Features: {len(features_df.columns)}")
    
    # Display feature information
    print(f"\nKey Features:")
    print(f"  - energy: Energy delivered (kWh)")
    print(f"  - billing_per_kwh: Cost per kWh ratio")
    print(f"  - vehicle_mean_energy: Historical average per vehicle")
    print(f"  - z_score_energy: Vehicle-specific energy deviation")
    print(f"  - energy_per_minute: Charging efficiency")
    print(f"  - power_efficiency: Power utilization ratio")
    print(f"  - Time-based features: hour, night/weekend indicators")
    
    # Train the model
    print(f"\nTraining IsolationForest Model...")
    print("Parameters:")
    print("  - contamination: 7% (expected anomaly rate)")
    print("  - n_estimators: 100")
    print("  - unsupervised learning approach")
    
    training_start = datetime.now()
    metrics = detector.train(features_df, labels)
    training_time = (datetime.now() - training_start).total_seconds()
    
    print(f"\nTraining completed in {training_time:.1f} seconds")
    print(f"\nModel Performance:")
    print(f"  Accuracy: {metrics['accuracy']:.1%}")
    print(f"  Precision: {metrics['precision']:.1%}")
    print(f"  Recall: {metrics['recall']:.1%}")
    print(f"  Total samples: {metrics['total_samples']:,}")
    print(f"  Detected anomalies: {metrics['detected_anomalies']:,}")
    print(f"  Actual anomalies: {metrics['actual_anomalies']:,}")
    
    # Test with sample scenarios
    print(f"\nTesting with sample charging scenarios...")
    
    test_scenarios = [
        {
            "name": "Normal Tesla Charging",
            "data": {
                "timestamp": "2024-01-15T14:30:00Z",
                "vehicle_id": "VH_001",
                "station_id": "CHG_01",
                "energy": 28.5,
                "billing": 7.12,
                "duration_minutes": 45,
                "power_kw": 50,
                "hour_of_day": 14,
                "day_of_week": 0
            }
        },
        {
            "name": "High Energy Consumption (Suspicious)",
            "data": {
                "timestamp": "2024-01-15T09:15:00Z",
                "vehicle_id": "VH_003",
                "station_id": "CHG_02",
                "energy": 95.0,  # Very high
                "billing": 23.75,
                "duration_minutes": 320,
                "power_kw": 25,
                "hour_of_day": 9,
                "day_of_week": 1
            }
        },
        {
            "name": "Low Energy Consumption (Phantom Charge)",
            "data": {
                "timestamp": "2024-01-15T20:45:00Z",
                "vehicle_id": "VH_005",
                "station_id": "CHG_03",
                "energy": 2.1,  # Very low
                "billing": 0.53,
                "duration_minutes": 8,
                "power_kw": 22,
                "hour_of_day": 20,
                "day_of_week": 2
            }
        },
        {
            "name": "Billing Irregularity",
            "data": {
                "timestamp": "2024-01-15T16:20:00Z",
                "vehicle_id": "VH_007",
                "station_id": "CHG_04",
                "energy": 35.0,
                "billing": 3.50,  # Too low ($0.10/kWh instead of ~$0.25)
                "duration_minutes": 75,
                "power_kw": 30,
                "hour_of_day": 16,
                "day_of_week": 3
            }
        }
    ]
    
    print(f"\nDetection Results:")
    print("=" * 50)
    
    for scenario in test_scenarios:
        session_json = json.dumps(scenario["data"])
        result = detector.analyze_charging_session(session_json)
        
        print(f"\n{scenario['name']}:")
        print(f"  Vehicle: {result['vehicle_id']} | Station: {result['station_id']}")
        print(f"  Energy: {result['energy']} kWh | Expected: {result['expected_range']}")
        print(f"  Alert Type: {result['alert_type']}")
        print(f"  Anomaly Score: {result['anomaly_score']:.3f}")
        print(f"  Severity: {result['severity']}")
        if result['is_anomaly']:
            print(f"  Anomaly Type: {result['anomaly_type']}")
            print(f"  Billing Rate: ${result['billing_per_kwh']:.3f}/kWh")
    
    # Test time window analysis
    print(f"\nTime Window Analysis (24-hour window):")
    print("-" * 40)
    
    # Create multiple sessions for time window test
    window_sessions = [json.dumps(scenario["data"]) for scenario in test_scenarios]
    window_analysis = detector.analyze_time_window(window_sessions, window_hours=24)
    
    print(f"Total sessions in window: {window_analysis['total_sessions']}")
    print(f"Anomalies detected: {window_analysis['anomalies_detected']}")
    print(f"Anomaly rate: {window_analysis['anomaly_rate']:.1%}")
    print(f"Average energy: {window_analysis['average_energy']} kWh")
    print(f"Energy std dev: {window_analysis['energy_std']} kWh")
    print(f"Max anomaly score: {window_analysis['max_anomaly_score']}")
    
    if window_analysis['anomaly_types']:
        print(f"Anomaly types detected:")
        for atype, count in window_analysis['anomaly_types'].items():
            print(f"  - {atype}: {count}")
    
    # Save the model
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'trained')
    model_path = os.path.join(model_dir, 'ev_charging_anomaly_detector.pkl')
    detector.save_model(model_path)
    
    # Save training summary
    summary = {
        "training_date": datetime.now().isoformat(),
        "model_type": "IsolationForest",
        "dataset_size": len(features_df),
        "anomaly_rate": float(labels.mean()),
        "features": list(features_df.columns),
        "performance": metrics,
        "normal_energy_range": detector.normal_energy_range,
        "anomaly_thresholds": detector.anomaly_thresholds
    }
    
    summary_path = os.path.join(model_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTraining completed successfully!")
    print(f"Model saved: {model_path}")
    print(f"Summary saved: {summary_path}")
    
    print(f"\nFleet Energy Consumption Monitoring Ready:")
    print(f"  * 10 vehicle fleet configuration")
    print(f"  * Normal range: {detector.normal_energy_range[0]}-{detector.normal_energy_range[1]} kWh")
    print(f"  * Anomaly thresholds: <{detector.anomaly_thresholds['low_energy']} kWh or >{detector.anomaly_thresholds['high_energy']} kWh")
    print(f"  * Time-window analysis capability")
    print(f"  * Unsupervised anomaly detection with {metrics['accuracy']:.1%} accuracy")
    
    print(f"\nReady for production deployment!")


if __name__ == "__main__":
    main() 