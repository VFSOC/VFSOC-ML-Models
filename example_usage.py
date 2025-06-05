#!/usr/bin/env python3
"""
EV Charging Anomaly Detection - Example Usage
Fleet energy consumption monitoring with time window analysis
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from vfsoc_ml.models.ev_charging_anomaly_detector import EVChargingAnomalyDetector


def print_banner():
    """Print system banner."""
    print("EV Charging Energy Consumption Anomaly Detection")
    print("=" * 60)
    print("Fleet Management - Irregular Energy Consumption Monitoring")
    print("Focus: Energy theft, meter tampering, billing irregularities")
    print("=" * 60)


def print_session_result(name: str, result: dict, session_data: dict):
    """Print individual session analysis results."""
    print(f"\n{name}")
    print("-" * 40)
    
    # Basic session info
    vehicle = session_data.get('vehicle_id', 'Unknown')
    station = session_data.get('station_id', 'Unknown')
    energy = session_data.get('energy', 0)
    billing = session_data.get('billing', 0)
    
    print(f"Vehicle: {vehicle} | Station: {station}")
    print(f"Energy: {energy} kWh | Cost: ${billing:.2f}")
    
    # Detection results
    print(f"Alert Type: {result['alert_type']}")
    print(f"Anomaly Score: {result['anomaly_score']:.3f}")
    print(f"Expected Range: {result['expected_range']}")
    
    if result['is_anomaly']:
        print(f"ANOMALY DETECTED")
        print(f"Type: {result['anomaly_type']}")
        print(f"Severity: {result['severity']}")
        print(f"Billing Rate: ${result['billing_per_kwh']:.3f}/kWh")
    else:
        print(f"Status: Normal Operation")


def main():
    """Demonstrate energy consumption anomaly detection."""
    print_banner()
    
    # Load the model
    model_path = os.path.join("models", "trained", "ev_charging_anomaly_detector.pkl")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please run the training script first:")
        print("python scripts/train_ev_charging_anomaly_model_simple.py")
        return
    
    print("Loading energy consumption anomaly detection model...")
    detector = EVChargingAnomalyDetector()
    
    try:
        detector.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print(f"\nModel Configuration:")
    print(f"  Normal energy range: {detector.normal_energy_range[0]}-{detector.normal_energy_range[1]} kWh")
    print(f"  Low energy threshold: <{detector.anomaly_thresholds['low_energy']} kWh")
    print(f"  High energy threshold: >{detector.anomaly_thresholds['high_energy']} kWh")
    print(f"  Billing rate range: ${detector.anomaly_thresholds['billing_ratio_min']}-${detector.anomaly_thresholds['billing_ratio_max']}/kWh")
    
    # Test scenarios for fleet monitoring
    test_scenarios = [
        {
            "name": "Normal Fleet Vehicle Charging",
            "data": {
                "timestamp": datetime.now().isoformat(),
                "vehicle_id": "VH_001",
                "station_id": "CHG_01",
                "energy": 32.5,
                "billing": 8.13,
                "duration_minutes": 65,
                "power_kw": 50,
                "hour_of_day": 14,
                "day_of_week": 2
            }
        },
        {
            "name": "High Energy Consumption - Possible Meter Tampering",
            "data": {
                "timestamp": datetime.now().isoformat(),
                "vehicle_id": "VH_003",
                "station_id": "CHG_02",
                "energy": 95.0,  # Well above normal range
                "billing": 23.75,
                "duration_minutes": 180,
                "power_kw": 35,
                "hour_of_day": 9,
                "day_of_week": 1
            }
        },
        {
            "name": "Low Energy Consumption - Phantom Charge",
            "data": {
                "timestamp": datetime.now().isoformat(),
                "vehicle_id": "VH_005",
                "station_id": "CHG_03",
                "energy": 1.8,  # Suspiciously low
                "billing": 0.45,
                "duration_minutes": 5,
                "power_kw": 22,
                "hour_of_day": 20,
                "day_of_week": 3
            }
        },
        {
            "name": "Billing Rate Irregularity - Possible Fraud",
            "data": {
                "timestamp": datetime.now().isoformat(),
                "vehicle_id": "VH_007",
                "station_id": "CHG_04",
                "energy": 40.0,
                "billing": 4.00,  # $0.10/kWh instead of normal $0.25/kWh
                "duration_minutes": 90,
                "power_kw": 30,
                "hour_of_day": 16,
                "day_of_week": 4
            }
        },
        {
            "name": "Normal Long-Range Vehicle",
            "data": {
                "timestamp": datetime.now().isoformat(),
                "vehicle_id": "VH_008",
                "station_id": "CHG_05",
                "energy": 42.0,
                "billing": 10.50,
                "duration_minutes": 120,
                "power_kw": 25,
                "hour_of_day": 11,
                "day_of_week": 5
            }
        },
        {
            "name": "Overnight Charging - High Consumption",
            "data": {
                "timestamp": "2024-01-15T02:30:00Z",
                "vehicle_id": "VH_002",
                "station_id": "CHG_01",
                "energy": 85.0,  # High but during overnight charging
                "billing": 21.25,
                "duration_minutes": 380,
                "power_kw": 15,
                "hour_of_day": 2,
                "day_of_week": 0
            }
        }
    ]
    
    print(f"\nAnalyzing {len(test_scenarios)} Fleet Charging Sessions...")
    print("=" * 60)
    
    session_results = []
    anomaly_count = 0
    
    # Analyze each session
    for scenario in test_scenarios:
        session_json = json.dumps(scenario["data"])
        
        # Simulate real-time processing
        print("Processing...", end="", flush=True)
        time.sleep(0.3)
        print("\r" + " " * 15 + "\r", end="")
        
        result = detector.analyze_charging_session(session_json)
        session_results.append(result)
        
        print_session_result(scenario["name"], result, scenario["data"])
        
        if result['is_anomaly']:
            anomaly_count += 1
    
    # Time window analysis
    print(f"\n" + "=" * 60)
    print("TIME WINDOW ANALYSIS - 24 HOUR FLEET OVERVIEW")
    print("=" * 60)
    
    # Create session data for time window analysis
    session_logs = [json.dumps(scenario["data"]) for scenario in test_scenarios]
    window_analysis = detector.analyze_time_window(session_logs, window_hours=24)
    
    print(f"Analysis Period: {window_analysis['time_window_hours']} hours")
    print(f"Total Sessions: {window_analysis['total_sessions']}")
    print(f"Anomalies Detected: {window_analysis['anomalies_detected']}")
    print(f"Fleet Anomaly Rate: {window_analysis['anomaly_rate']:.1%}")
    print(f"Average Energy Consumption: {window_analysis['average_energy']} kWh")
    print(f"Energy Consumption Std Dev: {window_analysis['energy_std']} kWh")
    print(f"Maximum Anomaly Score: {window_analysis['max_anomaly_score']}")
    
    # Anomaly breakdown
    if window_analysis['anomaly_types']:
        print(f"\nANOMALY TYPES DETECTED:")
        for anomaly_type, count in window_analysis['anomaly_types'].items():
            print(f"  {anomaly_type}: {count} occurrences")
    
    # Fleet health assessment
    print(f"\nFLEET HEALTH ASSESSMENT:")
    if window_analysis['anomaly_rate'] > 0.15:  # >15%
        print("  STATUS: HIGH RISK - Multiple irregularities detected")
        print("  ACTION: Immediate investigation recommended")
    elif window_analysis['anomaly_rate'] > 0.08:  # >8%
        print("  STATUS: MODERATE RISK - Some irregularities detected")
        print("  ACTION: Enhanced monitoring recommended")
    else:
        print("  STATUS: NORMAL - Fleet operating within expected parameters")
        print("  ACTION: Continue routine monitoring")
    
    # Energy consumption analysis
    avg_energy = window_analysis['average_energy']
    expected_min, expected_max = detector.normal_energy_range
    
    print(f"\nENERGY CONSUMPTION ANALYSIS:")
    print(f"  Fleet Average: {avg_energy} kWh")
    print(f"  Expected Range: {expected_min}-{expected_max} kWh")
    
    if avg_energy < expected_min:
        print("  ALERT: Below normal consumption - check for connection issues")
    elif avg_energy > expected_max:
        print("  ALERT: Above normal consumption - check for meter accuracy")
    else:
        print("  STATUS: Energy consumption within normal fleet parameters")
    
    # Real-time monitoring simulation
    print(f"\nREAL-TIME MONITORING CAPABILITIES:")
    print("  * Continuous session analysis")
    print("  * Time-window pattern detection")
    print("  * Fleet-wide anomaly rate tracking")
    print("  * Vehicle-specific consumption profiling")
    print("  * Billing irregularity detection")
    print("  * Automated alert generation")
    
    print(f"\nFLEET MONITORING SUMMARY:")
    print(f"Sessions Analyzed: {len(test_scenarios)}")
    print(f"Anomalies Found: {anomaly_count}")
    print(f"Detection Rate: {(anomaly_count/len(test_scenarios))*100:.1f}%")
    print(f"System Status: Operational")
    
    print(f"\nFleet energy consumption monitoring system ready for production!")


if __name__ == "__main__":
    main()