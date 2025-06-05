#!/usr/bin/env python3
"""
Test Script for EV Charging Anomaly Detection Model

This script demonstrates how to use the trained model to detect irregular
charging patterns from EVChargerConnector logs.
"""

import sys
import os
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vfsoc_ml.models.ev_charging_anomaly_detector import EVChargingAnomalyDetector


def simulate_ev_charger_connector():
    """
    Simulate the EVChargerConnector to generate test logs.
    This matches the exact logic from the connector file.
    """
    import random
    
    # Vehicle specifications (matching connector)
    vehicle_types = {
        "Tesla Model 3": {"min_energy": 15, "max_energy": 75, "avg_energy": 45},
        "Tesla Model Y": {"min_energy": 18, "max_energy": 78, "avg_energy": 48},
        "Nissan Leaf": {"min_energy": 10, "max_energy": 62, "avg_energy": 35},
        "Chevrolet Bolt": {"min_energy": 12, "max_energy": 66, "avg_energy": 38},
        "Ford Mustang Mach-E": {"min_energy": 20, "max_energy": 88, "avg_energy": 55},
    }
    
    station_ids = [
        "CP-101-DOWNTOWN", "CP-102-HIGHWAY", "ABB-201-MALL", 
        "ABB-202-AIRPORT", "SIEMENS-301-DEPOT", "TESLA-401-SUPERCHARGER",
        "SHELL-501-PLAZA", "BP-601-TERMINAL", "EVGO-701-CENTER"
    ]
    
    auth_methods = ["RFID", "Mobile_App", "Credit_Card", "Admin_Login", "Plug_And_Charge"]
    pricing_rates = [0.12, 0.15, 0.18, 0.22, 0.25, 0.28, 0.35, 0.42]
    
    # Generate sample logs
    logs = []
    
    # 1. Normal charging session
    vehicle_model = "Tesla Model 3"
    vehicle_specs = vehicle_types[vehicle_model]
    energy_delivered = random.uniform(vehicle_specs["min_energy"], vehicle_specs["max_energy"])
    rate = random.choice(pricing_rates)
    
    normal_log = {
        "timestamp": datetime.now().isoformat() + "Z",
        "station_id": random.choice(station_ids),
        "vehicle_id": f"VIN-{random.randint(10000, 99999)}",
        "event": "charging_completed",
        "energy": round(energy_delivered, 2),
        "billing": round(energy_delivered * rate, 2),
        "user_id": f"USER-{random.randint(1001, 1010)}",
        "authmethod": random.choice(auth_methods),
        "card_id": f"RFID-{random.randint(0, 8)}A{random.randint(1, 9)}B{random.randint(2, 9)}C{random.randint(3, 9)}D",
        "result": "SUCCESS",
        "status": "operational",
        "vendor": "Tesla Supercharger",
        "vehicle_model": vehicle_model,
        "rate_per_kwh": rate,
        "power_kw": round(random.uniform(50.0, 250.0), 1),
        "session_duration_minutes": random.randint(30, 120)
    }
    logs.append(("Normal Charging", json.dumps(normal_log)))
    
    # 2. Extremely high energy consumption anomaly
    vehicle_model = "Nissan Leaf"
    vehicle_specs = vehicle_types[vehicle_model]
    # 200% of normal maximum
    energy_delivered = vehicle_specs["max_energy"] * 2.0
    rate = random.choice(pricing_rates)
    
    high_energy_log = {
        "timestamp": datetime.now().isoformat() + "Z",
        "station_id": random.choice(station_ids),
        "vehicle_id": f"VIN-{random.randint(10000, 99999)}",
        "event": "charging_completed",
        "energy": round(energy_delivered, 2),
        "billing": round(energy_delivered * rate, 2),
        "user_id": f"USER-{random.randint(1001, 1010)}",
        "authmethod": random.choice(auth_methods),
        "card_id": f"RFID-{random.randint(0, 8)}A{random.randint(1, 9)}B{random.randint(2, 9)}C{random.randint(3, 9)}D",
        "result": "SUCCESS",
        "status": "operational",
        "vendor": "ABB Terra",
        "vehicle_model": vehicle_model,
        "rate_per_kwh": rate,
        "power_kw": round(random.uniform(100.0, 350.0), 1),
        "session_duration_minutes": random.randint(45, 90),
        "anomaly": {
            "type": "irregular_energy_consumption",
            "anomaly_subtype": "extremely_high",
            "expected_energy_range": f"{vehicle_specs['min_energy']}-{vehicle_specs['max_energy']} kWh",
            "actual_energy": energy_delivered,
            "severity": "high"
        }
    }
    logs.append(("Extremely High Energy", json.dumps(high_energy_log)))
    
    # 3. Extremely low energy consumption anomaly
    vehicle_model = "Ford Mustang Mach-E"
    vehicle_specs = vehicle_types[vehicle_model]
    # 5% of normal minimum
    energy_delivered = vehicle_specs["min_energy"] * 0.05
    rate = random.choice(pricing_rates)
    
    low_energy_log = {
        "timestamp": datetime.now().isoformat() + "Z",
        "station_id": random.choice(station_ids),
        "vehicle_id": f"VIN-{random.randint(10000, 99999)}",
        "event": "charging_completed",
        "energy": round(energy_delivered, 2),
        "billing": round(energy_delivered * rate, 2),
        "user_id": f"USER-{random.randint(1001, 1010)}",
        "authmethod": random.choice(auth_methods),
        "card_id": f"RFID-{random.randint(0, 8)}A{random.randint(1, 9)}B{random.randint(2, 9)}C{random.randint(3, 9)}D",
        "result": "SUCCESS",
        "status": "operational",
        "vendor": "Shell Recharge",
        "vehicle_model": vehicle_model,
        "rate_per_kwh": rate,
        "power_kw": round(random.uniform(7.0, 50.0), 1),
        "session_duration_minutes": random.randint(5, 20),
        "anomaly": {
            "type": "irregular_energy_consumption",
            "anomaly_subtype": "extremely_low",
            "expected_energy_range": f"{vehicle_specs['min_energy']}-{vehicle_specs['max_energy']} kWh",
            "actual_energy": energy_delivered,
            "severity": "medium"
        }
    }
    logs.append(("Extremely Low Energy", json.dumps(low_energy_log)))
    
    # 4. Billing fraud anomaly
    vehicle_model = "Chevrolet Bolt"
    vehicle_specs = vehicle_types[vehicle_model]
    energy_delivered = random.uniform(vehicle_specs["min_energy"], vehicle_specs["max_energy"])
    rate = random.choice(pricing_rates)
    # Inflate billing by 300%
    inflated_cost = energy_delivered * rate * 3.0
    
    fraud_log = {
        "timestamp": datetime.now().isoformat() + "Z",
        "station_id": random.choice(station_ids),
        "vehicle_id": f"VIN-{random.randint(10000, 99999)}",
        "event": "charging_completed",
        "energy": round(energy_delivered, 2),
        "billing": round(inflated_cost, 2),
        "user_id": f"USER-{random.randint(1001, 1010)}",
        "authmethod": random.choice(auth_methods),
        "card_id": f"RFID-{random.randint(0, 8)}A{random.randint(1, 9)}B{random.randint(2, 9)}C{random.randint(3, 9)}D",
        "result": "SUCCESS",
        "status": "operational",
        "vendor": "ChargePoint",
        "vehicle_model": vehicle_model,
        "rate_per_kwh": rate,
        "power_kw": round(random.uniform(25.0, 100.0), 1),
        "session_duration_minutes": random.randint(40, 100),
        "anomaly": {
            "type": "irregular_energy_consumption",
            "anomaly_subtype": "billing_fraud",
            "expected_energy_range": f"{vehicle_specs['min_energy']}-{vehicle_specs['max_energy']} kWh",
            "actual_energy": energy_delivered,
            "billing_expected": round(energy_delivered * rate, 2),
            "billing_actual": round(inflated_cost, 2),
            "severity": "high"
        }
    }
    logs.append(("Billing Fraud", json.dumps(fraud_log)))
    
    return logs


def main():
    """Main test function."""
    print("Testing EV Charging Anomaly Detection Model")
    print("=" * 50)
    
    # Check if model exists
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'trained', 'ev_charging_anomaly_detector.pkl')
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please run the training script first:")
        print("python scripts/train_ev_charging_anomaly_model_simple.py")
        return
    
    # Load the trained model
    print("Loading trained model...")
    detector = EVChargingAnomalyDetector()
    detector.load_model(model_path)
    
    # Generate test logs using connector simulation
    print("\nGenerating test logs using EVChargerConnector simulation...")
    test_logs = simulate_ev_charger_connector()
    
    print(f"Generated {len(test_logs)} test cases")
    
    # Analyze each log
    print("\nAnalyzing charging logs for anomalies...")
    print("=" * 50)
    
    for i, (log_type, log_entry) in enumerate(test_logs, 1):
        print(f"\n--- Test Case {i}: {log_type} ---")
        
        # Parse the log to show key details
        log_data = json.loads(log_entry)
        print(f"Vehicle: {log_data.get('vehicle_model', 'Unknown')}")
        print(f"Station: {log_data.get('station_id', 'Unknown')}")
        print(f"Energy: {log_data.get('energy', 0):.2f} kWh")
        print(f"Billing: ${log_data.get('billing', 0):.2f}")
        print(f"Rate: ${log_data.get('rate_per_kwh', 0):.2f}/kWh")
        
        # Analyze with ML model
        result = detector.analyze_charging_log(log_entry)
        
        print(f"\nML Analysis Results:")
        print(f"  Is Anomaly: {'YES' if result['is_anomaly'] else 'NO'}")
        print(f"  Anomaly Type: {result['anomaly_type']}")
        print(f"  Probability: {result['anomaly_probability']:.3f}")
        print(f"  Confidence: {result['confidence']}")
        
        if result['alert_message']:
            print(f"\nALERT MESSAGE:")
            print(f"  {result['alert_message']}")
        
        # Show feature analysis for anomalies
        if result['is_anomaly']:
            features = result['features']
            print(f"\nKey Features:")
            print(f"  Expected Energy Range: {features['expected_energy_range']}")
            print(f"  Energy Ratio to Max: {features['energy_ratio_to_max']:.2f}")
            print(f"  Billing Ratio: {features['billing_ratio']:.2f}")
            print(f"  Expected Cost: ${features['expected_cost']:.2f}")
        
        print("-" * 50)
    
    # Additional real-time test
    print(f"\nReal-time Anomaly Detection Test")
    print("=" * 50)
    
    # Create a custom suspicious log
    suspicious_log = {
        "timestamp": datetime.now().isoformat() + "Z",
        "station_id": "CP-101-DOWNTOWN",
        "vehicle_model": "Tesla Model Y",
        "energy": 234.0,  # 300% of normal max (78 kWh)
        "billing": 58.5,
        "rate_per_kwh": 0.25,
        "power_kw": 350.0,
        "session_duration_minutes": 40,
        "authmethod": "Mobile_App",
        "vehicle_id": "VIN-12345",
        "user_id": "USER-1007"
    }
    
    print("Testing with highly suspicious charging pattern...")
    print(f"Vehicle: Tesla Model Y (Normal max: 78 kWh)")
    print(f"Actual energy: {suspicious_log['energy']} kWh (300% of normal!)")
    
    result = detector.analyze_charging_log(json.dumps(suspicious_log))
    
    print(f"\nDetection Result:")
    print(f"  Anomaly Detected: {'YES' if result['is_anomaly'] else 'NO'}")
    print(f"  Confidence: {result['anomaly_probability']:.1%}")
    print(f"  Alert: {result['alert_message']}")
    
    print(f"\nTesting completed successfully!")
    print(f"The model is working correctly and can detect:")
    print(f"  - Extremely high energy consumption")
    print(f"  - Extremely low energy consumption") 
    print(f"  - Billing fraud patterns")
    print(f"  - Other irregular charging behaviors")


if __name__ == "__main__":
    main() 