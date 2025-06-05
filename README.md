# VFSOC ML Models - Comprehensive Anomaly Detection Platform

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

This repository contains a comprehensive machine learning platform for the VFSOC (Vehicle Fleet Security Operations Center) project, designed to detect various types of anomalies across vehicle fleet operations and infrastructure systems.

**Current MVP: Irregular Energy Consumption Pattern Detection**

The platform currently focuses on detecting irregular energy consumption patterns in EV charging stations using unsupervised anomaly detection techniques, with architecture designed to support future extensions.

## Platform Capabilities

### Current Implementation (MVP)
**Irregular Energy Consumption Pattern Detection** for EV charging stations can identify:

- **Meter Tampering**: Manipulation of energy measurement devices
- **Unauthorized Power Drain**: Excessive or unauthorized energy consumption  
- **Broken Billing Logic**: Faulty billing calculations and discrepancies
- **Station Configuration Errors**: Misconfigured charging equipment parameters
- **Equipment Malfunction**: Hardware failures affecting energy delivery

### Future Extensions (Planned)

**Phase 2: Abnormal Ignition Start Behavior Detection**
- Detection of unusual vehicle ignition patterns using Geotab Connector data
- Analysis of start/stop sequences, timing anomalies, and unauthorized access
- Integration with fleet management systems for real-time monitoring

**Phase 3: Sensor Data Spike Detection**
- Monitoring of roadside sensors and backend system data
- Detection of sensor malfunctions, data corruption, and environmental anomalies
- Real-time processing of time-series sensor data

## Architecture Design

The platform follows a modular, extensible architecture that supports multiple use cases:

```
VFSOC Platform Architecture
├── Data Layer (Multi-source support)
├── Feature Engineering (Domain-specific)
├── ML Models (Algorithm library)
├── Evaluation Framework (Unified metrics)
├── Deployment Pipeline (ONNX/Production ready)
└── Alert System (Configurable outputs)
```

### Core ML Algorithms

**Primary Algorithm: Isolation Forest**
- **Unsupervised Learning**: No need for labeled anomaly data across all use cases
- **Outlier Detection**: Excellent for detecting sparse outliers in various data types
- **Real-time Performance**: Fast inference suitable for production environments
- **Scalable**: Efficient processing of large datasets from multiple sources
- **Robust**: Handles normal variations across different domains

**Secondary Algorithms**
1. **Z-score Analysis**: Statistical approach for small training sets
2. **One-Class SVM**: Learns tight boundaries around normal patterns
3. **LSTM Autoencoders**: For time-series anomaly detection (future implementation)

## Current Focus: Energy Consumption Detection

### Problem Statement
Irregular energy consumption patterns in EV charging stations indicating potential security and operational issues.

### ML Approach
- **Model**: Isolation Forest (primary) + Z-score analysis (secondary)
- **Features**: Energy consumption patterns, billing ratios, vehicle baselines, temporal factors
- **Output**: Real-time JSON alerts with severity classification

### Key Features Engineered

| Feature | Description | Use Case |
|---------|-------------|----------|
| `energy` | Energy delivered in session (kWh) | Energy Consumption |
| `billing_per_kWh` | Derived billing rate | Energy Consumption |
| `vehicle_mean_energy` | Historical average per vehicle | Energy Consumption |
| `z_score_energy` | Standardized energy score | Energy Consumption |

### Alert Output Format

```json
{
  "alert_type": "IrregularEnergyConsumption",
  "vehicle_id": "VH_002",
  "station_id": "CHG_01", 
  "timestamp": "2025-06-05T08:24:00Z",
  "energy": 92.5,
  "expected_range": "12-45 kWh",
  "anomaly_score": 0.98,
  "severity": "high"
}
```

### Data Simulation Strategy

**Current Dataset Simulation:**
- **Total Logs**: ~3,100 charging sessions
- **Coverage**: 5-10 vehicles across 5 charging stations
- **Normal Sessions**: 12-45 kWh with ±10% variability by vehicle type
- **Anomaly Injection**: 5-8% rate with specific patterns:
  - Low Energy: <5 kWh (phantom charges, equipment failures)
  - High Energy: >80 kWh (over-delivery, false logging)
  - Billing Anomalies: Distorted billing calculations

## Platform Features

- **Modular Architecture**: Easy extension for new use cases and data sources
- **Multiple ML Algorithms**: Isolation Forest, statistical methods, ensemble approaches
- **Domain-Agnostic Feature Engineering**: Configurable feature extraction for different data types
- **Unified Evaluation Framework**: Consistent metrics across all anomaly detection tasks
- **Production-Ready Deployment**: ONNX export, containerization support
- **Real-Time Processing**: Optimized for streaming data and immediate alerts
- **Comprehensive Logging**: MLflow integration for experiment tracking and model versioning
- **Industry Standards**: Following security, scalability, and maintainability best practices

## Project Structure

```
VFSOC-ML-Models/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── config/
│   ├── energy_consumption_config.yaml      # Current MVP configuration
│   ├── ignition_behavior_config.yaml       # Future Phase 2
│   └── sensor_spike_config.yaml            # Future Phase 3
├── src/
│   └── vfsoc_ml/
│       ├── __init__.py
│       ├── data/                           # Multi-source data handling
│       │   ├── __init__.py
│       │   ├── base_loader.py              # Base data loader interface
│       │   ├── energy_data_loader.py       # Energy consumption data
│       │   ├── geotab_data_loader.py       # Future: Geotab connector data
│       │   ├── sensor_data_loader.py       # Future: Sensor data
│       │   └── feature_engineering.py     # Domain-specific feature engineering
│       ├── models/                         # ML model library
│       │   ├── __init__.py
│       │   ├── base_anomaly_detector.py    # Base anomaly detection interface
│       │   ├── isolation_forest.py        # Isolation Forest implementation
│       │   ├── statistical_detectors.py   # Z-score, MAD, etc.
│       │   └── ensemble.py                # Ensemble methods
│       ├── evaluation/                     # Unified evaluation framework
│       │   ├── __init__.py
│       │   ├── metrics.py                  # Domain-agnostic metrics
│       │   └── validators.py              # Data quality validation
│       ├── deployment/                     # Production deployment
│       │   ├── __init__.py
│       │   ├── onnx_converter.py          # ONNX model conversion
│       │   ├── api_server.py              # REST API server
│       │   └── batch_processor.py         # Batch processing
│       └── utils/                          # Common utilities
│           ├── __init__.py
│           ├── logger.py                   # Logging utilities
│           ├── config_manager.py           # Configuration management
│           └── visualization.py           # Plotting and visualization
├── data/                                   # Data storage
│   ├── raw/                               # Raw datasets (all use cases)
│   ├── processed/                         # Processed datasets
│   └── features/                          # Feature datasets
├── models/                                # Model storage
│   ├── trained/                           # Trained models
│   └── onnx/                             # ONNX exported models
├── scripts/                               # Execution scripts
│   ├── train_model.py                     # General training script
│   ├── generate_synthetic_data.py         # Data generation
│   ├── convert_to_onnx.py                # Model conversion
│   └── deploy_model.py                   # Deployment script
├── notebooks/                             # Analysis notebooks
│   ├── 01_data_exploration.ipynb         # Data analysis
│   ├── 02_feature_engineering.ipynb      # Feature development
│   ├── 03_model_development.ipynb        # Model experimentation
│   └── 04_evaluation_analysis.ipynb      # Results analysis
├── tests/                                 # Test suite
│   ├── __init__.py
│   ├── unit/                             # Unit tests
│   ├── integration/                      # Integration tests
│   └── performance/                      # Performance tests
└── deployment/                           # Deployment configurations
    ├── docker/                           # Docker configurations
    ├── k8s/                             # Kubernetes configurations
    └── cloud/                           # Cloud deployment configs
```

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd VFSOC-ML-Models

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### 2. Configuration

```bash
# Edit configuration for your use case
# Current MVP: config/energy_consumption_config.yaml
# Future: config/ignition_behavior_config.yaml, config/sensor_spike_config.yaml
```

### 3. Generate Synthetic Data (for testing)

```bash
# Generate synthetic energy consumption data
python scripts/generate_synthetic_data.py --output data/raw/synthetic_energy_data.csv
```

### 4. Train Models

```bash
# Train the current MVP model (energy consumption)
python scripts/train_model.py

# Future: Train other models with different configs
# python scripts/train_model.py --config config/ignition_behavior_config.yaml
# python scripts/train_model.py --config config/sensor_spike_config.yaml
```

### 5. Convert to ONNX (Production Deployment)

```bash
# Convert trained model to ONNX for production
python scripts/convert_to_onnx.py
```

## Model Performance Targets

### Current MVP (Energy Consumption)
| Metric | Target | Description |
|--------|--------|-------------|
| Precision | > 0.90 | Minimize false positive alerts |
| Recall | > 0.85 | Catch majority of actual anomalies |
| F1-Score | > 0.87 | Balanced precision and recall |
| False Positive Rate | < 0.05 | Keep false alarms low |
| Inference Time | < 100ms | Real-time processing capability |

### Future Performance Standards
- **Ignition Behavior**: Sub-second detection of abnormal start sequences
- **Sensor Spikes**: Real-time processing of high-frequency sensor data streams
- **Cross-Platform**: Consistent performance across all use cases

## Use Case Extensions

### Adding New Use Cases

The platform is designed for easy extension. To add a new use case:

1. **Create Configuration**: Add new config file (e.g., `config/new_usecase_config.yaml`)
2. **Data Loader**: Implement data loader in `src/vfsoc_ml/data/`
3. **Features**: Add domain-specific features to feature engineering
4. **Training**: Use existing training pipeline with new configuration
5. **Evaluation**: Leverage unified evaluation framework
6. **Deployment**: Use existing ONNX conversion and deployment tools

### Configuration Management

Each use case has its own configuration file following a standard structure:
- Data sources and preprocessing parameters
- Feature engineering specifications
- Model parameters and hyperparameters
- Evaluation metrics and thresholds
- Alert formatting and severity levels
- Deployment configurations

## Production Deployment Options

### 1. Real-time API
```bash
# Start REST API server
python scripts/deploy_model.py --mode api --port 8080
```

### 2. Batch Processing
```bash
# Process batch data
python scripts/deploy_model.py --mode batch --input data.csv --output results.json
```

### 3. Container Deployment
```bash
# Build Docker container
docker build -t vfsoc-ml:latest .

# Run container
docker run -p 8080:8080 vfsoc-ml:latest
```

### 4. Cloud Deployment
- Kubernetes configurations provided in `deployment/k8s/`
- Cloud-specific configs in `deployment/cloud/`
- Supports AWS, Azure, GCP deployments

## Development Workflow

### 1. Research Phase
- Use Jupyter notebooks for exploration (`notebooks/`)
- Experiment with new algorithms and features
- Validate approaches on synthetic/test data

### 2. Implementation Phase
- Implement new features in modular components
- Add comprehensive unit and integration tests
- Update configuration files and documentation

### 3. Evaluation Phase
- Use unified evaluation framework for consistent metrics
- Compare performance across different algorithms
- Validate on real-world data when available

### 4. Deployment Phase
- Convert models to ONNX for production efficiency
- Deploy using containerized infrastructure
- Monitor performance and data drift

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-detection-algorithm`)
3. Make your changes following the established patterns
4. Add tests for new functionality
5. Update documentation and configuration as needed
6. Submit a pull request

## Future Roadmap

### Phase 1: MVP (Current) - Energy Consumption Detection
- [x] Core platform architecture
- [x] Energy consumption anomaly detection
- [x] Synthetic data generation
- [x] Model training and evaluation pipeline
- [x] ONNX export capabilities

### Phase 2: Ignition Behavior Detection
- [ ] Geotab Connector integration
- [ ] Ignition sequence analysis algorithms
- [ ] Temporal pattern recognition for start/stop events
- [ ] Fleet-specific behavioral baselines

### Phase 3: Sensor Data Spike Detection
- [ ] Time-series data processing pipeline
- [ ] Real-time streaming data support
- [ ] Environmental sensor integration
- [ ] Advanced time-series anomaly detection

### Phase 4: Platform Integration
- [ ] Multi-use case ensemble models
- [ ] Cross-domain anomaly correlation
- [ ] Advanced visualization dashboards
- [ ] Enterprise security integrations


## Support

For questions, issues, or contributions:
- Create an issue in the repository
- Contact the VFSOC team
- Review documentation in `docs/` directory
