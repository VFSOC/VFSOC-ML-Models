<<<<<<< HEAD
# VFSOC ML Models - GPS Jamming Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

This repository contains machine learning models for the VFSOC (Vehicle Fleet Security Operations Center) project, specifically focused on **GPS Jamming Detection** using advanced signal processing and anomaly detection techniques.

## Problem Statement

GPS jamming attacks represent a critical security threat to vehicle fleets, where malicious actors use jamming devices to disrupt GPS signals. This can lead to:

- Loss of vehicle tracking capabilities
- Potential vehicle theft
- Fleet operational disruptions
- Safety risks for drivers and cargo

## Solution Approach

Our ML-based GPS jamming detection system uses:

- **Signal Processing Features**: GPS signal strength, loss patterns, temporal analysis
- **Anomaly Detection**: Statistical and machine learning approaches to identify unusual signal patterns
- **Ensemble Methods**: Combining multiple algorithms for robust detection
- **Real-time Processing**: Models optimized for production deployment with ONNX

## Key Features

-  **Industry Standard Architecture**: Modular, scalable, and maintainable codebase
-  **Multiple ML Algorithms**: Isolation Forest, Random Forest, LSTM, and ensemble methods
-  **Comprehensive Feature Engineering**: Advanced signal processing and temporal features
-  **Model Versioning**: MLflow integration for experiment tracking
-  **Production Ready**: ONNX export for deployment in ingestion pipeline
-  **Synthetic Data Generation**: Integration with Geotab connector for training data
-  **Comprehensive Testing**: Unit tests, integration tests, and model validation
-  **Documentation**: Detailed docs with research references and implementation guides

## Project Structure

```
VFSOC-ML-Models/
├── README.md
├── requirements.txt
├── pyproject.toml
├── setup.py
├── .env.example
├── .gitignore
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── config/
│   ├── model_config.yaml
│   ├── feature_config.yaml
│   └── training_config.yaml
├── src/
│   └── vfsoc_ml/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── data_loader.py
│       │   ├── feature_engineering.py
│       │   └── synthetic_generator.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base_model.py
│       │   ├── isolation_forest.py
│       │   ├── random_forest.py
│       │   ├── lstm_detector.py
│       │   └── ensemble.py
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── signal_processor.py
│       │   └── feature_extractor.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   └── evaluator.py
│       ├── deployment/
│       │   ├── __init__.py
│       │   ├── onnx_converter.py
│       │   └── model_server.py
│       └── utils/
│           ├── __init__.py
│           ├── logger.py
│           ├── metrics.py
│           └── visualization.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_model_evaluation.ipynb
├── tests/
│   ├── __init__.py
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── models/
│   ├── trained/
│   ├── onnx/
│   └── experiments/
├── scripts/
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── generate_synthetic_data.py
│   └── convert_to_onnx.py
├── docs/
│   ├── api_reference.md
│   ├── model_architecture.md
│   ├── deployment_guide.md
│   └── research_background.md
└── research/
    ├── papers/
    ├── benchmarks/
    └── experiments/
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
# Copy environment template
cp .env.example .env

# Edit configuration files
# config/model_config.yaml - Model hyperparameters
# config/feature_config.yaml - Feature engineering settings
# config/training_config.yaml - Training parameters
```

### 3. Generate Synthetic Data

```bash
# Generate training data using Geotab connector
python scripts/generate_synthetic_data.py --config config/training_config.yaml
```

### 4. Train Models

```bash
# Train all models
python scripts/train_model.py --config config/model_config.yaml

# Train specific model
python scripts/train_model.py --model isolation_forest --config config/model_config.yaml
```

### 5. Evaluate Models

```bash
# Evaluate trained models
python scripts/evaluate_model.py --model-path models/trained/gps_jamming_detector.pkl
```

### 6. Convert to ONNX

```bash
# Convert best model to ONNX for production deployment
python scripts/convert_to_onnx.py --model-path models/trained/best_model.pkl --output models/onnx/gps_jamming_detector.onnx
```

## Models Overview

### Primary Algorithm: Isolation Forest

Based on research, **Isolation Forest** is selected as the primary algorithm for GPS jamming detection because:

- **Unsupervised Learning**: No need for labeled jamming data
- **Anomaly Detection**: Excellent for detecting unusual signal patterns
- **Real-time Performance**: Fast inference suitable for production
- **Robust to Noise**: Handles GPS signal variations well

### Secondary Algorithms

1. **Random Forest Classifier**: For supervised learning when labeled data is available
2. **LSTM Autoencoder**: For temporal pattern analysis and sequence anomaly detection
3. **Ensemble Model**: Combines multiple approaches for improved accuracy

## Key Features for GPS Jamming Detection

Based on research and VFSOC feature engineering:

### Signal Features
- GPS signal strength variations
- Signal loss frequency and duration
- Signal strength variance and patterns
- Trip-based jamming event clustering

### Temporal Features
- Time-based patterns in signal loss
- Frequency of jamming events per trip
- Duration and spacing of jamming incidents
- Cross-correlation analysis of signal patterns

### Contextual Features
- Vehicle location and movement patterns
- Time of day and operational context
- Driver presence and authentication status
- Environmental factors affecting GPS reception

## Model Performance Targets

| Metric | Target | Current Best |
|--------|--------|-------------|
| Precision | > 0.90 | TBD |
| Recall | > 0.85 | TBD |
| F1-Score | > 0.87 | TBD |
| False Positive Rate | < 0.05 | TBD |
| Inference Time | < 100ms | TBD |

## Research Foundation

This implementation is based on cutting-edge research in GNSS interference detection:

- **Signal Processing Approaches**: Correlation peak monitoring, power-based detection
- **Machine Learning Methods**: SVM, Random Forest, Neural Networks for anomaly detection
- **Feature Engineering**: Time-series analysis, statistical signal processing
- **Ensemble Techniques**: Combining multiple detection algorithms

Key papers referenced:
- "Recent Advances on Jamming and Spoofing Detection in GNSS" (Sensors, 2024)
- "Self-Supervised Federated GNSS Spoofing Detection" (arXiv, 2025)
- "Towards Simple Machine Learning Baselines for GNSS RFI Detection" (arXiv, 2024)

## Integration with VFSOC

This ML model integrates seamlessly with the existing VFSOC infrastructure:

1. **Data Source**: Uses synthetic data from `GeotabConnector` for training
2. **Feature Pipeline**: Leverages existing `GeotabFeatureExtractor`
3. **Alert System**: Integrates with `GpsJammingRule` for production deployment
4. **Database**: Stores model predictions and confidence scores

## Contributing

Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions and support:
- Create an issue in this repository
- Contact the VFSOC team
- Check the [documentation](docs/)

## Acknowledgments

- VFSOC project team for infrastructure and requirements
- Research community for GPS jamming detection methodologies
- Open source ML community for tools and frameworks 
=======
# VFSOC-ML-Models
>>>>>>> a56e80c169df664c6ca637533262909c7694b4e7
