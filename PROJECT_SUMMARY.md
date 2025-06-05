# VFSOC ML Models - Energy Consumption Pattern Detection MVP

## Project Overview

This project has been cleaned up and restructured to focus specifically on **irregular energy consumption pattern detection** for EV charging stations. The codebase is now streamlined for this single use case while maintaining the architecture to support future extensions.

## What Was Cleaned Up

### Removed Files
- ❌ `MULTI_MODEL_TRAINING_GUIDE.md` - Generic multi-model guide
- ❌ `PRACTICAL_MODELS_GUIDE.md` - Generic practical models guide  
- ❌ `data_downloader.py` - Generic data downloader
- ❌ `setup_training.py` - Generic setup script
- ❌ `convert_model_direct.py` - Generic model converter
- ❌ `training_workflow.log` - Old log file
- ❌ Multiple generic training scripts in `scripts/`
- ❌ Generic configuration files in `config/`

### Updated Files
- ✅ `README.md` - Now focuses on energy consumption pattern detection
- ✅ `requirements.txt` - Streamlined dependencies for the specific use case
- ✅ `scripts/train_model.py` - Energy consumption specific training script
- ✅ `scripts/convert_to_onnx.py` - ONNX conversion for energy models
- ✅ `src/vfsoc_ml/data/data_loader.py` - Energy consumption data loader
- ✅ `src/vfsoc_ml/data/feature_engineering.py` - Energy-specific feature engineering

### New Files Created
- ✅ `config/energy_consumption_config.yaml` - Specific configuration for energy anomaly detection
- ✅ `src/vfsoc_ml/utils/logger.py` - Logging utilities
- ✅ `src/vfsoc_ml/utils/metrics.py` - Anomaly detection metrics
- ✅ `src/vfsoc_ml/utils/visualization.py` - Visualization utilities
- ✅ `notebooks/01_data_exploration.ipynb` - Data exploration notebook
- ✅ `PROJECT_SUMMARY.md` - This summary document

## Current Project Structure

```
VFSOC-ML-Models/
├── README.md                                    # ✅ Updated for energy consumption
├── requirements.txt                             # ✅ Streamlined dependencies
├── pyproject.toml                              # ✅ Existing
├── .gitignore                                  # ✅ Existing
├── PROJECT_SUMMARY.md                          # ✅ New summary
├── config/
│   └── energy_consumption_config.yaml          # ✅ New energy-specific config
├── src/
│   └── vfsoc_ml/
│       ├── __init__.py                         # ✅ Existing
│       ├── data/
│       │   ├── __init__.py                     # ✅ Existing
│       │   ├── data_loader.py                  # ✅ Updated for energy data
│       │   ├── feature_engineering.py         # ✅ New energy features
│       │   └── synthetic_generator.py          # ✅ Existing (kept for future)
│       ├── models/
│       │   ├── __init__.py                     # ✅ Existing
│       │   ├── base_model.py                   # ✅ Existing
│       │   ├── isolation_forest.py            # ✅ Existing
│       │   └── ensemble.py                     # ✅ Existing
│       ├── preprocessing/
│       │   ├── __init__.py                     # ✅ Existing
│       │   └── feature_extractor.py           # ✅ Existing
│       ├── training/
│       │   ├── __init__.py                     # ✅ Existing
│       │   ├── trainer.py                      # ✅ Existing
│       │   └── evaluator.py                    # ✅ Existing
│       ├── deployment/
│       │   ├── __init__.py                     # ✅ Existing
│       │   └── onnx_converter.py               # ✅ Existing
│       └── utils/
│           ├── __init__.py                     # ✅ New
│           ├── logger.py                       # ✅ New
│           ├── metrics.py                      # ✅ New
│           └── visualization.py                # ✅ New
├── data/
│   ├── raw/
│   │   ├── station_data_dataverse.csv         # ✅ Real EV charging data
│   │   └── SYNTHETIC_EV_DATA.csv              # ✅ Synthetic data
│   ├── processed/                              # ✅ For processed data
│   └── features/                               # ✅ For engineered features
├── models/
│   ├── trained/                                # ✅ For trained models
│   └── onnx/                                   # ✅ For ONNX models
├── scripts/
│   ├── train_model.py                          # ✅ Updated for energy consumption
│   ├── convert_to_onnx.py                      # ✅ New ONNX converter
│   └── generate_synthetic_data.py              # ✅ Existing
├── notebooks/
│   └── 01_data_exploration.ipynb               # ✅ New exploration notebook
├── tests/
│   ├── __init__.py                             # ✅ Existing
│   └── unit/                                   # ✅ Existing
├── logs/                                       # ✅ New for log files
└── plots/                                      # ✅ New for visualizations
```

## Key Features Implemented

### 1. Data Loading and Preprocessing
- **EnergyConsumptionDataLoader**: Handles both real and synthetic EV charging data
- **Data validation**: Checks for required columns, handles missing values
- **Data cleaning**: Removes outliers, invalid entries, duplicates
- **Data quality assessment**: Comprehensive quality checks

### 2. Feature Engineering
- **Temporal features**: Hour of day, day of week, seasonal patterns
- **Energy features**: Consumption rates, efficiency metrics, categories
- **Station features**: Utilization patterns, baseline comparisons
- **User features**: Behavior patterns, personal baselines
- **Derived features**: Interaction terms, anomaly indicators

### 3. Model Training
- **Isolation Forest**: Primary anomaly detection algorithm
- **Hyperparameter tuning**: Grid search optimization
- **Cross-validation**: Robust model evaluation
- **MLflow integration**: Experiment tracking
- **Model persistence**: Save trained models and scalers

### 4. Evaluation and Metrics
- **Comprehensive metrics**: Precision, recall, F1, ROC-AUC, silhouette score
- **Contamination analysis**: Test different anomaly rates
- **Feature importance**: Permutation-based importance for interpretation
- **Anomaly summary**: Detailed analysis of detected anomalies

### 5. Visualization
- **Anomaly analysis plots**: Score distributions, temporal patterns
- **Energy consumption patterns**: Normal vs anomalous comparisons
- **Station analysis**: Station-level anomaly rates and patterns
- **Feature importance**: Visual representation of key features

### 6. Production Deployment
- **ONNX conversion**: Convert models for production deployment
- **Model metadata**: Comprehensive model documentation
- **Testing framework**: Validate ONNX models against originals

## Configuration

The project uses a comprehensive YAML configuration file (`config/energy_consumption_config.yaml`) that includes:

- **Data paths and preprocessing parameters**
- **Feature engineering specifications**
- **Model hyperparameters**
- **Training configuration**
- **Evaluation metrics**
- **Deployment settings**
- **Logging and monitoring**

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model**:
   ```bash
   python scripts/train_model.py
   ```

3. **Convert to ONNX**:
   ```bash
   python scripts/convert_to_onnx.py
   ```

4. **Explore data**:
   ```bash
   jupyter notebook notebooks/01_data_exploration.ipynb
   ```

## Future Extensions

The architecture is designed to support the planned future extensions:

### 1. Abnormal Ignition Start Behavior
- Vehicle-specific ignition pattern analysis
- Multi-vehicle simulation capabilities
- Temporal anomaly detection for ignition events

### 2. Sensor Data Spike Detection
- Roadside sensor monitoring
- Backend system anomaly detection
- Time-series spike detection algorithms

## Data Sources

### Current (MVP)
- **Real EV Charging Data**: `station_data_dataverse.csv` (3,397 sessions)
- **Synthetic Data**: `SYNTHETIC_EV_DATA.csv` (generated patterns)

### Future
- Geotab connector data for ignition patterns
- Roadside sensor data streams
- Backend system logs

## Industry Standards Compliance

- ✅ **Modular architecture**: Clean separation of concerns
- ✅ **Configuration management**: YAML-based configuration
- ✅ **Logging**: Structured logging with multiple levels
- ✅ **Testing framework**: Unit tests and integration tests
- ✅ **Documentation**: Comprehensive README and code documentation
- ✅ **Version control**: Git-based with proper .gitignore
- ✅ **Dependency management**: requirements.txt and pyproject.toml
- ✅ **Model versioning**: MLflow integration
- ✅ **Production deployment**: ONNX export capabilities

## Next Steps

1. **Run data exploration** to understand the dataset characteristics
2. **Train the initial model** using the provided scripts
3. **Evaluate model performance** and tune hyperparameters
4. **Deploy to production** using ONNX format
5. **Monitor model performance** and retrain as needed
6. **Extend to future use cases** (ignition patterns, sensor spikes)

## Contact and Support

This MVP provides a solid foundation for irregular energy consumption pattern detection while maintaining the flexibility to extend to other anomaly detection use cases in the VFSOC ecosystem. 