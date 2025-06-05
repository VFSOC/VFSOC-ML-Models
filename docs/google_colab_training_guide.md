# Google Colab Training Guide for VFSOC ML Models

This guide provides step-by-step instructions for training the VFSOC ML models (Irregular Energy Consumption Detection) in Google Colab.

## Overview

Google Colab provides a free environment with GPU support that's perfect for training our anomaly detection models. This guide covers:

- Setting up the environment in Colab
- Uploading and preparing data
- Training the energy consumption anomaly detection model
- Downloading trained models and results
- Advanced configurations for different use cases

## Prerequisites

- Google account with access to Google Colab
- Basic familiarity with Python and Jupyter notebooks
- Understanding of the VFSOC ML platform architecture

---

## Step 1: Environment Setup

### 1.1 Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Sign in with your Google account
3. Create a new notebook: `File > New notebook`
4. Rename your notebook to "VFSOC_ML_Training"

### 1.2 Enable GPU (Optional but Recommended)

```python
# Check if GPU is available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

To enable GPU:
1. Go to `Runtime > Change runtime type`
2. Set `Hardware accelerator` to `GPU`
3. Click `Save`

### 1.3 Install Required Packages

```python
# Install required packages
!pip install scikit-learn pandas numpy matplotlib seaborn pyyaml
!pip install mlflow onnx onnxruntime joblib tqdm loguru
!pip install great-expectations plotly

# Verify installations
import sklearn
import pandas as pd
import numpy as np
import yaml
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Pandas version: {pd.__version__}")
```

---

## Step 2: Upload Project Files

### 2.1 Clone Repository (if public) or Upload Files

**Option A: Clone from Repository**
```python
# If your repository is public
!git clone <your-repository-url>
%cd VFSOC-ML-Models
```

**Option B: Upload Files Manually**
```python
# Create project structure
!mkdir -p VFSOC-ML-Models/{config,src/vfsoc_ml/{data,utils},scripts,data/{raw,processed,features},models/{trained,onnx},logs}

# Upload files using Colab's file upload
from google.colab import files
import io

# Upload key files one by one
print("Upload the following files:")
print("1. config/energy_consumption_config.yaml")
print("2. src/vfsoc_ml/ (all Python files)")
print("3. scripts/ (all Python scripts)")
print("4. requirements.txt")

# For each file upload:
uploaded = files.upload()
for filename in uploaded.keys():
    print(f'Uploaded: {filename}')
```

### 2.2 Set Up Project Structure

```python
# Navigate to project directory
%cd VFSOC-ML-Models

# Add src to Python path
import sys
sys.path.append('src')

# Verify structure
!ls -la
!ls src/vfsoc_ml/
```

---

## Step 3: Prepare Data

### 3.1 Generate Synthetic Data

```python
# Generate synthetic training data
!python scripts/generate_synthetic_data.py --output data/raw/synthetic_energy_data.csv

# Verify data generation
import pandas as pd
data = pd.read_csv('data/raw/synthetic_energy_data.csv')
print(f"Generated {len(data)} sessions")
print(f"Anomaly rate: {data['is_anomaly'].mean():.1%}")
print("\nData preview:")
print(data.head())
```

### 3.2 Upload Real Data (Optional)

```python
# If you have real EV charging data, upload it
from google.colab import files

print("Upload your real charging station data (CSV format):")
uploaded = files.upload()

# Move uploaded data to correct location
import shutil
for filename in uploaded.keys():
    if filename.endswith('.csv'):
        shutil.move(filename, f'data/raw/{filename}')
        print(f'Moved {filename} to data/raw/')
```

---

## Step 4: Configure Training

### 4.1 Review Configuration

```python
# Load and display configuration
import yaml

with open('config/energy_consumption_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

print("Current Configuration:")
print(f"- Total logs: {config['simulation']['total_logs']}")
print(f"- Anomaly rate: {config['simulation']['anomalies']['injection_rate']}")
print(f"- Contamination: {config['model']['isolation_forest']['contamination']}")
print(f"- Core features: {config['features']['core']}")
```

### 4.2 Modify Configuration (Optional)

```python
# Modify configuration for Colab environment
config['training']['model_save_path'] = '/content/VFSOC-ML-Models/models/trained/'
config['mlflow']['tracking_uri'] = '/content/mlflow.db'

# Adjust for faster training in Colab
config['model']['isolation_forest']['n_estimators'] = 100  # Reduce for speed
config['training']['hyperparameter_tuning'] = False  # Disable for faster training

# Save modified configuration
with open('config/energy_consumption_config.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)

print("Configuration updated for Colab environment")
```

---

## Step 5: Train the Model

### 5.1 Start Training

```python
# Import required modules
sys.path.append('src')

# Run training
!python scripts/train_model.py

# Monitor training progress
print("Training completed! Check the output above for results.")
```

### 5.2 Monitor Training Progress

```python
# Check if models were saved
!ls -la models/trained/

# Load and display training results
import json

try:
    with open('models/trained/evaluation_results.json', 'r') as f:
        results = json.load(f)
    
    print("Training Results:")
    print("="*50)
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Precision: {metrics.get('precision', 0):.3f}")
        print(f"  Recall: {metrics.get('recall', 0):.3f}")
        print(f"  F1-Score: {metrics.get('f1_score', 0):.3f}")
        print(f"  Anomaly Rate: {metrics.get('anomaly_rate', 0):.1%}")
        
except FileNotFoundError:
    print("Training results not found. Check training output for errors.")
```

### 5.3 View Sample Alerts

```python
# Display sample alerts generated by the model
try:
    with open('models/trained/sample_alerts.json', 'r') as f:
        alerts = json.load(f)
    
    print("Sample Alerts Generated:")
    print("="*50)
    for i, alert in enumerate(alerts[:3]):  # Show first 3 alerts
        print(f"\nAlert {i+1}:")
        print(json.dumps(alert, indent=2))
        
except FileNotFoundError:
    print("Sample alerts not found.")
```

---

## Step 6: Model Evaluation and Visualization

### 6.1 Evaluate Model Performance

```python
# Load trained model and evaluate
import joblib
import numpy as np
from sklearn.metrics import classification_report

# Load the isolation forest model
model_path = 'models/trained/isolation_forest_model.pkl'
try:
    isolation_forest = joblib.load(model_path)
    print("Model loaded successfully!")
    print(f"Model parameters: {isolation_forest.get_params()}")
except FileNotFoundError:
    print("Model file not found. Ensure training completed successfully.")
```

### 6.2 Create Visualizations

```python
# Create visualizations of results
import matplotlib.pyplot as plt
import seaborn as sns

# Load test data for visualization
test_data = pd.read_csv('data/raw/synthetic_energy_data.csv')

# Plot energy distribution
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.hist(test_data['kwhTotal'], bins=50, alpha=0.7, color='blue')
plt.title('Energy Consumption Distribution')
plt.xlabel('Energy (kWh)')
plt.ylabel('Frequency')

plt.subplot(2, 2, 2)
anomalies = test_data[test_data['is_anomaly'] == True]
normal = test_data[test_data['is_anomaly'] == False]
plt.scatter(normal['kwhTotal'], normal['chargeTimeHrs'], alpha=0.6, label='Normal', color='blue')
plt.scatter(anomalies['kwhTotal'], anomalies['chargeTimeHrs'], alpha=0.8, label='Anomaly', color='red')
plt.xlabel('Energy (kWh)')
plt.ylabel('Charge Time (hours)')
plt.title('Energy vs Charge Time')
plt.legend()

plt.subplot(2, 2, 3)
billing_rate = test_data['dollars'] / (test_data['kwhTotal'] + 1e-6)
plt.hist(billing_rate, bins=50, alpha=0.7, color='green')
plt.title('Billing Rate Distribution')
plt.xlabel('Rate ($/kWh)')
plt.ylabel('Frequency')

plt.subplot(2, 2, 4)
vehicle_counts = test_data['userId'].value_counts()
plt.bar(range(len(vehicle_counts)), vehicle_counts.values)
plt.title('Sessions per Vehicle')
plt.xlabel('Vehicle Index')
plt.ylabel('Number of Sessions')

plt.tight_layout()
plt.show()
```

---

## Step 7: Convert to ONNX (Production Ready)

### 7.1 Convert Model to ONNX Format

```python
# Convert the trained model to ONNX format for production deployment
!python scripts/convert_to_onnx.py

# Check if ONNX model was created
!ls -la models/onnx/
```

### 7.2 Test ONNX Model

```python
# Test the ONNX model
import onnxruntime as ort

try:
    # Load ONNX model
    onnx_path = 'models/onnx/energy_consumption_anomaly_detector.onnx'
    ort_session = ort.InferenceSession(onnx_path)
    
    print("ONNX model loaded successfully!")
    print(f"Input shape: {ort_session.get_inputs()[0].shape}")
    print(f"Output shape: {ort_session.get_outputs()[0].shape}")
    
    # Test with sample data
    # You would need to prepare input data in the correct format
    print("ONNX model is ready for production deployment!")
    
except FileNotFoundError:
    print("ONNX model not found. Check conversion process.")
```

---

## Step 8: Download Results

### 8.1 Package Results for Download

```python
# Create a zip file with all important results
import zipfile
import os

zip_filename = 'vfsoc_ml_trained_models.zip'

with zipfile.ZipFile(zip_filename, 'w') as zipf:
    # Add trained models
    if os.path.exists('models/trained/'):
        for root, dirs, files in os.walk('models/trained/'):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, '.'))
    
    # Add ONNX models
    if os.path.exists('models/onnx/'):
        for root, dirs, files in os.walk('models/onnx/'):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, '.'))
    
    # Add configuration
    if os.path.exists('config/energy_consumption_config.yaml'):
        zipf.write('config/energy_consumption_config.yaml')
    
    # Add logs if available
    if os.path.exists('logs/'):
        for root, dirs, files in os.walk('logs/'):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, '.'))

print(f"Results packaged in: {zip_filename}")
```

### 8.2 Download Results

```python
# Download the packaged results
from google.colab import files

files.download(zip_filename)
print("Download started! Check your Downloads folder.")

# Also download individual important files
files.download('models/trained/evaluation_results.json')
files.download('models/trained/sample_alerts.json')
```

---

## Step 9: Advanced Configurations

### 9.1 Training with Custom Data

```python
# If you want to train with your own data format
def prepare_custom_data(data_path):
    """
    Prepare custom data for training.
    Modify this function based on your data format.
    """
    df = pd.read_csv(data_path)
    
    # Map your columns to expected format
    column_mapping = {
        'your_energy_column': 'kwhTotal',
        'your_time_column': 'chargeTimeHrs',
        'your_cost_column': 'dollars',
        'your_vehicle_column': 'userId',
        'your_station_column': 'stationId'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Add any missing required columns
    required_columns = ['sessionId', 'kwhTotal', 'chargeTimeHrs', 'userId', 'stationId']
    for col in required_columns:
        if col not in df.columns:
            if col == 'sessionId':
                df[col] = [f"SESSION_{i:06d}" for i in range(len(df))]
            else:
                df[col] = 0  # Default value
    
    return df

# Use custom data
# custom_data = prepare_custom_data('path_to_your_data.csv')
# custom_data.to_csv('data/raw/station_data_dataverse.csv', index=False)
```

### 9.2 Hyperparameter Tuning

```python
# Enable hyperparameter tuning for better performance
config['training']['hyperparameter_tuning'] = True
config['training']['param_grid']['isolation_forest']['n_estimators'] = [50, 100, 200]
config['training']['param_grid']['isolation_forest']['contamination'] = [0.05, 0.06, 0.08]

# Save updated config
with open('config/energy_consumption_config.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)

print("Hyperparameter tuning enabled. Training will take longer but may yield better results.")
```

### 9.3 MLflow Experiment Tracking

```python
# View MLflow experiment results
import mlflow

# Set MLflow tracking URI
mlflow.set_tracking_uri('file:///content/mlflow.db')

# List experiments
experiments = mlflow.search_experiments()
for exp in experiments:
    print(f"Experiment: {exp.name} (ID: {exp.experiment_id})")

# Get runs from the experiment
runs = mlflow.search_runs(experiment_ids=[experiments[0].experiment_id])
print("\nRun Results:")
print(runs[['run_id', 'metrics.isolation_forest_precision', 'metrics.isolation_forest_recall', 'metrics.isolation_forest_f1_score']].head())
```

---

## Troubleshooting

### Common Issues and Solutions

**1. Module Import Errors**
```python
# If you get import errors, ensure the path is correct
import sys
sys.path.append('/content/VFSOC-ML-Models/src')

# Verify the module can be imported
try:
    from vfsoc_ml.data.data_loader import EnergyConsumptionDataLoader
    print("Import successful!")
except ImportError as e:
    print(f"Import error: {e}")
    print("Check that all files are uploaded correctly")
```

**2. Memory Issues**
```python
# If you run out of memory, reduce dataset size
config['simulation']['total_logs'] = 1000  # Reduce from 3100
config['model']['isolation_forest']['n_estimators'] = 50  # Reduce from 200

# Clear memory
import gc
gc.collect()
```

**3. Training Fails**
```python
# Check data quality
data = pd.read_csv('data/raw/synthetic_energy_data.csv')
print("Data Info:")
print(data.info())
print("\nMissing values:")
print(data.isnull().sum())
print("\nData types:")
print(data.dtypes)
```

**4. File Not Found Errors**
```python
# Check current directory and file structure
import os
print("Current directory:", os.getcwd())
print("\nDirectory contents:")
for root, dirs, files in os.walk('.'):
    level = root.replace('.', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        print(f"{subindent}{file}")
```

---

## Next Steps

After successful training in Colab:

1. **Download all results** and transfer to your local environment
2. **Deploy the ONNX model** to your production environment
3. **Set up monitoring** for the deployed model
4. **Prepare for Phase 2** (Ignition Behavior Detection) using the same process

### Production Deployment

The trained model can be deployed using:
- **REST API**: Use the ONNX model in a web service
- **Batch Processing**: Process large datasets offline
- **Real-time Streaming**: Integrate with data pipelines
- **Edge Deployment**: Deploy on edge devices using ONNX Runtime

### Model Monitoring

Set up monitoring for:
- **Data Drift**: Changes in input data distribution
- **Performance Degradation**: Model accuracy over time
- **Alert Volume**: Number of anomalies detected
- **False Positive Rate**: User feedback on alerts

---

## Conclusion

This guide provides a complete workflow for training VFSOC ML models in Google Colab. The trained models are production-ready and can be deployed in various environments using the provided ONNX exports.

For questions or issues:
- Check the troubleshooting section
- Review the main README.md for platform overview
- Consult the configuration files for parameter details

Happy training! 🚀 