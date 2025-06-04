"""
Data loader for VFSOC GPS jamming detection models.

This module handles loading and preprocessing of GPS jamming detection data
from various sources including synthetic data and real vehicle telemetry.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any
import logging
import json

logger = logging.getLogger(__name__)


class VFSOCDataLoader:
    """
    Data loader for GPS jamming detection datasets.
    
    Handles loading data from multiple formats and provides preprocessing
    capabilities for machine learning model training.
    """
    
    def __init__(self):
        """Initialize the data loader."""
        self.data = None
        self.labels = None
        self.feature_names = None
        self.metadata = None
        
    def load_synthetic_data(self, data_path: Union[str, Path]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load synthetic GPS jamming data.
        
        Args:
            data_path: Path to the synthetic data directory
            
        Returns:
            Tuple of (features_df, labels_array)
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
        
        # Try different file formats
        if (data_path / "features.pkl").exists():
            features_df = pd.read_pickle(data_path / "features.pkl")
            labels = np.load(data_path / "labels.npy")
        elif (data_path / "features.csv").exists():
            features_df = pd.read_csv(data_path / "features.csv")
            labels = np.loadtxt(data_path / "labels.csv", delimiter=",", dtype=int)
        elif (data_path / "features.parquet").exists():
            features_df = pd.read_parquet(data_path / "features.parquet")
            labels_df = pd.read_parquet(data_path / "labels.parquet")
            labels = labels_df["labels"].values
        else:
            raise FileNotFoundError("No supported data files found in the directory")
        
        # Load metadata if available
        metadata_path = data_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        self.data = features_df
        self.labels = labels
        self.feature_names = features_df.columns.tolist()
        
        logger.info(f"Loaded {len(features_df)} samples with {features_df.shape[1]} features")
        logger.info(f"Class distribution - Normal: {np.sum(labels == 1)}, Jamming: {np.sum(labels == -1)}")
        
        return features_df, labels
    
    def load_csv_data(self, features_path: Union[str, Path], 
                     labels_path: Optional[Union[str, Path]] = None) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """
        Load data from CSV files.
        
        Args:
            features_path: Path to features CSV file
            labels_path: Optional path to labels CSV file
            
        Returns:
            Tuple of (features_df, labels_array)
        """
        features_df = pd.read_csv(features_path)
        
        labels = None
        if labels_path:
            labels = np.loadtxt(labels_path, delimiter=",", dtype=int)
        
        self.data = features_df
        self.labels = labels
        self.feature_names = features_df.columns.tolist()
        
        logger.info(f"Loaded CSV data with {len(features_df)} samples and {features_df.shape[1]} features")
        
        return features_df, labels
    
    def get_train_test_split(self, test_size: float = 0.2, 
                           random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if self.data is None or self.labels is None:
            raise ValueError("No data loaded. Call load_* method first.")
        
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.data, self.labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.labels
        )
        
        logger.info(f"Split data - Train: {len(X_train)}, Test: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded features.
        
        Returns:
            Dictionary containing feature information
        """
        if self.data is None:
            return {}
        
        info = {
            "n_samples": len(self.data),
            "n_features": self.data.shape[1],
            "feature_names": self.feature_names,
            "feature_types": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "feature_statistics": self.data.describe().to_dict()
        }
        
        if self.labels is not None:
            info["label_distribution"] = {
                "normal": int(np.sum(self.labels == 1)),
                "jamming": int(np.sum(self.labels == -1)),
                "total": len(self.labels)
            }
        
        return info
    
    def preprocess_features(self, method: str = "robust") -> pd.DataFrame:
        """
        Preprocess features using various scaling methods.
        
        Args:
            method: Preprocessing method ('standard', 'robust', 'minmax')
            
        Returns:
            Preprocessed features DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_* method first.")
        
        from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
        
        if method == "standard":
            scaler = StandardScaler()
        elif method == "robust":
            scaler = RobustScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")
        
        # Only scale numeric columns
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        
        processed_data = self.data.copy()
        processed_data[numeric_columns] = scaler.fit_transform(self.data[numeric_columns])
        
        logger.info(f"Applied {method} scaling to {len(numeric_columns)} numeric features")
        
        return processed_data 