"""
Energy Consumption Data Loader

This module handles loading and preprocessing of EV charging station data
for irregular energy consumption pattern detection.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class EnergyConsumptionDataLoader:
    """
    Data loader for EV charging station energy consumption data.
    
    Handles loading, cleaning, and preprocessing of charging session data
    for anomaly detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data loader.
        
        Args:
            config: Configuration dictionary containing data paths and parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Data paths
        self.raw_data_path = config['data']['raw_data_path']
        self.synthetic_data_path = config['data']['synthetic_data_path']
        self.processed_data_path = config['data']['processed_data_path']
        
        # Preprocessing parameters
        self.preprocessing_config = config['data']['preprocessing']
        
    def load_station_data(self) -> pd.DataFrame:
        """
        Load the main EV charging station dataset.
        
        Returns:
            DataFrame containing charging session data
        """
        self.logger.info(f"Loading station data from: {self.raw_data_path}")
        
        try:
            # Load the CSV file
            df = pd.read_csv(self.raw_data_path)
            
            self.logger.info(f"Loaded {len(df)} charging sessions")
            self.logger.info(f"Columns: {list(df.columns)}")
            
            # Basic data validation
            self._validate_required_columns(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading station data: {str(e)}")
            raise
    
    def load_synthetic_data(self) -> pd.DataFrame:
        """
        Load synthetic EV charging data if available.
        
        Returns:
            DataFrame containing synthetic charging session data
        """
        self.logger.info(f"Loading synthetic data from: {self.synthetic_data_path}")
        
        try:
            # Load the synthetic CSV file
            df = pd.read_csv(self.synthetic_data_path)
            
            # Map synthetic data columns to match station data format
            df = self._map_synthetic_columns(df)
            
            self.logger.info(f"Loaded {len(df)} synthetic charging sessions")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading synthetic data: {str(e)}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the charging session data.
        
        Args:
            df: Raw charging session data
            
        Returns:
            Preprocessed DataFrame
        """
        self.logger.info("Starting data preprocessing...")
        
        # Make a copy to avoid modifying original data
        processed_df = df.copy()
        
        # Clean and validate data
        processed_df = self._clean_data(processed_df)
        
        # Handle missing values
        processed_df = self._handle_missing_values(processed_df)
        
        # Remove outliers
        if self.preprocessing_config['remove_outliers']:
            processed_df = self._remove_outliers(processed_df)
        
        # Parse datetime columns
        processed_df = self._parse_datetime_columns(processed_df)
        
        # Validate data ranges
        processed_df = self._validate_data_ranges(processed_df)
        
        self.logger.info(f"Preprocessing complete. Final dataset size: {len(processed_df)}")
        
        return processed_df
    
    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """Validate that required columns are present."""
        required_columns = [
            'sessionId', 'kwhTotal', 'chargeTimeHrs', 'stationId', 'userId'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    def _map_synthetic_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map synthetic data columns to match station data format."""
        # Map synthetic column names to standard format
        column_mapping = {
            'connectionTime_decimal': 'startTime',
            'chargingDuration': 'chargeTimeHrs',
            'kWhDelivered': 'kwhTotal',
            'dayIndicator': 'weekday'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Add missing columns with default values
        if 'sessionId' not in df.columns:
            df['sessionId'] = range(len(df))
        
        if 'stationId' not in df.columns:
            df['stationId'] = np.random.randint(1000, 9999, len(df))
        
        if 'userId' not in df.columns:
            df['userId'] = np.random.randint(10000, 99999, len(df))
        
        if 'dollars' not in df.columns:
            df['dollars'] = df['kwhTotal'] * 0.15  # Approximate cost
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the data by removing invalid entries."""
        initial_size = len(df)
        
        # Remove rows with negative or zero energy consumption
        df = df[df['kwhTotal'] > 0]
        
        # Remove rows with negative or zero charging time
        df = df[df['chargeTimeHrs'] > 0]
        
        # Remove duplicate sessions
        df = df.drop_duplicates(subset=['sessionId'], keep='first')
        
        removed_count = initial_size - len(df)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} invalid/duplicate records")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Check for missing values
        missing_counts = df.isnull().sum()
        
        if missing_counts.sum() > 0:
            self.logger.info(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")
            
            # Fill missing values based on column type
            for column in df.columns:
                if df[column].isnull().sum() > 0:
                    if df[column].dtype in ['int64', 'float64']:
                        # Fill numeric columns with median
                        df[column] = df[column].fillna(df[column].median())
                    else:
                        # Fill categorical columns with mode
                        df[column] = df[column].fillna(df[column].mode()[0])
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using statistical methods."""
        initial_size = len(df)
        threshold = self.preprocessing_config['outlier_threshold']
        
        # Define columns to check for outliers
        numeric_columns = ['kwhTotal', 'chargeTimeHrs']
        
        for column in numeric_columns:
            if column in df.columns:
                # Calculate z-scores
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                
                # Remove outliers
                df = df[z_scores <= threshold]
        
        removed_count = initial_size - len(df)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} outliers")
        
        return df
    
    def _parse_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse datetime columns and extract time features."""
        datetime_columns = ['created', 'ended', 'startTime', 'endTime']
        
        for column in datetime_columns:
            if column in df.columns:
                try:
                    # Try to parse datetime
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                    
                    # Extract time features
                    if column in ['created', 'startTime']:
                        df[f'{column}_hour'] = df[column].dt.hour
                        df[f'{column}_day_of_week'] = df[column].dt.dayofweek
                        df[f'{column}_month'] = df[column].dt.month
                        
                except Exception as e:
                    self.logger.warning(f"Could not parse datetime column {column}: {str(e)}")
        
        return df
    
    def _validate_data_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that data values are within reasonable ranges."""
        initial_size = len(df)
        
        # Energy consumption validation
        min_energy = self.preprocessing_config['min_energy_delivered']
        max_energy = self.preprocessing_config['max_energy_delivered']
        df = df[(df['kwhTotal'] >= min_energy) & (df['kwhTotal'] <= max_energy)]
        
        # Charging duration validation
        min_duration = self.preprocessing_config['min_session_duration']
        max_duration = self.preprocessing_config['max_session_duration']
        df = df[(df['chargeTimeHrs'] >= min_duration) & (df['chargeTimeHrs'] <= max_duration)]
        
        removed_count = initial_size - len(df)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} records outside valid ranges")
        
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics of the dataset.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary containing summary statistics
        """
        summary = {
            'total_sessions': len(df),
            'unique_stations': df['stationId'].nunique() if 'stationId' in df.columns else 0,
            'unique_users': df['userId'].nunique() if 'userId' in df.columns else 0,
            'total_energy_kwh': df['kwhTotal'].sum() if 'kwhTotal' in df.columns else 0,
            'avg_energy_per_session': df['kwhTotal'].mean() if 'kwhTotal' in df.columns else 0,
            'avg_session_duration': df['chargeTimeHrs'].mean() if 'chargeTimeHrs' in df.columns else 0,
            'date_range': {
                'start': df['created'].min() if 'created' in df.columns else None,
                'end': df['created'].max() if 'created' in df.columns else None
            }
        }
        
        return summary
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_charging_data.csv") -> str:
        """
        Save processed data to file.
        
        Args:
            df: Processed DataFrame
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        # Create output directory
        output_dir = Path(self.processed_data_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        output_path = output_dir / filename
        df.to_csv(output_path, index=False)
        
        self.logger.info(f"Processed data saved to: {output_path}")
        
        return str(output_path) 