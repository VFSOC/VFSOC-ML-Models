#!/usr/bin/env python3
"""
Generate Large-Scale Synthetic GPS Jamming Detection Dataset.

This script generates 50,000 high-quality synthetic samples for training
advanced GPS jamming detection models with enhanced feature engineering.
"""

import sys
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import time
from typing import Dict, List, Tuple, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vfsoc_ml.data.synthetic_generator import SyntheticDataGenerator, DataGenerationConfig
from vfsoc_ml.preprocessing.enhanced_feature_extractor import EnhancedFeatureExtractor


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


class LargeScaleDataGenerator:
    """
    Large-scale synthetic data generator for GPS jamming detection.
    
    Generates 50,000+ samples with enhanced features and balanced classes.
    """
    
    def __init__(self, 
                 total_samples: int = 50000,
                 jamming_ratio: float = 0.08,  # 8% jamming events for better balance
                 batch_size: int = 5000,
                 enhanced_features: bool = True):
        """
        Initialize large-scale data generator.
        
        Args:
            total_samples: Total number of samples to generate
            jamming_ratio: Proportion of jamming samples
            batch_size: Batch size for generation
            enhanced_features: Whether to use enhanced feature extraction
        """
        self.total_samples = total_samples
        self.jamming_ratio = jamming_ratio
        self.batch_size = batch_size
        self.enhanced_features = enhanced_features
        
        # Calculate expected class distribution
        self.expected_jamming = int(total_samples * jamming_ratio)
        self.expected_normal = total_samples - self.expected_jamming
        
        logger = logging.getLogger(__name__)
        logger.info(f"Configured to generate {total_samples} samples:")
        logger.info(f"  - Normal samples: {self.expected_normal} ({(1-jamming_ratio)*100:.1f}%)")
        logger.info(f"  - Jamming samples: {self.expected_jamming} ({jamming_ratio*100:.1f}%)")
        
    def generate_dataset(self, output_dir: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate the complete large-scale dataset.
        
        Args:
            output_dir: Directory to save the generated data
            
        Returns:
            Tuple of (features_dataframe, labels_array)
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Starting generation of {self.total_samples} samples...")
        
        start_time = time.time()
        
        # Calculate batches
        num_batches = int(np.ceil(self.total_samples / self.batch_size))
        
        all_features = []
        all_labels = []
        
        # Generate data in batches to manage memory
        for batch_idx in range(num_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min((batch_idx + 1) * self.batch_size, self.total_samples)
            current_batch_size = batch_end - batch_start
            
            logger.info(f"Generating batch {batch_idx + 1}/{num_batches} "
                       f"({current_batch_size} samples)...")
            
            # Generate batch
            batch_features, batch_labels = self._generate_batch(current_batch_size, batch_idx)
            
            all_features.append(batch_features)
            all_labels.append(batch_labels)
            
            # Log progress
            total_generated = batch_end
            progress = (total_generated / self.total_samples) * 100
            logger.info(f"Progress: {total_generated}/{self.total_samples} ({progress:.1f}%)")
        
        # Combine all batches
        logger.info("Combining batches...")
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_labels = np.concatenate(all_labels)
        
        # Verify class distribution
        jamming_count = np.sum(combined_labels == -1)
        normal_count = np.sum(combined_labels == 1)
        actual_jamming_ratio = jamming_count / len(combined_labels)
        
        logger.info(f"Final dataset statistics:")
        logger.info(f"  - Total samples: {len(combined_features)}")
        logger.info(f"  - Normal samples: {normal_count} ({(normal_count/len(combined_labels))*100:.1f}%)")
        logger.info(f"  - Jamming samples: {jamming_count} ({actual_jamming_ratio*100:.1f}%)")
        logger.info(f"  - Features: {combined_features.shape[1]}")
        
        # Save dataset
        self._save_dataset(combined_features, combined_labels, output_dir)
        
        generation_time = time.time() - start_time
        logger.info(f"Dataset generation completed in {generation_time:.2f} seconds")
        
        return combined_features, combined_labels
    
    def _generate_batch(self, batch_size: int, batch_idx: int) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate a single batch of data."""
        logger = logging.getLogger(__name__)
        
        # Create configuration for this batch
        config = DataGenerationConfig(
            total_logs=batch_size,
            jamming_ratio=self._adjust_jamming_ratio_for_batch(batch_idx),
            add_noise=True,
            noise_level=0.1,
            include_labels=True
        )
        
        # Generate basic data using existing generator
        generator = SyntheticDataGenerator(config)
        basic_features, labels = generator.generate_synthetic_data()
        
        # Apply enhanced feature extraction if requested
        if self.enhanced_features:
            features = self._apply_enhanced_features(basic_features)
        else:
            features = basic_features
        
        return features, labels
    
    def _adjust_jamming_ratio_for_batch(self, batch_idx: int) -> float:
        """Adjust jamming ratio for each batch to ensure overall balance."""
        # Add some variance to make data more realistic
        base_ratio = self.jamming_ratio
        variance = 0.02  # ±2% variance
        
        # Add deterministic variance based on batch index
        adjustment = (batch_idx % 10 - 5) * variance / 5
        adjusted_ratio = max(0.01, min(0.15, base_ratio + adjustment))
        
        return adjusted_ratio
    
    def _apply_enhanced_features(self, basic_features: pd.DataFrame) -> pd.DataFrame:
        """Apply enhanced feature extraction to basic features."""
        logger = logging.getLogger(__name__)
        
        # Convert DataFrame to vehilog format for enhanced extraction
        vehilogs = []
        for _, row in basic_features.iterrows():
            vehilog = {
                'latitude': row.get('latitude', 0),
                'longitude': row.get('longitude', 0),
                'speed': row.get('speed', 0),
                'bearing': row.get('bearing', 0),
                'altitude': row.get('altitude', 0),
                'gps_fix_quality': row.get('gps_fix_quality', 0),
                'satellite_count': row.get('satellite_count', 0),
                'device_connection_status': 'connected' if row.get('device_connection_status', 1) else 'disconnected',
                'ignition_status': 'on' if row.get('ignition_status', 1) else 'off',
                'engine_hours': row.get('engine_hours', 0),
                'odometer': row.get('odometer', 0),
                'fuel_level': row.get('fuel_level', 0),
                'timestamp': pd.Timestamp.now() + pd.Timedelta(hours=np.random.uniform(-24, 24)),
                'driver_authenticated': np.random.choice([True, False], p=[0.8, 0.2]),
                'driver_present': np.random.choice([True, False], p=[0.85, 0.15]),
                'security_alert': np.random.choice([True, False], p=[0.1, 0.9])
            }
            vehilogs.append(vehilog)
        
        # Apply enhanced feature extraction
        extractor = EnhancedFeatureExtractor(
            statistical_features=True,
            signal_processing_features=True,
            temporal_features=True,
            contextual_features=True
        )
        
        enhanced_features, _ = extractor.fit_transform(vehilogs)
        
        return enhanced_features
    
    def _save_dataset(self, features: pd.DataFrame, labels: np.ndarray, output_dir: str):
        """Save the generated dataset to disk."""
        logger = logging.getLogger(__name__)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save features
        features_path = output_path / "features.parquet"
        features.to_parquet(features_path, compression='snappy')
        logger.info(f"Features saved to {features_path}")
        
        # Save labels
        labels_path = output_path / "labels.npy"
        np.save(labels_path, labels)
        logger.info(f"Labels saved to {labels_path}")
        
        # Save metadata
        metadata = {
            'total_samples': len(features),
            'num_features': features.shape[1],
            'jamming_samples': int(np.sum(labels == -1)),
            'normal_samples': int(np.sum(labels == 1)),
            'jamming_ratio': float(np.sum(labels == -1) / len(labels)),
            'feature_names': features.columns.tolist(),
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'enhanced_features': self.enhanced_features
        }
        
        import json
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")
        
        # Save a CSV sample for inspection
        sample_size = min(1000, len(features))
        sample_indices = np.random.choice(len(features), sample_size, replace=False)
        
        sample_features = features.iloc[sample_indices]
        sample_labels = labels[sample_indices]
        
        sample_df = sample_features.copy()
        sample_df['label'] = sample_labels
        
        sample_path = output_path / "sample_data.csv"
        sample_df.to_csv(sample_path, index=False)
        logger.info(f"Sample data saved to {sample_path}")


def add_data_quality_enhancements(features: pd.DataFrame, labels: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    """Add data quality enhancements to make the dataset more realistic."""
    logger = logging.getLogger(__name__)
    logger.info("Applying data quality enhancements...")
    
    # Add realistic noise patterns
    features = features.copy()
    
    # Add correlated noise to GPS coordinates
    for coord in ['latitude', 'longitude']:
        if coord in features.columns:
            noise = np.random.normal(0, 0.001, len(features))  # Small GPS noise
            features[coord] += noise
    
    # Add temporal correlations for jamming events
    jamming_indices = np.where(labels == -1)[0]
    if len(jamming_indices) > 0:
        # Reduce satellite count for jamming events
        if 'satellite_count' in features.columns:
            features.loc[jamming_indices, 'satellite_count'] *= np.random.uniform(0.1, 0.6, len(jamming_indices))
        
        # Reduce GPS quality for jamming events
        if 'gps_fix_quality' in features.columns:
            features.loc[jamming_indices, 'gps_fix_quality'] *= np.random.uniform(0.1, 0.5, len(jamming_indices))
    
    # Add seasonal patterns
    if 'hour_of_day' in features.columns:
        # More jamming events during night hours
        night_mask = (features['hour_of_day'] < 6) | (features['hour_of_day'] > 22)
        night_indices = features[night_mask].index
        
        # Increase jamming probability at night (simulate criminal activity patterns)
        for idx in night_indices:
            if np.random.random() < 0.03:  # 3% chance to flip to jamming
                labels[idx] = -1
    
    logger.info("Data quality enhancements applied")
    return features, labels


def main():
    """Main function to generate large-scale dataset."""
    parser = argparse.ArgumentParser(description="Generate large-scale GPS jamming dataset")
    parser.add_argument("--samples", "-n", type=int, default=50000,
                       help="Total number of samples to generate")
    parser.add_argument("--jamming-ratio", "-j", type=float, default=0.08,
                       help="Ratio of jamming events (default: 0.08 = 8 percent)")
    parser.add_argument("--batch-size", "-b", type=int, default=5000,
                       help="Batch size for generation")
    parser.add_argument("--output", "-o", type=str, default="data/large_synthetic",
                       help="Output directory for generated data")
    parser.add_argument("--enhanced-features", action="store_true", default=True,
                       help="Use enhanced feature extraction")
    parser.add_argument("--quality-enhancements", action="store_true", default=True,
                       help="Apply data quality enhancements")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("=== Large-Scale GPS Jamming Dataset Generation ===")
    logger.info(f"Target samples: {args.samples}")
    logger.info(f"Jamming ratio: {args.jamming_ratio}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Enhanced features: {args.enhanced_features}")
    logger.info(f"Quality enhancements: {args.quality_enhancements}")
    
    try:
        # Initialize generator
        generator = LargeScaleDataGenerator(
            total_samples=args.samples,
            jamming_ratio=args.jamming_ratio,
            batch_size=args.batch_size,
            enhanced_features=args.enhanced_features
        )
        
        # Generate dataset
        features, labels = generator.generate_dataset(args.output)
        
        # Apply quality enhancements if requested
        if args.quality_enhancements:
            features, labels = add_data_quality_enhancements(features, labels)
        
        logger.info("=== Generation Summary ===")
        logger.info(f"✓ Successfully generated {len(features)} samples")
        logger.info(f"✓ Features: {features.shape[1]}")
        logger.info(f"✓ Normal samples: {np.sum(labels == 1)}")
        logger.info(f"✓ Jamming samples: {np.sum(labels == -1)}")
        logger.info(f"✓ Data saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main()) 