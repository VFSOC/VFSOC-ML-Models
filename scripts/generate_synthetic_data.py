#!/usr/bin/env python3
"""
Generate synthetic GPS jamming detection data.

This script uses the VFSOC Geotab connector to generate realistic
synthetic training data for GPS jamming detection models.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vfsoc_ml.data.synthetic_generator import SyntheticDataGenerator, DataGenerationConfig


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main function to generate synthetic data."""
    parser = argparse.ArgumentParser(description="Generate synthetic GPS jamming data")
    parser.add_argument("--output", "-o", type=str, default="data/synthetic",
                       help="Output directory for generated data")
    parser.add_argument("--total-logs", "-n", type=int, default=3150,
                       help="Total number of logs to generate")
    parser.add_argument("--jamming-ratio", "-j", type=float, default=0.05,
                       help="Ratio of jamming events (0.05 = 5%)")
    parser.add_argument("--format", "-f", type=str, default="dataframe",
                       choices=["dataframe", "csv", "parquet"],
                       help="Output format")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting synthetic data generation...")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Total logs: {args.total_logs}")
    logger.info(f"Jamming ratio: {args.jamming_ratio}")
    logger.info(f"Output format: {args.format}")
    
    # Create configuration
    config = DataGenerationConfig(
        total_logs=args.total_logs,
        jamming_ratio=args.jamming_ratio,
        output_format=args.format
    )
    
    try:
        # Initialize generator
        generator = SyntheticDataGenerator(config)
        
        # Generate data
        features_df, labels = generator.generate_synthetic_data(save_path=args.output)
        
        logger.info("Data generation completed successfully!")
        logger.info(f"Generated {len(features_df)} samples with {features_df.shape[1]} features")
        
        # Print summary
        summary = generator.get_generation_summary()
        logger.info(f"Generation summary: {summary}")
        
    except Exception as e:
        logger.error(f"Data generation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 