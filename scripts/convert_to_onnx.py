#!/usr/bin/env python3
"""
Convert trained models to ONNX format.

This script converts trained GPS jamming detection models to ONNX format
for production deployment and optimized inference.
"""

import sys
import argparse
import logging
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vfsoc_ml.deployment.onnx_converter import ONNXConverter


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main function to convert models to ONNX."""
    parser = argparse.ArgumentParser(description="Convert trained models to ONNX format")
    parser.add_argument("--model-path", "-m", type=str, required=True,
                       help="Path to the trained model (.pkl file)")
    parser.add_argument("--output", "-o", type=str, 
                       help="Output path for ONNX model (default: same dir as input)")
    parser.add_argument("--input-shape", "-s", type=int, nargs='+',
                       help="Input shape (number of features)")
    parser.add_argument("--model-name", "-n", type=str, default="GPSJammingDetector",
                       help="Name for the ONNX model")
    parser.add_argument("--benchmark", "-b", action="store_true",
                       help="Run benchmarks comparing original vs ONNX model")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    model_path = Path(args.model_path)
    
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return 1
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = model_path.parent / f"{model_path.stem}.onnx"
    
    logger.info("Starting ONNX conversion...")
    logger.info(f"Input model: {model_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Model name: {args.model_name}")
    
    try:
        # Initialize converter
        converter = ONNXConverter()
        
        # Convert model
        input_shape = tuple(args.input_shape) if args.input_shape else None
        onnx_path = converter.convert_model(
            model_path=model_path,
            output_path=output_path,
            input_shape=input_shape,
            model_name=args.model_name
        )
        
        logger.info(f"Model converted successfully to {onnx_path}")
        
        # Get model info
        model_info = converter.get_model_info(onnx_path)
        logger.info(f"ONNX model info: {model_info}")
        
        # Run benchmarks if requested
        if args.benchmark:
            logger.info("Running benchmarks...")
            
            benchmark_results = converter.benchmark_model(
                onnx_path=onnx_path,
                original_model_path=model_path,
                n_samples=1000
            )
            
            logger.info("Benchmark Results:")
            logger.info(f"  Original model time: {benchmark_results['original_time_sec']:.4f} seconds")
            logger.info(f"  ONNX model time: {benchmark_results['onnx_time_sec']:.4f} seconds")
            logger.info(f"  Speedup: {benchmark_results['speedup']:.2f}x")
            logger.info(f"  Prediction accuracy: {benchmark_results['prediction_accuracy']:.4f}")
            logger.info(f"  Original throughput: {benchmark_results['original_throughput']:.0f} samples/sec")
            logger.info(f"  ONNX throughput: {benchmark_results['onnx_throughput']:.0f} samples/sec")
            
            # Save benchmark results
            benchmark_path = output_path.parent / f"{output_path.stem}_benchmark.json"
            with open(benchmark_path, 'w') as f:
                json.dump(benchmark_results, f, indent=2)
            logger.info(f"Benchmark results saved to {benchmark_path}")
        
        # Save conversion metadata
        metadata = {
            "original_model_path": str(model_path),
            "onnx_model_path": str(onnx_path),
            "model_name": args.model_name,
            "input_shape": input_shape,
            "model_info": model_info
        }
        
        metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Conversion metadata saved to {metadata_path}")
        
        logger.info("ONNX conversion completed successfully!")
        
    except Exception as e:
        logger.error(f"ONNX conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 