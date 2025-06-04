#!/usr/bin/env python3
"""
Complete ML Workflow for GPS Jamming Detection.

This script orchestrates the entire machine learning pipeline:
1. Generate large-scale synthetic dataset (50,000 samples)
2. Train multiple advanced models with hyperparameter optimization
3. Perform comprehensive global accuracy testing
4. Select and save the best performing model
"""

import sys
import argparse
import logging
import subprocess
from pathlib import Path
import time
import json
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


class MLWorkflowOrchestrator:
    """
    Complete ML workflow orchestrator for GPS jamming detection.
    """
    
    def __init__(self, base_dir: str, verbose: bool = False):
        """
        Initialize the workflow orchestrator.
        
        Args:
            base_dir: Base directory for all operations
            verbose: Enable verbose logging
        """
        self.base_dir = Path(base_dir)
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        # Setup directories
        self.data_dir = self.base_dir / "data" / "large_synthetic"
        self.models_dir = self.base_dir / "models" / "advanced_trained"
        self.results_dir = self.base_dir / "results"
        self.scripts_dir = self.base_dir / "scripts"
        
        # Workflow configuration
        self.workflow_config = {
            'data_generation': {
                'samples': 50000,
                'jamming_ratio': 0.08,
                'batch_size': 5000,
                'enhanced_features': True,
                'quality_enhancements': True
            },
            'model_training': {
                'models': ['isolation_forest', 'xgboost', 'lightgbm', 'random_forest', 'neural_network'],
                'cv_folds': 5,
                'test_size': 0.2,
                'val_size': 0.1
            },
            'accuracy_testing': {
                'cv_folds': 10,
                'cv_repeats': 3,
                'confidence': 0.95
            }
        }
        
        # Execution tracking
        self.execution_log = {
            'workflow_start_time': None,
            'workflow_end_time': None,
            'steps_completed': [],
            'steps_failed': [],
            'final_results': {}
        }
    
    def run_step(self, step_name: str, command: List[str], check_output: bool = False) -> bool:
        """
        Execute a workflow step.
        
        Args:
            step_name: Name of the step for logging
            command: Command to execute
            check_output: Whether to check for specific output files
            
        Returns:
            True if step succeeded, False otherwise
        """
        self.logger.info(f"=== Starting Step: {step_name} ===")
        step_start_time = time.time()
        
        try:
            # Log the command being executed
            command_str = ' '.join(command)
            self.logger.info(f"Executing: {command_str}")
            
            # Execute the command
            result = subprocess.run(
                command,
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Log output if verbose
            if self.verbose and result.stdout:
                self.logger.info(f"STDOUT:\n{result.stdout}")
            
            if result.stderr:
                self.logger.warning(f"STDERR:\n{result.stderr}")
            
            step_duration = time.time() - step_start_time
            self.logger.info(f"✓ {step_name} completed successfully in {step_duration:.2f} seconds")
            
            # Track completion
            self.execution_log['steps_completed'].append({
                'step': step_name,
                'duration': step_duration,
                'command': command_str
            })
            
            return True
            
        except subprocess.CalledProcessError as e:
            step_duration = time.time() - step_start_time
            self.logger.error(f"✗ {step_name} failed after {step_duration:.2f} seconds")
            self.logger.error(f"Return code: {e.returncode}")
            self.logger.error(f"STDOUT: {e.stdout}")
            self.logger.error(f"STDERR: {e.stderr}")
            
            # Track failure
            self.execution_log['steps_failed'].append({
                'step': step_name,
                'duration': step_duration,
                'error': str(e),
                'return_code': e.returncode
            })
            
            return False
            
        except Exception as e:
            step_duration = time.time() - step_start_time
            self.logger.error(f"✗ {step_name} failed with unexpected error after {step_duration:.2f} seconds: {e}")
            
            self.execution_log['steps_failed'].append({
                'step': step_name,
                'duration': step_duration,
                'error': str(e)
            })
            
            return False
    
    def step_1_generate_dataset(self) -> bool:
        """Step 1: Generate large-scale synthetic dataset."""
        config = self.workflow_config['data_generation']
        
        command = [
            'python', 'scripts/generate_large_dataset.py',
            '--samples', str(config['samples']),
            '--jamming-ratio', str(config['jamming_ratio']),
            '--batch-size', str(config['batch_size']),
            '--output', str(self.data_dir),
            '--enhanced-features',
            '--quality-enhancements'
        ]
        
        if self.verbose:
            command.append('--verbose')
        
        success = self.run_step("Dataset Generation", command)
        
        # Verify output files exist
        if success:
            required_files = ['features.parquet', 'labels.npy', 'metadata.json']
            for file_name in required_files:
                file_path = self.data_dir / file_name
                if not file_path.exists():
                    self.logger.error(f"Expected output file not found: {file_path}")
                    return False
            
            # Log dataset statistics
            try:
                metadata_path = self.data_dir / "metadata.json"
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.logger.info(f"Dataset generated successfully:")
                self.logger.info(f"  - Total samples: {metadata['total_samples']}")
                self.logger.info(f"  - Features: {metadata['num_features']}")
                self.logger.info(f"  - Jamming samples: {metadata['jamming_samples']}")
                self.logger.info(f"  - Normal samples: {metadata['normal_samples']}")
                self.logger.info(f"  - Jamming ratio: {metadata['jamming_ratio']:.3f}")
                
                self.execution_log['final_results']['dataset_metadata'] = metadata
                
            except Exception as e:
                self.logger.warning(f"Could not read dataset metadata: {e}")
        
        return success
    
    def step_2_train_models(self) -> bool:
        """Step 2: Train multiple advanced models."""
        config = self.workflow_config['model_training']
        
        command = [
            'python', 'scripts/train_advanced_models.py',
            '--data', str(self.data_dir),
            '--output', str(self.models_dir),
            '--models'] + config['models'] + [
            '--cv-folds', str(config['cv_folds']),
            '--test-size', str(config['test_size']),
            '--val-size', str(config['val_size'])
        ]
        
        if self.verbose:
            command.append('--verbose')
        
        success = self.run_step("Model Training", command)
        
        # Verify trained models exist
        if success:
            model_files = list(self.models_dir.glob("*_model.pkl"))
            if not model_files:
                self.logger.error("No trained model files found")
                return False
            
            self.logger.info(f"Successfully trained {len(model_files)} models:")
            for model_file in model_files:
                self.logger.info(f"  - {model_file.name}")
            
            # Load training summary if available
            try:
                summary_path = self.models_dir / "training_summary.json"
                if summary_path.exists():
                    with open(summary_path, 'r') as f:
                        training_summary = json.load(f)
                    
                    self.execution_log['final_results']['training_summary'] = training_summary
                    
                    if 'best_model' in training_summary:
                        best_model = training_summary['best_model']
                        self.logger.info(f"Best training model: {best_model['name']} (F1: {best_model['f1_score']:.4f})")
                
            except Exception as e:
                self.logger.warning(f"Could not read training summary: {e}")
        
        return success
    
    def step_3_global_accuracy_testing(self) -> bool:
        """Step 3: Perform comprehensive global accuracy testing."""
        config = self.workflow_config['accuracy_testing']
        accuracy_results_dir = self.results_dir / "accuracy_testing"
        
        command = [
            'python', 'scripts/global_accuracy_testing.py',
            '--models', str(self.models_dir),
            '--data', str(self.data_dir),
            '--output', str(accuracy_results_dir),
            '--cv-folds', str(config['cv_folds']),
            '--cv-repeats', str(config['cv_repeats']),
            '--confidence', str(config['confidence'])
        ]
        
        if self.verbose:
            command.append('--verbose')
        
        success = self.run_step("Global Accuracy Testing", command)
        
        # Process testing results
        if success:
            try:
                report_path = accuracy_results_dir / "global_accuracy_report.json"
                if report_path.exists():
                    with open(report_path, 'r') as f:
                        accuracy_report = json.load(f)
                    
                    self.execution_log['final_results']['accuracy_report'] = accuracy_report
                    
                    # Extract and log key results
                    if 'statistical_analysis' in accuracy_report and 'model_ranking' in accuracy_report['statistical_analysis']:
                        ranking = accuracy_report['statistical_analysis']['model_ranking']
                        
                        self.logger.info("=== Final Model Ranking ===")
                        for rank_info in ranking[:3]:  # Top 3 models
                            self.logger.info(f"{rank_info['rank']}. {rank_info['model']}: "
                                           f"F1 = {rank_info['f1_score']:.4f} "
                                           f"CI = [{rank_info['confidence_interval'][0]:.4f}, {rank_info['confidence_interval'][1]:.4f}]")
                        
                        # Save best model info
                        if ranking:
                            best_model = ranking[0]
                            self.execution_log['final_results']['best_model'] = best_model
                            self.logger.info(f"\n🏆 BEST MODEL: {best_model['model']} with F1 Score: {best_model['f1_score']:.4f}")
                
                # Check for summary CSV
                summary_csv_path = accuracy_results_dir / "model_comparison_summary.csv"
                if summary_csv_path.exists():
                    self.logger.info(f"Model comparison summary saved to: {summary_csv_path}")
                
            except Exception as e:
                self.logger.warning(f"Could not process accuracy testing results: {e}")
        
        return success
    
    def step_4_finalize_workflow(self) -> bool:
        """Step 4: Finalize workflow and generate summary."""
        self.logger.info("=== Finalizing Workflow ===")
        
        try:
            # Create final summary
            workflow_summary = {
                'workflow_metadata': {
                    'total_duration': time.time() - self.execution_log['workflow_start_time'],
                    'steps_completed': len(self.execution_log['steps_completed']),
                    'steps_failed': len(self.execution_log['steps_failed']),
                    'success_rate': len(self.execution_log['steps_completed']) / (len(self.execution_log['steps_completed']) + len(self.execution_log['steps_failed'])) if (len(self.execution_log['steps_completed']) + len(self.execution_log['steps_failed'])) > 0 else 0
                },
                'execution_log': self.execution_log,
                'workflow_config': self.workflow_config
            }
            
            # Save workflow summary
            summary_path = self.results_dir / "complete_workflow_summary.json"
            self.results_dir.mkdir(parents=True, exist_ok=True)
            
            with open(summary_path, 'w') as f:
                json.dump(workflow_summary, f, indent=2, default=str)
            
            self.logger.info(f"Workflow summary saved to: {summary_path}")
            
            # Generate final report
            self.logger.info("\n" + "="*80)
            self.logger.info("COMPLETE WORKFLOW SUMMARY")
            self.logger.info("="*80)
            
            duration = workflow_summary['workflow_metadata']['total_duration']
            self.logger.info(f"Total Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
            self.logger.info(f"Steps Completed: {workflow_summary['workflow_metadata']['steps_completed']}")
            self.logger.info(f"Steps Failed: {workflow_summary['workflow_metadata']['steps_failed']}")
            self.logger.info(f"Success Rate: {workflow_summary['workflow_metadata']['success_rate']:.2%}")
            
            # Dataset info
            if 'dataset_metadata' in self.execution_log['final_results']:
                metadata = self.execution_log['final_results']['dataset_metadata']
                self.logger.info(f"\nDataset: {metadata['total_samples']} samples, {metadata['num_features']} features")
                self.logger.info(f"Jamming Ratio: {metadata['jamming_ratio']:.1%}")
            
            # Best model info
            if 'best_model' in self.execution_log['final_results']:
                best = self.execution_log['final_results']['best_model']
                self.logger.info(f"\nBest Model: {best['model']}")
                self.logger.info(f"F1 Score: {best['f1_score']:.4f}")
                
                # Copy best model to final location
                best_model_src = self.models_dir / f"{best['model']}_model.pkl"
                best_model_dst = self.results_dir / "production_model.pkl"
                
                if best_model_src.exists():
                    import shutil
                    shutil.copy2(best_model_src, best_model_dst)
                    self.logger.info(f"Production model saved to: {best_model_dst}")
            
            self.logger.info("="*80)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to finalize workflow: {e}")
            return False
    
    def run_complete_workflow(self) -> bool:
        """Run the complete ML workflow."""
        self.logger.info("🚀 Starting Complete ML Workflow for GPS Jamming Detection")
        self.execution_log['workflow_start_time'] = time.time()
        
        workflow_steps = [
            ("Generate Dataset", self.step_1_generate_dataset),
            ("Train Models", self.step_2_train_models),
            ("Global Accuracy Testing", self.step_3_global_accuracy_testing),
            ("Finalize Workflow", self.step_4_finalize_workflow)
        ]
        
        for step_name, step_function in workflow_steps:
            success = step_function()
            if not success:
                self.logger.error(f"Workflow failed at step: {step_name}")
                self.execution_log['workflow_end_time'] = time.time()
                return False
        
        self.execution_log['workflow_end_time'] = time.time()
        self.logger.info("🎉 Complete ML Workflow finished successfully!")
        return True


def main():
    """Main function for complete workflow execution."""
    parser = argparse.ArgumentParser(description="Complete ML workflow for GPS jamming detection")
    parser.add_argument("--base-dir", "-b", type=str, default=".",
                       help="Base directory for all operations")
    parser.add_argument("--samples", type=int, default=50000,
                       help="Number of samples to generate")
    parser.add_argument("--jamming-ratio", type=float, default=0.08,
                       help="Ratio of jamming events")
    parser.add_argument("--models", nargs='+', 
                       default=['isolation_forest', 'xgboost', 'lightgbm', 'random_forest'],
                       help="Models to train")
    parser.add_argument("--skip-data-generation", action="store_true",
                       help="Skip data generation step (use existing data)")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip model training step (use existing models)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("=== Complete ML Workflow for GPS Jamming Detection ===")
    logger.info(f"Base directory: {args.base_dir}")
    logger.info(f"Target samples: {args.samples}")
    logger.info(f"Jamming ratio: {args.jamming_ratio}")
    logger.info(f"Models to train: {args.models}")
    
    try:
        # Initialize orchestrator
        orchestrator = MLWorkflowOrchestrator(args.base_dir, args.verbose)
        
        # Update configuration with command line args
        orchestrator.workflow_config['data_generation']['samples'] = args.samples
        orchestrator.workflow_config['data_generation']['jamming_ratio'] = args.jamming_ratio
        orchestrator.workflow_config['model_training']['models'] = args.models
        
        # Handle skip options
        if args.skip_data_generation:
            logger.info("Skipping data generation step")
            orchestrator.step_1_generate_dataset = lambda: True
        
        if args.skip_training:
            logger.info("Skipping model training step")
            orchestrator.step_2_train_models = lambda: True
        
        # Run complete workflow
        success = orchestrator.run_complete_workflow()
        
        if success:
            logger.info("Complete workflow executed successfully!")
            return 0
        else:
            logger.error("Workflow execution failed!")
            return 1
            
    except Exception as e:
        logger.error(f"Workflow execution failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main()) 