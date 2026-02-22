#!/usr/bin/env python3
"""
Quick Start Example for TATO Framework

This example demonstrates the basic usage of TATO for time series forecasting
with adaptive transformation optimization.
"""

import sys
import os
import argparse
import logging
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import get_dataset, CustomDataset
from model.model_factory import ModelFactory
from transformation.transformation_factory import TransformationFactory
from pipeline.pipeline_factory import PipelineFactory
from tuner.tuner_factory import TunerFactory
from torch.utils.data import DataLoader
from utils.metrics import cal_metric_stat, set_seed
from utils.tools import visualize_samples, loss_plot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_quick_experiment(args):
    """Run a quick experiment with TATO."""
    
    logger.info("Starting TATO Quick Experiment")
    logger.info(f"Configuration: {vars(args)}")
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Step 1: Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    dataset = get_dataset(args.dataset, fast_split=args.fast_split)
    
    # Step 2: Load model
    logger.info(f"Loading model: {args.model}")
    model = ModelFactory.load_model(
        model_name=args.model,
        device=args.device,
        args=args
    )
    
    # Step 3: Create dataloader
    logger.info(f"Creating dataloader with {args.num_samples} samples")
    train_dataset = CustomDataset(
        dataset=dataset,
        mode='train',
        target_column='OT',
        max_seq_len=args.max_seq_len,
        pred_len=args.pred_len,
        augmentor=None,
        num_sample=args.num_samples
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # Step 4: Define transformation search space
    transformation_names = ['normalizer', 'sampler', 'trimmer']
    logger.info(f"Using transformations: {transformation_names}")
    
    # Step 5: Build search space
    distribution = TunerFactory.build_search_space(
        transformation_names,
        patch_len=args.patch_len
    )
    
    # Step 6: Get transformation factory
    transformation_dict = TransformationFactory.get_transformation_dict()
    
    # Step 7: Run a few trials to demonstrate the concept
    logger.info(f"Running {args.num_trials} search trials")
    
    trial_results = []
    for trial_idx in range(args.num_trials):
        logger.info(f"Trial {trial_idx + 1}/{args.num_trials}")
        
        # Sample transformation parameters
        params = {}
        for key in distribution.keys():
            if hasattr(distribution[key], 'sample'):
                params[key] = distribution[key].sample()
            else:
                params[key] = np.random.choice(distribution[key])
        
        # Build transformations
        transformations = []
        
        # Normalizer
        if params.get('normalizer_method', 'none') != 'none':
            transformations.append(
                transformation_dict['normalizer'](
                    method=params['normalizer_method'],
                    mode=params.get('normalizer_mode', 'input')
                )
            )
        
        # Sampler
        if params.get('sampler_factor', 1) != 1:
            transformations.append(
                transformation_dict['sampler'](
                    factor=params['sampler_factor']
                )
            )
        
        # Trimmer
        if params.get('trimmer_seq_l', 1) != 1:
            transformations.append(
                transformation_dict['trimmer'](
                    seq_l=params['trimmer_seq_l'],
                    patch_len=args.patch_len,
                    pred_len=args.pred_len
                )
            )
        
        # Step 8: Create pipeline
        config = {
            'patch_len': args.patch_len,
            'pred_len': args.pred_len,
            'data_patch_len': args.patch_len,
            'model_patch_len': args.patch_len
        }
        
        # Note: In the actual code, PipelineFactory uses build_trial_pipeline_by_transformation_names
        # For simplicity in this example, we'll create BasePipeline directly
        from pipeline.base import BasePipeline
        pipeline = BasePipeline(
            transformations=transformations,
            config=config,
            model=model,
            pred_len=args.pred_len,
            augmentor=None,
            mode='train'
        )
        
        # Step 9: Evaluate pipeline
        try:
            metrics = pipeline.evaluate(train_loader)
            avg_metrics = {k: float(np.mean(vals)) for k, vals in metrics.items()}
            
            trial_results.append({
                'trial_idx': trial_idx,
                'params': params,
                'metrics': avg_metrics,
                'transformations': [type(tf).__name__ for tf in transformations]
            })
            
            logger.info(f"Trial {trial_idx + 1} - MSE: {avg_metrics.get('MSE', 'N/A'):.6f}")
            
        except Exception as e:
            logger.warning(f"Trial {trial_idx + 1} failed: {str(e)}")
            continue
    
    # Step 10: Analyze results
    if trial_results:
        logger.info("\n" + "="*50)
        logger.info("Experiment Results Summary")
        logger.info("="*50)
        
        # Find best trial
        best_trial = min(trial_results, key=lambda x: x['metrics'].get('MSE', float('inf')))
        
        logger.info(f"Total trials completed: {len(trial_results)}")
        logger.info(f"Best trial index: {best_trial['trial_idx']}")
        logger.info(f"Best MSE: {best_trial['metrics'].get('MSE', 'N/A'):.6f}")
        logger.info(f"Best transformations: {best_trial['transformations']}")
        logger.info(f"Best parameters: {best_trial['params']}")
        
        # Calculate average metrics
        all_mse = [t['metrics'].get('MSE', float('inf')) for t in trial_results]
        valid_mse = [m for m in all_mse if m != float('inf')]
        
        if valid_mse:
            logger.info(f"Average MSE: {np.mean(valid_mse):.6f}")
            logger.info(f"Std MSE: {np.std(valid_mse):.6f}")
            logger.info(f"Min MSE: {np.min(valid_mse):.6f}")
            logger.info(f"Max MSE: {np.max(valid_mse):.6f}")
        
        return best_trial
    else:
        logger.error("No trials completed successfully")
        return None

def main():
    parser = argparse.ArgumentParser(description='TATO Quick Start Example')
    
    # Experiment configuration
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='ETTh1', 
                       choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 
                               'Exchange', 'Weather', 'Traffic', 'Electricity'],
                       help='Dataset to use')
    parser.add_argument('--fast_split', action='store_true', 
                       help='Use fast split for quick testing')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='Timer-LOTSA',
                       choices=['Timer-LOTSA', 'Timer-UTSD', 'MOIRAI-small',
                               'MOIRAI-base', 'Chronos-tiny', 'Sundial'],
                       help='Model to use')
    
    # Forecasting configuration
    parser.add_argument('--pred_len', type=int, default=96, 
                       help='Prediction length')
    parser.add_argument('--max_seq_len', type=int, default=1440,
                       help='Maximum sequence length')
    parser.add_argument('--patch_len', type=int, default=96,
                       help='Patch length')
    
    # Experiment parameters
    parser.add_argument('--num_trials', type=int, default=10,
                       help='Number of search trials')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of data samples')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    
    # Output configuration
    parser.add_argument('--save_results', action='store_true',
                       help='Save results to file')
    parser.add_argument('--output_dir', type=str, default='./quick_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiment
    result = run_quick_experiment(args)
    
    # Save results if requested
    if args.save_results and result:
        import json
        output_file = os.path.join(args.output_dir, 'quick_start_results.json')
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")
    
    logger.info("Quick start example completed!")

if __name__ == '__main__':
    main()
