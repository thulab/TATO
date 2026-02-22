import sys
import os
import logging
import argparse
from typing import Dict, List, Any
import numpy as np
from multidict import MultiDict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import TATO modules
from pipeline.pipeline_factory import PipelineFactory
from tuner.tuner_factory import TunerFactory
from model.model_factory import ModelFactory
from data.dataset import *
from torch.utils.data import DataLoader
from utils.augmentor import Augmentor
from utils.metrics import cal_metric_stat, set_seed
from utils.tools import topk_by_wrank, pareto_select


def main(args: argparse.Namespace) -> None:
    """Main function for running TATO experiments.
    
    This function implements the complete TATO experimental pipeline:
    1. Training phase: Search for optimal transformation parameters
    2. Validation phase: Evaluate top-k configurations from training
    3. Testing phase: Final evaluation on test data
    
    Args:
        args: Command-line arguments containing experiment configuration
    """
    # Define available transformations
    transformation_names = ['trimmer', 'inputer', 'denoiser', 'warper', 
                           'differentiator', 'normalizer', 'sampler', 'aligner']

    # Default/original parameter configuration (no transformations)
    ori_param = {
        'inference_mode': 'infer1',
        'clip_factor': 'none',
        'trimmer_seq_l': 7,
        'inputer_detect_method': 'none',
        'inputer_fill_method': 'linear_interpolate',
        'denoiser_method': 'none',
        'warper_method': 'none',
        'differentiator_n': 0,
        'normalizer_method': 'none',
        'normalizer_mode': 'input',
        'sampler_factor': 1,
        'aligner_mode': 'none',
        'aligner_method': 'edge_pad'
    }

    # Extract configuration from arguments
    model_name = args.model
    data_name = args.dataset
    patch_len = args.patch_len
    pred_len = args.pred_len
    
    # Build search space for transformations
    distribution = TunerFactory.build_search_space(transformation_names, patch_len)
    
    # Load foundation model
    model = ModelFactory.load_model(model_name, device=args.device, args=args)
    
    # Load dataset
    dataset = get_dataset(data_name)
    
    # Configure pipeline parameters
    configs = {
        'patch_len': patch_len,
        'pred_len': pred_len,
        'data_patch_len': patch_len,
        'model_patch_len': patch_len
    }
    
    # Create save directory for results
    save_dir = f'{args.save_dir}/SD-{args.seed}-P-{args.train_trials}-S-{args.num_samples}-topk-{args.top_k}/{model_name}/{data_name}/pred_{pred_len}/'
    os.makedirs(save_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    set_seed(args.seed)

    # ==================== TRAINING PHASE ====================
    logging.info("Starting training phase (transformation search)")
    
    # Initialize tuner with original parameters
    tuner = TunerFactory.build_optuna_tuner(enqueue_param_dicts=[ori_param], seed=args.seed)
    
    # Create training dataset and dataloader
    custom_dataset = CustomDataset(dataset, 'train', 'OT', 1440, pred_len, None, args.num_samples)
    dataloader = DataLoader(custom_dataset, batch_size=512)
    
    # Get scaler for training data
    scaler_for_train_val = dataset.get_mode_scaler('train', 'standard', 'OT')
    
    # Initialize augmentor (if enabled)
    if args.no_aug:
        augmentor = None
    else:
        augmentor = Augmentor(aug_method='none', mode='rotate', pred_len=patch_len)
    
    # Dictionary to store training results
    train_results_dicts = {}
    train_trials = args.train_trials
    
    # Run training trials
    for i in range(train_trials):
        if augmentor:
            augmentor.reset_aug_method(aug_method='none')
        logging.info(f'Starting trial {i+1}/{train_trials}')
        
        # Pick a trial with transformation parameters
        trial = tuner.pick_trial(distribution, {})
        param_dict = trial.params
        s = str(param_dict)
        
        # Build pipeline with selected transformations
        pipeline = PipelineFactory.build_trial_pipeline_by_transformation_names(
            trial, model, transformation_names, configs, augmentor)
        
        # Evaluate pipeline
        metric = pipeline.evaluate(dataloader, scaler_for_train_val)
        avg_metrics = {k: float(np.mean(vals)) for k, vals in metric.items()}
        all_metric = cal_metric_stat(metric, ['mean', 'std', 'median', 'iqr', 'max', 'min'])
        
        # Report results to tuner
        tuner.tell(trial, avg_metrics['MSE'])
        train_results_dicts[s] = all_metric
    
    # Save training trials
    tuner.save_trials(f'{save_dir}train_trials.csv')
    
    # Select top-k configurations using Pareto front
    top_k = args.top_k
    metric_weight_dict = {'MSE_mean': 3, 'MAE_mean': 2, 'RMSE_mean': 1, 
                         'MAPE_mean': 1, 'MSPE_mean': 1}
    topk_param_list = pareto_select(train_results_dicts, metric_weight_dict, 
                                   top_k=top_k, save_dir=save_dir, mode='train', save=True)

    # ==================== VALIDATION PHASE ====================
    logging.info("Starting validation phase")
    
    # Ensure original parameters are included in validation
    val_unique_param_dict_list = topk_param_list if ori_param in topk_param_list else [ori_param] + topk_param_list
    
    # Initialize tuner for validation
    tuner = TunerFactory.build_optuna_tuner(enqueue_param_dicts=val_unique_param_dict_list, seed=args.seed)
    
    # Create validation dataset (using training data for validation in this setup)
    custom_dataset = CustomDataset(dataset, 'train', 'OT', 1440, pred_len, None, args.num_samples)
    dataloader = DataLoader(custom_dataset, batch_size=512)
    
    val_trials = len(val_unique_param_dict_list)
    val_results_dicts = {}
    
    # Run validation trials
    for i in range(val_trials):
        logging.info(f'Starting trial {i+1}/{val_trials}')
        trial = tuner.pick_trial(distribution)
        param_dict = trial.params
        s = str(param_dict)
        
        # Build pipeline with visualization enabled
        pipeline = PipelineFactory.build_trial_pipeline_by_transformation_names(
            trial, model, transformation_names, configs, None, 'val', i + 1, save_dir, args.plt)
        
        # Evaluate pipeline
        metric = pipeline.evaluate(dataloader, scaler_for_train_val)
        avg_metrics = {k: float(np.mean(vals)) for k, vals in metric.items()}
        all_metric = cal_metric_stat(metric, ['mean', 'std', 'median', 'iqr', 'max', 'min'])
        
        # Report results to tuner
        tuner.tell(trial, avg_metrics['MSE'])
        val_results_dicts[s] = all_metric
    
    # Save validation trials
    tuner.save_trials(f'{save_dir}val_trials.csv')
    
    # Select top parameters using weighted ranking
    metric_weight_dict = {'MSE_mean': 9, 'MAE_mean': 6, 'RMSE_mean': 3, 
                         'MSE_median': 6, 'MAE_median': 4, 'RMSE_median': 2}
    top_param_list = topk_by_wrank(val_results_dicts, metric_weight_dict, 
                                  top_k=val_trials, save_dir=save_dir, mode='val', save=True)

    # ==================== TESTING PHASE ====================
    logging.info("Starting testing phase")
    
    # Initialize tuner for testing (includes original and top parameters)
    tuner = TunerFactory.build_optuna_tuner(
        enqueue_param_dicts=[ori_param] + top_param_list, mode='test', seed=args.seed)
    
    # Create test dataset and dataloader (using all test samples)
    custom_dataset = CustomDataset(dataset, 'test', 'OT', 1440, pred_len, None, 'all')
    dataloader = DataLoader(custom_dataset, batch_size=512)
    
    # Get scaler for test data
    scaler_for_test = dataset.get_mode_scaler('test', 'standard', 'OT')
    
    # MultiDict allows duplicate keys (same parameter configuration can appear multiple times)
    test_results_dicts = MultiDict()
    
    # Run testing trials (includes original + top parameters from validation)
    for i in range(val_trials + 1):
        logging.info(f'Starting trial {i+1}/{val_trials+1}')
        trial = tuner.pick_trial(distribution)
        s = str(trial.params)
        
        # Build pipeline for testing with visualization
        pipeline = PipelineFactory.build_trial_pipeline_by_transformation_names(
            trial, model, transformation_names, configs, None, 'test', i + 1, save_dir, args.plt)
        
        # Evaluate pipeline on test data
        metric = pipeline.evaluate(dataloader, scaler_for_test)
        avg_metrics = {k: float(np.mean(vals)) for k, vals in metric.items()}
        all_metric = cal_metric_stat(metric, ['mean', 'std', 'median', 'iqr', 'max', 'min'])
        
        # Report results to tuner
        tuner.tell(trial, avg_metrics['MSE'])
        test_results_dicts.add(s, all_metric)
    
    # Save testing trials
    tuner.save_trials(f'{save_dir}test_trials.csv')
    
    # Select final top parameters from test results
    topk_param_list = topk_by_wrank(test_results_dicts, metric_weight_dict, 
                                   top_k=val_trials + 1, save_dir=save_dir, mode='test', save=True)
    
    logging.info("TATO experiment completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TATO: Adaptive Transformation Optimization for Time Series Foundation Models')
    
    # Experiment configuration
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for dataloaders')
    parser.add_argument('--device', type=str, default='cuda:7', help='GPU device (e.g., cuda:0, cpu)')
    parser.add_argument('--dataset', type=str, default='ETTh1', 
                       help='Dataset name (ETTh1, ETTh2, ETTm1, ETTm2, Exchange, Weather, Electricity, Traffic)')
    parser.add_argument('--model', type=str, default='Timer-LOTSA', 
                       help='Foundation model name (Timer-LOTSA, Timer-UTSD, MOIRAI-small, MOIRAI-base, MOIRAI-large, Chronos-tiny, Sundial)')
    parser.add_argument('--pred_len', type=int, default=96, help='Prediction length (forecast horizon)')
    parser.add_argument('--patch_len', type=int, default=96, help='Patch length for models')
    parser.add_argument('--train_trials', type=int, default=500, 
                       help='Number of search trials during training phase')
    parser.add_argument('--num_samples', type=int, default=500, 
                       help='Number of samples to use during search (for faster experimentation)')
    parser.add_argument('--top_k', type=int, default=16, 
                       help='Number of top-k configurations to select after training phase')
    parser.add_argument('--save_dir', type=str, default='./results', 
                       help='Base directory for saving results')
    parser.add_argument('--no_aug', action='store_true', default=False, 
                       help='Disable data augmentation during search')
    parser.add_argument('--plt', action='store_true', default=False, 
                       help='Enable visualization plots (saves sample visualizations)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run main experiment
    main(args)
