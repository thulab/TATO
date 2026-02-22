#!/usr/bin/env python3
"""
TATO Tutorial: Basic Time Series Forecasting

This tutorial demonstrates the basic usage of TATO (Adaptive Transformation Optimization) 
for time series forecasting.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import TATO modules
from data.dataset import get_dataset, CustomDataset
from model.model_factory import ModelFactory
from transformation.transformation_factory import TransformationFactory
from pipeline.base import BasePipeline
from torch.utils.data import DataLoader
from utils.metrics import set_seed

def main():
    """Main tutorial function."""
    
    print("=" * 60)
    print("TATO Tutorial: Basic Time Series Forecasting")
    print("=" * 60)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Step 1: Load Dataset
    print("\nStep 1: Loading Dataset")
    print("-" * 40)
    
    dataset_name = 'ETTh1'
    dataset = get_dataset(dataset_name, fast_split=True)  # fast_split=True for quick testing
    
    print(f"Dataset: {dataset_name}")
    print(f"Training samples: {dataset.train_len}")
    print(f"Validation samples: {dataset.val_len}")
    print(f"Test samples: {dataset.test_len}")
    print(f"Feature dimension: {dataset.feature_dim}")
    
    # Step 2: Visualize Data
    print("\nStep 2: Visualizing Data")
    print("-" * 40)
    
    # Extract data for visualization
    train_data = dataset.np_data_dict['OT'][dataset.train_start:dataset.train_end]
    val_data = dataset.np_data_dict['OT'][dataset.val_start:dataset.val_end]
    test_data = dataset.np_data_dict['OT'][dataset.test_start:dataset.test_end]
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    axes[0].plot(train_data, label='Training Data', color='blue', alpha=0.7)
    axes[0].set_title('Training Set')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(val_data, label='Validation Data', color='green', alpha=0.7)
    axes[1].set_title('Validation Set')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(test_data, label='Test Data', color='red', alpha=0.7)
    axes[2].set_title('Test Set')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Value')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tutorial_data_visualization.png', dpi=150, bbox_inches='tight')
    print("Data visualization saved as 'tutorial_data_visualization.png'")
    
    # Step 3: Initialize Model
    print("\nStep 3: Initializing Model")
    print("-" * 40)
    
    model_name = 'Timer-LOTSA'
    
    # Check for GPU availability
    try:
        import torch
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        device = 'cpu'
    
    model = ModelFactory.load_model(
        model_name=model_name,
        device=device,
        args=None
    )
    
    print(f"Model loaded: {model_name}")
    print(f"Device: {device}")
    # Note: Some models may not have patch_len attribute
    if hasattr(model, 'patch_len'):
        print(f"Model patch length: {model.patch_len}")
    else:
        print("Model patch length: Not available (using default)")
    
    # Step 4: Create Data Loader
    print("\nStep 4: Creating Data Loader")
    print("-" * 40)
    
    pred_len = 96  # Predict next 96 time steps
    max_seq_len = 1440  # Use 1440 historical points
    batch_size = 32
    num_samples = 100  # Use 100 samples for quick demonstration
    
    train_dataset = CustomDataset(
        dataset=dataset,
        mode='train',
        target_column='OT',
        max_seq_len=max_seq_len,
        pred_len=pred_len,
        augmentor=None,
        num_sample=str(num_samples)  # Convert to string as expected by the API
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    print(f"Dataloader created with {len(train_dataset)} samples")
    print(f"Batch size: {batch_size}")
    
    # Inspect one batch
    for real_idx, history, label in train_loader:
        print(f"Batch history shape: {history.shape}")  # (batch_size, seq_len, 1)
        print(f"Batch label shape: {label.shape}")      # (batch_size, pred_len, 1)
        break
    
    # Step 5: Configure Transformations
    print("\nStep 5: Configuring Transformations")
    print("-" * 40)
    
    # Get available transformations
    transformation_dict = TransformationFactory.get_transformation_dict()
    print(f"Available transformations: {list(transformation_dict.keys())}")
    
    # Create transformation instances
    transformations = [
        transformation_dict['normalizer'](
            method='standard',
            mode='input'
        ),
        transformation_dict['trimmer'](
            seq_l=10,
            patch_len=96,
            pred_len=pred_len
        )
    ]
    
    print(f"Created {len(transformations)} transformations:")
    for tf in transformations:
        print(f"  - {type(tf).__name__}")
    
    # Step 6: Create and Test Pipeline
    print("\nStep 6: Creating and Testing Pipeline")
    print("-" * 40)
    
    config = {
        'patch_len': 96,
        'pred_len': pred_len,
        'data_patch_len': 96,
        'model_patch_len': 96
    }
    
    pipeline = BasePipeline(
        transformations=transformations,
        config=config,
        model=model,
        pred_len=pred_len,
        augmentor=None,
        mode='train'
    )
    
    # Test pipeline on a single batch
    for real_idx, history, label in train_loader:
        # Convert to numpy for pipeline
        history_np = history.numpy()
        label_np = label.numpy()
        
        # Run pipeline
        predictions, _ = pipeline.run_batch(history_np, label_np)
        
        print(f"Input shape: {history_np.shape}")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Labels shape: {label_np.shape}")
        
        # Calculate MSE for this batch
        mse = np.mean((predictions - label_np) ** 2)
        print(f"Batch MSE: {mse:.6f}")
        
        break
    
    # Step 7: Visualize Predictions
    print("\nStep 7: Visualizing Predictions")
    print("-" * 40)
    
    # Visualize predictions for first 3 samples
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    
    for i in range(3):
        # Get data for sample i
        sample_history = history_np[i].squeeze()
        sample_pred = predictions[i].squeeze()
        sample_label = label_np[i].squeeze()
        
        # Create time indices
        history_time = np.arange(len(sample_history))
        pred_time = np.arange(len(sample_history), len(sample_history) + len(sample_pred))
        
        # Plot
        axes[i].plot(history_time, sample_history, 
                    label='History', color='blue', linewidth=2)
        axes[i].plot(pred_time, sample_pred, 
                    label='Prediction', color='red', linestyle='--', linewidth=2)
        axes[i].plot(pred_time, sample_label, 
                    label='Ground Truth', color='green', linestyle=':', linewidth=2)
        
        axes[i].set_title(f'Sample {i+1} Predictions')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Value')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Add vertical line separating history and prediction
        axes[i].axvline(x=len(sample_history), color='black', 
                       linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('tutorial_predictions.png', dpi=150, bbox_inches='tight')
    print("Predictions visualization saved as 'tutorial_predictions.png'")
    
    # Step 8: Evaluate Full Pipeline
    print("\nStep 8: Evaluating Full Pipeline")
    print("-" * 40)
    
    print("Evaluating pipeline on training data...")
    
    metrics = pipeline.evaluate(train_loader)
    
    # Calculate average metrics
    avg_metrics = {}
    for metric_name, values in metrics.items():
        if len(values) > 0:
            avg_metrics[metric_name] = np.mean(values)
    
    print("\nAverage Metrics:")
    for metric_name, value in avg_metrics.items():
        print(f"  {metric_name}: {value:.6f}")
    
    # Step 9: Conclusion
    print("\n" + "=" * 60)
    print("Tutorial Completed Successfully!")
    print("=" * 60)
    print("\nIn this tutorial, we've demonstrated:")
    print("1. How to load and visualize time series data with TATO")
    print("2. How to initialize foundation models")
    print("3. How to configure data transformations")
    print("4. How to create and evaluate prediction pipelines")
    print("5. How to visualize and analyze results")
    print("\nNext steps:")
    print("1. Try different datasets (ETTh2, Weather, Exchange, etc.)")
    print("2. Experiment with different foundation models (MOIRAI, Chronos, etc.)")
    print("3. Explore more transformations (sampler, warper, differentiator, etc.)")
    print("4. Run adaptive transformation search with more trials")
    print("\nGenerated files:")
    print("1. tutorial_data_visualization.png - Data visualization")
    print("2. tutorial_predictions.png - Prediction visualization")

if __name__ == '__main__':
    main()
