from typing import List, Dict, Any, Optional, Tuple
from utils.metrics import metric_five_dict
from utils.tools import visualize_samples, loss_plot
from utils.clip import my_clip
import numpy as np
import logging
import copy


class BasePipeline:
    """Base pipeline for time series forecasting with adaptive transformations.
    
    This class orchestrates the complete forecasting pipeline:
    1. Preprocessing with transformations
    2. Model inference
    3. Post-processing with inverse transformations
    4. Evaluation and visualization
    
    The pipeline supports adaptive transformation optimization (TATO) by
    applying a sequence of transformations to the data before feeding it
    to foundation models.
    
    Attributes:
        transformations: List of transformation instances to apply
        config: Configuration dictionary for the pipeline
        model: Foundation model for forecasting
        pred_len: Prediction length (forecast horizon)
        augmentor: Optional data augmentation module
        save_dir: Directory to save visualizations and results
        mode: Pipeline mode ('train', 'val', 'test')
        id: Pipeline identifier for logging
        plt: Whether to generate visualizations
    """
    
    def __init__(self, transformations: List[Any], config: Dict[str, Any], model: Any, 
                 pred_len: int, augmentor: Optional[Any] = None, mode: str = 'train', 
                 id: Optional[str] = None, save_dir: Optional[str] = None, plt: bool = False):
        """Initialize the base pipeline.
        
        Args:
            transformations: List of transformation instances to apply in sequence
            config: Configuration dictionary for the pipeline
            model: Foundation model instance for forecasting
            pred_len: Number of time steps to forecast
            augmentor: Optional data augmentation module
            mode: Pipeline mode ('train', 'val', or 'test')
            id: Identifier for the pipeline (used in logging and saving)
            save_dir: Directory to save visualizations and results
            plt: Whether to generate visualization plots
        """
        self.transformations = transformations
        self.config = config
        self.model = model
        self.pred_len = pred_len
        self.augmentor = augmentor
        self.save_dir = save_dir
        self.mode = mode
        self.id = id
        self.plt = plt

    def preprocess(self, x: np.ndarray) -> np.ndarray:
        """Apply all transformations to input data.
        
        This method applies each transformation in sequence to preprocess
        the input data before feeding it to the model. Some transformations
        may require historical context, which is provided via update_history.
        
        Args:
            x: Input time series data of shape (batch_size, seq_len, num_features)
            
        Returns:
            Preprocessed data ready for model input
        """
        # Store a copy of original history for transformations that need it
        history = copy.deepcopy(x)
        
        # Apply each transformation in sequence
        for transformation in self.transformations:
            # Some transformations (e.g., normalizer with 'history' mode) need historical context
            if hasattr(transformation, "update_history") and callable(getattr(transformation, "update_history")):
                transformation.update_history(history)
            x = transformation.pre_process(x)
        return x

    def postprocess(self, x: np.ndarray) -> np.ndarray:
        """Apply inverse transformations to model output.
        
        This method applies the inverse of each transformation in reverse order
        to bring model predictions back to the original data scale.
        
        Args:
            x: Model predictions of shape (batch_size, pred_len, num_features)
            
        Returns:
            Post-processed predictions in original data scale
        """
        # Apply inverse transformations in reverse order
        for transformation in reversed(self.transformations):
            x = transformation.post_process(x)
        return x

    def run_batch(self, x: np.ndarray, y: Optional[np.ndarray] = None, 
                  split_idx: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Run the complete pipeline on a single batch.
        
        This method executes the full pipeline:
        1. Data augmentation (if enabled)
        2. Preprocessing with transformations
        3. Model forecasting
        4. Post-processing with inverse transformations
        5. Visualization (if enabled)
        
        Args:
            x: Input batch of shape (batch_size, seq_len, num_features)
            y: Target batch of shape (batch_size, pred_len, num_features) or None
            split_idx: Indices for tracking samples in visualization
            
        Returns:
            Tuple of (predictions, targets) where predictions are in original scale
        """
        # Apply data augmentation if augmentor is provided
        if self.augmentor is not None:
            x, y = self.augmentor.apply_augmentation(x, y)
        
        # Store original batch for visualization
        original_batch = x.clone() if hasattr(x, 'clone') else x.copy()
        
        # Step 1: Preprocess with transformations
        x_processed = self.preprocess(x)
        preprocessed_batch = x_processed.clone() if hasattr(x_processed, 'clone') else x_processed.copy()
        
        # Step 2: Model forecasting
        original_pred = self.model.forecast(x_processed, self.pred_len)

        # Step 3: Post-process predictions
        processed_pred = self.postprocess(original_pred)
        
        # Step 4: Generate visualizations if enabled
        if self.mode != 'train' and self.plt:
            visualize_samples(
                original_batch, y,
                preprocessed_batch, 
                original_pred,
                processed_pred, 
                self.save_dir, k=2, mode=self.mode, split_idx_batch=split_idx, id=self.id
            )
        
        return processed_pred, y

    def evaluate(self, dataloader: Any, scale: Optional[Any] = None) -> Dict[str, List[float]]:
        """Evaluate the pipeline on a dataloader.
        
        This method runs the pipeline on all batches in the dataloader and
        computes evaluation metrics for each batch.
        
        Args:
            dataloader: PyTorch DataLoader or similar iterable providing batches
            scale: Optional scaler to apply additional scaling to predictions
            
        Returns:
            Dictionary mapping metric names to lists of values (one per sample)
        """
        metrics = {}

        # Iterate through all batches in the dataloader
        for i, (split_idx, batch_x, batch_y) in enumerate(dataloader):
            # Convert PyTorch tensors to numpy arrays
            batch_x, batch_y, split_idx = batch_x.numpy(), batch_y.numpy(), split_idx.numpy()
            
            # Run pipeline on the batch
            output, batch_y = self.run_batch(batch_x, batch_y, split_idx)
            
            # Handle NaN/Inf values in predictions
            if np.isnan(output).any() or np.isinf(output).any():
                print(f"NaN or Inf values in output during evaluation")
                output = my_clip(batch_x, output, nan_inf_clip_factor=5)
            
            # Apply additional scaling if provided
            if scale is not None:
                output = scale.transform(output.reshape(-1, output.shape[2])).reshape(output.shape)
                batch_y = scale.transform(batch_y.reshape(-1, batch_y.shape[2])).reshape(batch_y.shape)
            
            # Compute metrics for this batch
            metric = self.metric_function(output, batch_y)

            # Aggregate metrics across batches
            for k, v in metric.items():
                metrics.setdefault(k, []).extend(v)

        return metrics

    def predict(self, dataloader: Any, scale: Optional[Any] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions for all samples in a dataloader.
        
        Args:
            dataloader: PyTorch DataLoader or similar iterable providing batches
            scale: Optional scaler to apply additional scaling to predictions
            
        Returns:
            Tuple of (predictions, labels) where both are numpy arrays
            concatenated across all batches
        """
        predictions, labels = [], []
        
        for i, (split_idx, batch_x, batch_y) in enumerate(dataloader):
            # Convert PyTorch tensors to numpy arrays
            batch_x, batch_y, split_idx = batch_x.numpy(), batch_y.numpy(), split_idx.numpy()
            
            # Run pipeline on the batch
            output, batch_y = self.run_batch(batch_x, batch_y, split_idx)
            
            # Handle NaN/Inf values in predictions
            if np.isnan(output).any() or np.isinf(output).any():
                print(f"NaN or Inf values in output during prediction")
                output = my_clip(batch_x, output, nan_inf_clip_factor=5)
            
            # Apply additional scaling if provided
            if scale is not None:
                output = scale.transform(output.reshape(-1, output.shape[2])).reshape(output.shape)
                batch_y = scale.transform(batch_y.reshape(-1, batch_y.shape[2])).reshape(batch_y.shape)
            
            # Collect predictions and labels
            labels.append(batch_y)
            predictions.append(output)
            
        # Concatenate all batches
        return np.concatenate(predictions, axis=0), np.concatenate(labels, axis=0)

    def metric_function(self, output: np.ndarray, batch_y: np.ndarray) -> Dict[str, List[float]]:
        """Compute evaluation metrics for predictions.
        
        Args:
            output: Model predictions of shape (batch_size, pred_len, num_features)
            batch_y: Ground truth labels of shape (batch_size, pred_len, num_features)
            
        Returns:
            Dictionary mapping metric names to lists of values (one per sample)
        """
        return metric_five_dict(output, batch_y)
    
    def avg_metric_list(self, metric_list: List[Dict[str, float]], 
                        weights_dict: Optional[Dict[str, float]] = None) -> float:
        """Compute weighted average of metrics across a list.
        
        Args:
            metric_list: List of metric dictionaries
            weights_dict: Dictionary mapping metric names to weights.
                         If None, all metrics have equal weight (1.0).
            
        Returns:
            Weighted average metric value
        """
        avg_metric = 0
        if weights_dict is None:
            weights_dict = {}
            
        # Sum weighted metrics
        for metric in metric_list:
            for key, value in metric.items():
                avg_metric += value * weights_dict.get(key, 1.0)
                
        # Divide by number of metric dictionaries
        avg_metric /= len(metric_list)
        return avg_metric
