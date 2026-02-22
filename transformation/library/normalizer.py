import logging
import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from transformation.base import BaseTransformation
from utils.clip import my_clip

# Global constant for clipping NaN/Inf values
nan_inf_clip_factor = 5


class Transformation(BaseTransformation):
    """Normalization transformation for time series data.
    
    This transformation scales time series data using various normalization methods.
    It supports different normalization modes and methods, and can handle both
    numpy arrays and torch tensors.
    
    Attributes:
        search_space: Hyperparameter search space for the normalizer
        method: Normalization method ('none', 'standard', 'minmax', 'maxabs', 'robust')
        mode: Normalization mode ('none', 'dataset', 'input', 'history')
        data_in: Original input data (stored for post-processing)
        scaler_params: Parameters computed during pre-processing for each feature
        history_data: Historical data for 'history' mode normalization
        clip_factor: Factor for clipping extreme values during post-processing
        scaler: Scikit-learn scaler instance for 'dataset' mode
    """
    
    search_space = {
        # 'method': ['none', 'standard', 'minmax', 'maxabs', 'robust'],
        # 'mode': ['none', 'dataset', 'input', 'history']
        'method': ['none', 'standard', 'robust'],
        'mode': ['input']
    }

    def __init__(self, method: str, mode: str, **kwargs):
        """Initialize the normalizer transformation.
        
        Args:
            method: Normalization method. Options:
                   - 'none': No normalization
                   - 'standard': Standard scaling (mean=0, std=1)
                   - 'minmax': Min-max scaling to [0, 1]
                   - 'maxabs': Max-absolute scaling to [-1, 1]
                   - 'robust': Robust scaling using median and IQR
            mode: Normalization mode. Options:
                 - 'none': No normalization
                 - 'dataset': Fit scaler on entire dataset
                 - 'input': Normalize each input independently
                 - 'history': Normalize using historical context
            **kwargs: Additional parameters including:
                     - clip_factor: Factor for clipping extreme values ('none' or float)
        """
        # Validate input parameters
        assert method in self.search_space['method'], f"Method must be one of {self.search_space['method']}"
        assert mode in self.search_space['mode'], f"Mode must be one of {self.search_space['mode']}"
        
        self.method = method
        self.mode = mode
        self.data_in = None  # Store original input for post-processing
        self.scaler_params = {}  # Store scaling parameters for each feature
        self.history_data = None  # Historical data for 'history' mode
        
        # Get clipping factor for handling extreme values
        clip_factor = kwargs.get('clip_factor', 'none')
        self.clip_factor = float(clip_factor) if clip_factor != 'none' else 5

        # Early return for 'none' modes
        if mode == 'none' or method == 'none':
            return
        
        # Initialize scaler for 'dataset' mode
        if mode == 'dataset':
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
            elif method == 'maxabs':
                self.scaler = MaxAbsScaler()
            elif method == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError('Invalid normalizer method: {}'.format(self.method))
        elif mode in ['input', 'history']:
            # For 'input' and 'history' modes, scaler parameters are computed per batch
            pass
        else:
            raise ValueError('Invalid normalizer mode: {}'.format(self.mode))

    def update_history(self, history_data: Union[np.ndarray, torch.Tensor]):
        """Update historical data for 'history' mode normalization.
        
        Args:
            history_data: Historical time series data used for computing
                         normalization statistics in 'history' mode.
        """
        self.history_data = history_data


    def _compute_scaler_params(self, data: Union[np.ndarray, torch.Tensor], 
                              look_back_ratio: float) -> Dict[int, Tuple]:
        """Compute scaling parameters for each feature in the data.
        
        Args:
            data: Input data of shape (batch_size, seq_len, num_features)
            look_back_ratio: Ratio of sequence length to use for computing statistics
                           (e.g., 1.0 uses entire sequence, 0.5 uses last half)
            
        Returns:
            Dictionary mapping feature index to scaling parameters tuple.
            The tuple format depends on the normalization method:
            - 'standard': (mean, std)
            - 'minmax': (min_val, max_val)
            - 'robust': (median, q1, q3)
        """
        assert data.ndim == 3  # Must be (batch, time, feature)
        batch, time, feature = data.shape

        # Compute look-back length based on ratio
        look_back_len = int(time * look_back_ratio)

        # Store original data for post-processing
        self.data_in = data
        
        params = {}
        for i in range(feature):
            # Extract feature data and use only look-back portion
            feature_data = data[:, :, i].reshape(batch, -1)[:, -look_back_len:]
            
            if self.method == 'standard':
                if isinstance(data, np.ndarray):
                    mean = np.mean(feature_data, axis=1, keepdims=True)
                    std = np.std(feature_data, axis=1, keepdims=True)
                elif isinstance(data, torch.Tensor):
                    mean = torch.mean(feature_data, dim=1, keepdims=True)
                    std = torch.std(feature_data, dim=1, keepdims=True)
                else:
                    raise ValueError('Invalid data type: {}'.format(type(data)))
                params[i] = (mean, std)
                
            elif self.method == 'minmax':
                if isinstance(data, np.ndarray):
                    min_val = np.min(feature_data, axis=1, keepdims=True)
                    max_val = np.max(feature_data, axis=1, keepdims=True)
                elif isinstance(data, torch.Tensor):
                    min_val = torch.min(feature_data, dim=1, keepdims=True)
                    max_val = torch.max(feature_data, dim=1, keepdims=True)
                else:
                    raise ValueError('Invalid data type: {}'.format(type(data)))
                params[i] = (min_val, max_val)
                
            elif self.method == 'robust':
                if isinstance(data, np.ndarray):
                    median = np.median(feature_data, axis=1, keepdims=True)
                    q1 = np.percentile(feature_data, 25, axis=1, keepdims=True)
                    q3 = np.percentile(feature_data, 75, axis=1, keepdims=True)
                elif isinstance(data, torch.Tensor):
                    median = torch.median(feature_data, dim=1, keepdims=True).values
                    q1 = torch.quantile(feature_data, 0.25, dim=1, keepdim=True)
                    q3 = torch.quantile(feature_data, 0.75, dim=1, keepdim=True)
                else:
                    raise ValueError('Invalid data type: {}'.format(type(data)))
                params[i] = (median, q1, q3)
                
        return params


    def pre_process(self, data):
        if self.mode == 'none' or self.method == 'none':
            return data
        assert len(data.shape) == 3  # (batch, time, feature)
        batch, time, feature = data.shape
        if isinstance(data, np.ndarray):
            res = np.zeros_like(data)
        elif isinstance(data, torch.Tensor):
            res = torch.zeros_like(data)
        else:
            raise ValueError('Invalid data type: {}'.format(type(data)))
        if self.mode == 'dataset':
            for i in range(feature):
                feature_data = data[:, :, i].reshape(-1, 1)
                res[:, :, i] = self.scaler.transform(feature_data).reshape(batch, time)
        else:
            tmp_d = data if self.mode == 'input' else self.history_data
            self.scaler_params = self._compute_scaler_params(tmp_d, 1)
            for i in range(feature):
                feature_data = data[:, :, i].reshape(batch, -1)
                if self.method == 'standard':
                    mean, std = self.scaler_params[i]
                    res[:, :, i] = ((feature_data - mean) / (std + 1e-8)).reshape(batch, time)
                elif self.method == 'minmax':
                    min_val, max_val = self.scaler_params[i]
                    res[:, :, i] = ((feature_data - min_val) / (max_val - min_val + 1e-8)).reshape(batch, time)
                elif self.method == 'maxabs':
                    max_abs_val = self.scaler_params[i]
                    res[:, :, i] = (feature_data / (max_abs_val + 1e-8)).reshape(batch, time)
                elif self.method == 'robust':
                    median, q1, q3 = self.scaler_params[i]
                    res[:, :, i] = ((feature_data - median) / (q3 - q1 + 1e-8)).reshape(batch, time)

        return res

    def post_process(self, data):
        if self.mode == 'none' or self.method == 'none':
            return data
        assert len(data.shape) == 3  # (batch, time, feature)
        batch, time, feature = data.shape
        if isinstance(data, np.ndarray):
            res = np.zeros_like(data)
        elif isinstance(data, torch.Tensor):
            res = torch.zeros_like(data)
        else:
            raise ValueError('Invalid data type: {}'.format(type(data)))
        if self.mode == 'dataset':
            for i in range(feature):
                feature_data = data[:, :, i].reshape(-1, 1)
                res[:, :, i] = self.scaler.inverse_transform(
                    feature_data).reshape(batch, time)
        else:
            for i in range(feature):
                feature_data = data[:, :, i].reshape(batch, -1)
                if self.method == 'standard':
                    mean, std = self.scaler_params[i]
                    res[:, :, i] = (feature_data * std +
                                    mean).reshape(batch, time)
                elif self.method == 'minmax':
                    min_val, max_val = self.scaler_params[i]
                    res[:, :, i] = (
                        feature_data * (max_val - min_val) + min_val).reshape(batch, time)
                elif self.method == 'maxabs':
                    max_abs_val = self.scaler_params[i]
                    res[:, :, i] = (
                        feature_data * max_abs_val).reshape(batch, time)
                elif self.method == 'robust':
                    median, q1, q3 = self.scaler_params[i]
                    res[:, :, i] = (feature_data * (q3 - q1) +
                                    median).reshape(batch, time)
        if isinstance(self.data_in, np.ndarray):
            if np.isnan(res).any() or np.isinf(res).any():
                logging.error(f"NaN or Inf values in restored data")
                res = my_clip(
                    self.data_in, res, nan_inf_clip_factor=nan_inf_clip_factor)
            elif self.clip_factor is not None:
                res = my_clip(self.data_in, res,
                              min_max_clip_factor=self.clip_factor)
        elif isinstance(self.data_in, torch.Tensor):
            if torch.isnan(res).any() or torch.isinf(res).any():
                res = my_clip(
                    self.data_in, res, nan_inf_clip_factor=nan_inf_clip_factor)
            elif self.clip_factor is not None:
                res = my_clip(self.data_in, res,
                              min_max_clip_factor=self.clip_factor)
        else:
            raise ValueError('Invalid data type: {}'.format(type(data)))

        return res
