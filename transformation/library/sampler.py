from math import ceil
import numpy as np
import torch
from scipy import signal
from transformation.base import BaseTransformation

def assert_timeseries_3d_np(data):
    assert type(data) is np.ndarray and data.ndim == 3, \
        f'type(data)={type(data)}, data.ndim={data.ndim}, data.shape={data.shape}'

def assert_timeseries_3d_tensor(data):
    assert type(data) is torch.Tensor and data.ndim == 3, \
        f'type(data)={type(data)}, data.ndim={data.ndim}, data.shape={data.shape}'
    
    
class Transformation(BaseTransformation):
    search_space = {
        # 'factor': [0.5, 1, 2]
        'factor': [1, 2]
    }
    
    def __init__(self, factor, **kwargs):
        assert factor in self.search_space['factor'], f"Factor must be one of {self.search_space['factor']}"
        self.factor = factor
    
    def torch_resample(self, x: torch.Tensor, num: int, dim: int = -1) -> torch.Tensor:
        """Resample a tensor along a dimension (similar to scipy.signal.resample)."""
        N = x.size(dim)
        
        X = torch.fft.rfft(x, dim=dim)
        
        if num > N:
            X_resampled = torch.zeros((num,), dtype=X.dtype)
            slices = [slice(None)] * X.ndim
            slices[dim] = slice(0, X.size(dim))
            X_resampled[slices] = X
        else:
            X_resampled = X.narrow(dim, 0, num // 2 + 1)
        
        x_resampled = torch.fft.irfft(X_resampled, n=num, dim=dim)
        
        x_resampled *= (num / N) ** 0.5
        
        return x_resampled

    def pre_process(self, data):
        if self.factor == 1:
            return data
 
        if type(data) is np.ndarray:
            assert_timeseries_3d_np(data)
            batch, time, feature = data.shape
            res = np.zeros((batch, ceil(time / self.factor), feature))
            for b in range(batch):
                for f in range(feature):
                    res[b, :, f] = signal.resample(data[b, :, f], ceil(time / self.factor))
        elif type(data) is torch.Tensor:
            assert_timeseries_3d_tensor(data)
            batch, time, feature = data.shape
            res = torch.zeros((batch, ceil(time * self.factor), feature)).to(data.device)
            for b in range(batch):
                for f in range(feature):
                    res[b, :, f] = self.torch_resample(data[b, :, f], ceil(time * self.factor)).to(data.device)
        else:
            res = data
 
        return res

    def post_process(self, data):
        if self.factor == 1:
            return data
        if type(data) is np.ndarray:
            assert_timeseries_3d_np(data)
            batch, time, feature = data.shape
            res = np.zeros((batch, ceil(time * self.factor), feature))
            for b in range(batch):
                for f in range(feature):
                    res[b, :, f] = signal.resample(data[b, :, f], ceil(time * self.factor))
        elif type(data) is torch.Tensor:
            assert_timeseries_3d_tensor(data)
            batch, time, feature = data.shape
            res = torch.zeros((batch, ceil(time * self.factor), feature)).to(data.device)
            for b in range(batch):
                for f in range(feature):
                    res[b, :, f] = self.torch_resample(data[b, :, f], ceil(time * self.factor))
        else:
            res = data
        return res
