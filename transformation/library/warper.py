import logging
import numpy as np
import torch
from transformation.base import BaseTransformation
from utils.clip import my_clip

nan_inf_clip_factor = 5

def assert_timeseries_3d_np(data):
    assert type(data) is np.ndarray and data.ndim == 3, \
        f'type(data)={type(data)}, data.ndim={data.ndim}, data.shape={data.shape}'

def assert_timeseries_3d_tensor(data):
    assert type(data) is torch.Tensor and data.ndim == 3, \
        f'type(data)={type(data)}, data.ndim={data.ndim}, data.shape={data.shape}'
    
class Transformation(BaseTransformation):
    search_space = {
        # 'method': ['none', 'log', 'sqrt'],
        'method': ['none', 'log']
    }

    def __init__(self, method, clip_factor, **kwargs):
        assert method in self.search_space['method'], f"Method must be one of {self.search_space['method']}"
        self.method = method
        self.shift_values = None
        self.box_cox_lambda = None
        self.fail = False
        self.data_in = None
        self.clip_factor = float(clip_factor) if clip_factor != 'none' else nan_inf_clip_factor

    def pre_process(self, data):
        assert len(data.shape)==3, f'Invalid data shape: {data.shape}'
        self.data_in = data
        if self.method == 'none':
            return data

        batch_size, time_len, feature_dim = data.shape

        if self.method == 'log':
            if isinstance(data, np.ndarray):
                min_values = np.min(data, axis=1, keepdims=True)
                self.shift_values = np.where(min_values <= 1, 1 - min_values, 0)
                data_shifted = data + self.shift_values
                res = np.log(data_shifted)
            elif isinstance(data, torch.Tensor):
                min_values = torch.min(data, dim=1, keepdim=True).values
                self.shift_values = torch.where(min_values <= 1, 1 - min_values, torch.tensor(0., dtype=data.dtype, device=data.device))
                data_shifted = data + self.shift_values
                res = torch.log(data_shifted)
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
        
        elif self.method == 'sqrt':
            if isinstance(data, np.ndarray):
                min_values = np.min(data, axis=1, keepdims=True)
                self.shift_values = np.where(min_values < 0, 1 - min_values, 0)
                data_shifted = data + self.shift_values
                res = np.sqrt(data_shifted)
            elif isinstance(data, torch.Tensor):
                min_values = torch.min(data, dim=1, keepdim=True).values
                zero_tensor = torch.tensor(0., dtype=data.dtype, device=data.device)
                self.shift_values = torch.where(min_values < 0, 1 - min_values, zero_tensor)
                data_shifted = data + self.shift_values
                res = torch.sqrt(data_shifted)
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
        else:
            raise Exception('Invalid warper method: {}'.format(self.method))
        
        assert len(res.shape) == len(data.shape), f'Invalid data shape: {res.shape}'
        if isinstance(res, np.ndarray):
            assert np.isnan(res).sum() == 0 and np.isinf(res).sum() == 0, \
                f'Invalid data: {res}, method: {self.method}'

            if np.isnan(res).any() or np.isinf(res).any():
                logging.error(f"NaN or Inf values in transformed data")
                self.fail = True
                return data
        elif isinstance(res, torch.Tensor):
            assert torch.isnan(res).sum() == 0 and torch.isinf(res).sum() == 0, \
                f'Invalid data: {res}, method: {self.method}'
            
            if torch.isnan(res).any() or torch.isinf(res).any():
                logging.error(f"NaN or Inf values in transformed data")
                self.fail = True
                return data
        else:
            raise ValueError(f"Unsupported data type: {type(res)}")
        return res

    def post_process(self, data):
        if self.method == 'none' or self.fail:
            if isinstance(data, np.ndarray):
                if np.isnan(data).any() or np.isinf(data).any():
                    logging.error(f"NaN or Inf values in restored data")
                    data = my_clip(self.data_in, data, nan_inf_clip_factor=5)
                elif data.shape[2] > 1 and self.clip_factor is not None:
                    data = my_clip(self.data_in, data, min_max_clip_factor=self.clip_factor)
            elif isinstance(data, torch.Tensor):
                if torch.isnan(data).any() or torch.isinf(data).any():
                    logging.error(f"NaN or Inf values in restored data")
                    data = my_clip(self.data_in, data, nan_inf_clip_factor=5)
                elif data.shape[2] > 1 and self.clip_factor is not None:
                    data = my_clip(self.data_in, data, min_max_clip_factor=self.clip_factor)
            return data

        # if self.fail:
        #     return data

        if self.method == 'log':
            if isinstance(data, np.ndarray):
                _data = np.exp(data)
            elif isinstance(data, torch.Tensor):
                _data = torch.exp(data)
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
            data_restored = _data - self.shift_values

        elif self.method == 'sqrt':
            if isinstance(data, np.ndarray):
                data_restored = np.square(data)
            elif isinstance(data, torch.Tensor):
                data_restored = torch.square(data)
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
            data_restored = data_restored - self.shift_values

        else:
            raise Exception('Invalid warper method: {}'.format(self.method))

        res = data_restored
        if isinstance(data, np.ndarray):
            if np.isnan(res).any() or np.isinf(res).any():
                logging.error(f"NaN or Inf values in restored data")
                res = my_clip(self.data_in, res, nan_inf_clip_factor=5)
            elif self.clip_factor is not None:
                res = my_clip(self.data_in, res, min_max_clip_factor=self.clip_factor)
        elif isinstance(data, torch.Tensor):
            if torch.isnan(res).any() or torch.isinf(res).any():
                logging.error(f"NaN or Inf values in restored data")
                res = my_clip(self.data_in, res, nan_inf_clip_factor=5)
            elif self.clip_factor is not None:
                res = my_clip(self.data_in, res, min_max_clip_factor=self.clip_factor)
        return res
