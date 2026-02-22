import numpy as np
import torch
from transformation.base import BaseTransformation


def assert_timeseries_3d_np(data):
    assert type(data) is np.ndarray and data.ndim == 3, \
        f'type(data)={type(data)}, data.ndim={data.ndim}, data.shape={data.shape}'


def assert_timeseries_3d_tensor(data):
    assert type(data) is torch.Tensor and data.ndim == 3, \
        f'type(data)={type(data)}, data.ndim={data.ndim}, data.shape={data.shape}'

class Transformation(BaseTransformation):
    search_space = {
        'seq_l': list(range(5, 16)),
    }

    def __init__(self, seq_l, patch_len, pred_len, **kwargs):
        assert seq_l in self.search_space['seq_l'], f"Seq_l must be one of {self.search_space['seq_l']}"
        self.seq_l = seq_l * patch_len
        self.pred_l = pred_len
    
    def pre_process(self, data):
        if data.shape[1] == self.seq_l:
            return data
        if type(data) is np.ndarray:
            assert_timeseries_3d_np(data)
            assert data.shape[1] >= self.seq_l, f'Invalid data shape: {data.shape} for seq_l={self.seq_l}'
            return data[:, -self.seq_l:, :]
        if type(data) is torch.Tensor:
            assert_timeseries_3d_tensor(data)
            assert data.shape[1] >= self.seq_l, f'Invalid data shape: {tuple(data.shape)} for seq_l={self.seq_l}'
            return data[:, -self.seq_l:, :]
        raise ValueError(f"Unsupported data type: {type(data)}")

    def post_process(self, data):
        if data.shape[1] == self.pred_l:
            return data
        if type(data) is np.ndarray:
            assert_timeseries_3d_np(data)
            assert data.shape[1] >= self.pred_l, f'Invalid data shape: {data.shape} for pred_l={self.pred_l}'
            return data[:, :self.pred_l, :]
        if type(data) is torch.Tensor:
            assert_timeseries_3d_tensor(data)
            assert data.shape[1] >= self.pred_l
            return data[:, :self.pred_l, :]
        raise ValueError(f"Unsupported data type: {type(data)}")
