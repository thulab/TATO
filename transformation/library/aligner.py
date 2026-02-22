import numpy as np
import torch
import torch.nn.functional as F
from transformation.base import BaseTransformation


def assert_timeseries_3d_np(data):
    assert type(data) is np.ndarray and data.ndim == 3, \
        f'type(data)={type(data)}, data.ndim={data.ndim}, data.shape={data.shape}'

def assert_timeseries_3d_tensor(data):
    assert type(data) is torch.Tensor and data.ndim == 3, \
        f'type(data)={type(data)}, data.ndim={data.ndim}, data.shape={data.shape}'

class Transformation(BaseTransformation):
    search_space = {
        'mode': ['none', 'data_patch'],
        'method': ['edge_pad']
    }
    
    def __init__(self, mode, method, data_patch_len=96, model_patch_len=96, **kwargs):
        assert mode in ['none', 'data_patch', 'model_patch']
        assert method in ['none', 'trim', 'zero_pad', 'mean_pad', 'edge_pad']
        self.mode = mode
        self.method = method
        self.patch_len = data_patch_len if mode == 'data_patch' else model_patch_len

    def pre_process(self, data):  # padding mostly
        if self.mode == 'none' or self.method == 'none':
            return data
        if type(data) is np.ndarray:
            assert_timeseries_3d_np(data)
        elif type(data) is torch.Tensor:
            assert_timeseries_3d_tensor(data)
        batch, time, feature = data.shape
        if time % self.patch_len == 0:
            return data
        pad_l = self.patch_len - time % self.patch_len if time % self.patch_len != 0 else 0
        if isinstance(data, np.ndarray):
            res = np.zeros((batch, pad_l + time, feature))
        elif isinstance(data, torch.Tensor):
            res = torch.zeros((batch, pad_l + time, feature)).to(data.device)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        if self.method == 'trim':
            if time < self.patch_len:
                self.method = 'edge_pad'
            else:
                valid_len = time // self.patch_len * self.patch_len
                return data[:, -valid_len:, :]

        for b in range(batch):
            for f in range(feature):
                if isinstance(data, np.ndarray):
                    if self.method == 'zero_pad':
                        res[b, :, f] = np.pad(data[b, :, f], (pad_l, 0), 'constant', constant_values=0)
                    elif self.method == 'mean_pad':
                        res[b, :, f] = np.pad(data[b, :, f], (pad_l, 0), 'constant', constant_values=np.mean(data[b, :, f]))
                    elif self.method == 'edge_pad':
                        res[b, :, f] = np.pad(data[b, :, f], (pad_l, 0), 'edge')
                    else:
                        raise Exception('Invalid aligner: {}'.format(self.method))
                elif isinstance(data, torch.Tensor):
                    if self.method == 'zero_pad':
                        res[b, :, f] = F.pad(data[b, :, f].unsqueeze(0).unsqueeze(0), (pad_l, 0), mode='constant', value=0).squeeze()
                    elif self.method == 'mean_pad':
                        mean_val = torch.mean(data[b, :, f])
                        res[b, :, f] = F.pad(data[b, :, f].unsqueeze(0).unsqueeze(0), (pad_l, 0), mode='constant', value=mean_val).squeeze()
                    elif self.method == 'edge_pad':
                        res[b, :, f] = F.pad(data[b, :, f].unsqueeze(0).unsqueeze(0), (pad_l, 0), mode='replicate').squeeze()
                    else:
                        raise Exception('Invalid aligner: {}'.format(self.method))
                else:
                    raise ValueError(f"Unsupported data type: {type(data)}")
        return res

    def post_process(self, data):
        return data
