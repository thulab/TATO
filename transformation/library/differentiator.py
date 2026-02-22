import logging
import numpy as np
import torch
from transformation.base import BaseTransformation
from utils.clip import my_clip
from transformation.library.aligner import Transformation as Aligner

class Transformation(BaseTransformation):
    search_space = {
        # 'n': [0, 1, 2, 3]
        'n': [0, 1],
    }
    
    # def __init__(self, n, clip_factor, **kwargs):
    def __init__(self, **kwargs):
        self.n = kwargs.get('n', 0)
        self.history_diff_data = []
        self.diff_data = None
        self.data_in = None
        clip_factor = kwargs.get('clip_factor', 'none')
        self.clip_factor = float(clip_factor) if clip_factor != 'none' else 5


    def pre_process(self, data):
        if self.n == 0:
            return data

        batch, time, feature = data.shape
        self.data_in = data

        self.history_diff_data = []
        # diff_data = data.copy()
        diff_data = data

        for _ in range(self.n):
            self.history_diff_data.append(diff_data[:, 0:1, :])
            if isinstance(data, np.ndarray):
                diff_data = np.diff(diff_data, axis=1)
            elif isinstance(data, torch.Tensor):
                diff_data = torch.diff(diff_data, dim=1)
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")

        self.diff_data = diff_data

        aligner = Aligner('data_patch', 'zero_pad', time, time)  # FIXME
        res = aligner.pre_process(diff_data)
        return res

    def post_process(self, data):
        if self.n == 0:
            return data

        batch, time, feature = data.shape


        if isinstance(data, np.ndarray):
            inv_diff_data_total = np.concatenate([self.diff_data, data], axis=1)
            for i in range(self.n - 1, -1, -1):
                inv_diff_data_total = np.concatenate([self.history_diff_data[i], inv_diff_data_total], axis=1)
                inv_diff_data_total = np.cumsum(inv_diff_data_total, axis=1)
    
            pre_time = self.diff_data.shape[1]
            assert pre_time + time + self.n == inv_diff_data_total.shape[1], \
                f"{pre_time} + {self.n} + {time} != {inv_diff_data_total.shape[1]}"
            inv_diff_data = inv_diff_data_total[:, pre_time:pre_time + time, :]

            res = inv_diff_data
            # IQR-variant
            if np.isnan(res).any() or np.isinf(res).any():
                logging.error(f"NaN or Inf values in restored data")
                res = my_clip(self.data_in, res, nan_inf_clip_factor=5)
            elif self.clip_factor is not None:
                res = my_clip(self.data_in, res, min_max_clip_factor=self.clip_factor)

        elif isinstance(data, torch.Tensor):
            inv_diff_data_total = torch.cat([self.diff_data, data], dim=1)
            for i in range(self.n - 1, -1, -1):
                inv_diff_data_total = torch.cat([self.history_diff_data[i], inv_diff_data_total], dim=1)
                inv_diff_data_total = torch.cumsum(inv_diff_data_total, dim=1)
            
            pre_time = self.diff_data.shape[1]
            assert pre_time + time + self.n == inv_diff_data_total.shape[1], \
                f"{pre_time} + {self.n} + {time} != {inv_diff_data_total.shape[1]}"
            inv_diff_data = inv_diff_data_total[:, pre_time:pre_time + time, :]

            res = inv_diff_data
            # IQR-variant
            if torch.isnan(res).any() or torch.isinf(res).any():
                logging.error(f"NaN or Inf values in restored data")
                res = my_clip(self.data_in, res, nan_inf_clip_factor=5)
            elif self.clip_factor is not None:
                res = my_clip(self.data_in, res, min_max_clip_factor=self.clip_factor)
        else:
            raise ValueError(f"Invalid data type: {type(data)}")

        return res
