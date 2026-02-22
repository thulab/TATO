import logging
import numpy as np
import torch
from transformation.base import BaseTransformation


class Transformation(BaseTransformation):
    search_space = {
        # 'detect_method': ['none', '2_sigma', '3_sigma', '1.5_iqr', '3_iqr'],
        # 'fill_method': ['none', 'linear_interpolate', 'rolling_mean', 'forward_fill', 'backward_fill']
        'detect_method': ['none', '3_sigma', '1.5_iqr'],
        'fill_method': ['linear_interpolate']
    }
    
    def __init__(self, detect_method, fill_method, **kwargs):
        # history_seq: (batch, time, feature)
        assert detect_method in self.search_space['detect_method'], f"Detect method must be one of {self.search_space['detect_method']}"
        assert fill_method in self.search_space['fill_method'], f"Fill method must be one of {self.search_space['fill_method']}"
        self.detect_method = detect_method
        self.fill_method = fill_method
        self.history_data = None
        self.statistics_dict = {}
        
    def update_history(self, history_data):
        self.history_data = history_data

    def get_statistics_dict(self, history_seq):
        if self.detect_method == 'none' or self.fill_method == 'none':
            return None
        if 'sigma' in self.detect_method:
            if type(history_seq) is torch.Tensor:
                mean = torch.mean(history_seq, axis=1, keepdims=True)
                std = torch.std(history_seq, axis=1, keepdims=True)
            elif type(history_seq) is np.ndarray:
                 mean = np.mean(history_seq, axis=1, keepdims=True)
                 std = np.std(history_seq, axis=1, keepdims=True)
            statistics_dict = {'mean': mean, 'std': std}
        elif 'iqr' in self.detect_method:
            if type(history_seq) is torch.Tensor:
                q1 = torch.quantile(history_seq, 0.25, dim=1, keepdim=True)
                q3 = torch.quantile(history_seq, 0.75, dim=1, keepdim=True)
            elif type(history_seq) is np.ndarray:
                q1 = np.percentile(history_seq, 25, axis=1, keepdims=True)
                q3 = np.percentile(history_seq, 75, axis=1, keepdims=True)
            statistics_dict = {'q1': q1, 'q3': q3}
        else:
            raise ValueError(f"Unsupported detect method: {self.detect_method}")
        return statistics_dict


    def pre_process(self, data):
        if self.detect_method == 'none' or self.fill_method == 'none':
            return data

        self.statistics_dict = self.get_statistics_dict(self.history_data)

        if 'sigma' in self.detect_method:
            k_sigma = float(self.detect_method.split('_')[0])
            fill_indices = self.detect_outliers_k_sigma(data, k_sigma)
        elif 'iqr' in self.detect_method:
            ratio = float(self.detect_method.split('_')[0])
            fill_indices = self.detect_outliers_iqr(data, ratio)
        else:
            raise ValueError(f"Unsupported detect method: {self.detect_method}")

        # tail_ratio = 1 / 4
        tail_ratio = 1
        batch_size, seq_len, feature_dim = data.shape
        rm_indices = set()


        consecutive_count = 0
        threshold = 1
        if len(fill_indices) > 0:  # ! Is 'fill_indices' possible to be None?
            for idx in range(1, len(fill_indices[0])):
                batch_idx_last, batch_idx_cur = fill_indices[0][idx - 1], fill_indices[0][idx]
                feature_idx_last, feature_idx_cur = fill_indices[2][idx - 1], fill_indices[2][idx]
                time_idx_last, time_idx_cur = fill_indices[1][idx - 1], fill_indices[1][idx]
                if batch_idx_cur == batch_idx_last and feature_idx_last == feature_idx_cur \
                        and time_idx_cur - time_idx_last == 1 and time_idx_cur > seq_len * (1 - tail_ratio):
                    consecutive_count += 1
                    if consecutive_count >= threshold:
                        rm_indices.update(range(idx - threshold, idx))
                else:
                    consecutive_count = 1 
            # TODO
            new_fill_indices = [[], [], []]
            for idx in range(len(fill_indices[0])):
                if idx not in rm_indices:
                    new_fill_indices[0].append(fill_indices[0][idx])
                    new_fill_indices[1].append(fill_indices[1][idx])
                    new_fill_indices[2].append(fill_indices[2][idx])
            if isinstance(data, np.ndarray):
                new_fill_indices[0] = np.array(new_fill_indices[0])
                new_fill_indices[1] = np.array(new_fill_indices[1])
                new_fill_indices[2] = np.array(new_fill_indices[2])
            elif isinstance(data, torch.Tensor):
                new_fill_indices[0] = torch.tensor(new_fill_indices[0]).to(data.device)
                new_fill_indices[1] = torch.tensor(new_fill_indices[1]).to(data.device)
                new_fill_indices[2] = torch.tensor(new_fill_indices[2]).to(data.device)
            new_fill_indices = tuple(new_fill_indices)
            logging.debug(f"fill_indices: {fill_indices}")
            logging.debug(f"new_fill_indices: {new_fill_indices}")
            fill_indices = new_fill_indices
    
            filled_data = self.fill_outliers(data, fill_indices)
            if isinstance(data, np.ndarray):
                if np.isnan(filled_data).any() or np.isinf(filled_data).any():
                    logging.error(f"NaN or Inf values in filled data: {filled_data}")
                    return data
            elif isinstance(data, torch.Tensor):
                if torch.isnan(filled_data).any() or torch.isinf(filled_data).any():
                    logging.error(f"NaN or Inf values in filled data: {filled_data}")
                    return data
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
        else:
            filled_data = data
        return filled_data

    def post_process(self, data):
        return data


    def detect_outliers_k_sigma(self, data, k_sigma):
        seq_len = data.shape[1]
        cutoff_index = seq_len  # 2-sigma
        mean = self.statistics_dict['mean']
        std = self.statistics_dict['std']
        lower_bound = mean - k_sigma * std
        upper_bound = mean + k_sigma * std
        mask = (data[:, :cutoff_index] < lower_bound) | (data[:, :cutoff_index] > upper_bound)
        if type(data) is np.ndarray:
            fill_indices = np.where(mask)
        elif type(data) is torch.Tensor:
            fill_indices = torch.where(mask)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        return fill_indices


    def detect_outliers_iqr(self, data, ratio):
        seq_len = data.shape[1]
        cutoff_index = seq_len
        q1 = self.statistics_dict['q1']
        q3 = self.statistics_dict['q3']
        iqr = q3 - q1
        lower_bound = q1 - ratio * iqr
        upper_bound = q3 + ratio * iqr
        if type(data) is torch.Tensor:
            lower_bound = lower_bound.to(data.device)
            upper_bound = upper_bound.to(data.device)
        
        mask = (data[:, :cutoff_index] < lower_bound) | (data[:, :cutoff_index] > upper_bound)
        if type(data) is np.ndarray:
            fill_indices = np.where(mask)
        elif type(data) is torch.Tensor:
            # fill_indices = torch.nonzero(mask)
            fill_indices = torch.where(mask)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        return fill_indices


    def fill_outliers(self, data, fill_indices):
        if self.detect_method == 'none' or self.fill_method == 'none':
            return data

        if self.fill_method == 'linear_interpolate':
            filled_data = self.linear_interpolate(data, fill_indices)
        elif self.fill_method == 'rolling_mean':
            filled_data = self.rolling_mean(data, fill_indices)
        elif self.fill_method == 'forward_fill':
            filled_data = self.forward_fill(data, fill_indices)
        elif self.fill_method == 'backward_fill':
            filled_data = self.backward_fill(data, fill_indices)
        else:
            raise ValueError(f"Unsupported fill method: {self.fill_method}")

        return filled_data


    def rolling_mean(self, data, fill_indices):
        window_size = 1000 
        if type(data) is np.ndarray:
            filled_data = data.copy()
        elif type(data) is torch.Tensor:
            filled_data = data.clone()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        batch_size, seq_len, feature_dim = data.shape
        for b in range(batch_size):
            for f in range(feature_dim):
                indices = fill_indices[1][fill_indices[0] == b]
                for idx in indices:
                    start = max(0, idx - window_size)
                    end = min(seq_len, idx + window_size + 1)
                    neighbors = data[b, start:end, f]
                    valid_neighbors = neighbors[neighbors != 0]
                    if len(valid_neighbors) > 0:
                        if type(data) is np.ndarray:
                            filled_data[b, idx, f] = np.mean(valid_neighbors)
                        elif type(data) is torch.Tensor:
                            filled_data[b, idx, f] = torch.mean(valid_neighbors)
                        else:
                            raise ValueError(f"Unsupported data type: {type(data)}")
        return filled_data


    def forward_fill(self, data, fill_indices):
        if type(data) is np.ndarray:
            filled_data = data.copy()
        elif type(data) is torch.Tensor:
            filled_data = data.clone()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        batch_size, seq_len, feature_dim = data.shape
        for b in range(batch_size):
            for f in range(feature_dim):
                indices = fill_indices[1][fill_indices[0] == b]
                for idx in indices:
                    if idx > 0:
                        filled_data[b, idx, f] = filled_data[b, idx - 1, f]
                    else:
                        filled_data[b, idx, f] = filled_data[b, idx + 1, f]
        return filled_data


    def linear_interpolate_torch(self, data, indices, normal_indices, values):
        """1D linear interpolation implemented in PyTorch."""
        delta = (normal_indices[1:] - normal_indices[:-1]).type_as(values)
        delta_values = (values[1:] - values[:-1]) / delta
        
        cumsum_delta = torch.cumsum(delta, dim=0)
        cumsum_delta = torch.hstack([torch.zeros(1, device=data.device), cumsum_delta])
        cumsum_values = torch.cumsum(delta_values, dim=0)
        cumsum_values = torch.hstack([torch.zeros(1, device=data.device), cumsum_values])
        
        left_idx = torch.searchsorted(normal_indices, indices) - 1
        left_idx[left_idx < 0] = 0
        
        left_normal = normal_indices[left_idx]
        left_values = values[left_idx]
        
        slope = delta_values[left_idx]
        slope = torch.nan_to_num(slope)
        
        interpolated_values = left_values + slope * (indices - left_normal)
        return interpolated_values


    def get_normal_indices(self, seq_len, indices):
        all_indices = torch.arange(seq_len, device=indices.device)
        mask = torch.ones_like(all_indices, dtype=torch.bool)
        valid_indices = indices[(indices >= 0) & (indices < seq_len)]
        if valid_indices.numel() > 0:
            mask[valid_indices] = False
        normal_indices = all_indices[mask]
        return normal_indices


    def linear_interpolate(self, data, fill_indices):
        batch_size, seq_len, feature_dim = data.shape
        if type(data) is np.ndarray:
            filled_data = data.copy()
        elif type(data) is torch.Tensor:
            filled_data = data.clone()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        if len(fill_indices) > 0:  # ! Is 'fill_indices' possible to be None?
            for b in range(batch_size):
                for f in range(feature_dim):
                    indices = fill_indices[1][fill_indices[0] == b]
                    if len(indices) > 0:
                        if type(data) is np.ndarray:
                            normal_indices = np.setdiff1d(np.arange(seq_len), indices)
                            if len(normal_indices) == 0:
                                logging.warning(f"No normal indices for batch {b} feature {f} data: {data[b, :, f]}")
                                continue
                            filled_data[b, indices, f] = np.interp(indices, normal_indices, data[b, normal_indices, f])
                        elif type(data) is torch.Tensor:
 
                            normal_indices = self.get_normal_indices(seq_len, indices)
                            if normal_indices.numel() == 0:
                                logging.warning(f"No normal indices for batch {b} feature {f} data: {data[b, :, f]}")
                                return data[b, :, f]
                            
                            x = normal_indices.to(torch.float32)
                            y = data[b, normal_indices, f].to(torch.float32)
                            x_new = indices.to(torch.float32)
                            
                            def interp_torch(x_new, x, y):
                                """Custom 1D linear interpolation in PyTorch."""
                                ind = torch.searchsorted(x, x_new)
                                ind = torch.clamp(ind, 1, x.numel() - 1)
                                lo = ind - 1
                                hi = ind
                                dx = x[hi] - x[lo]
                                dy = y[hi] - y[lo]
                                slope = dy / dx
                                return y[lo] + slope * (x_new - x[lo])
                            
                            interpolated_values = interp_torch(x_new, x, y)
                            filled_data = data.clone()
                            filled_data[b, indices, f] = interpolated_values
                            return filled_data
                        
                        else:
                            raise ValueError(f"Unsupported data type: {type(data)}")
        else:
            filled_data = data
                        
        return filled_data


    def rolling_mean(self, data, fill_indices):
        window_size = 1000  # FIXME: magic number
        if type(data) is np.ndarray:
            filled_data = data.copy()
        elif type(data) is torch.Tensor:
            filled_data = data.clone()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        batch_size, seq_len, feature_dim = data.shape
        for b in range(batch_size):
            for f in range(feature_dim):
                indices = fill_indices[1][fill_indices[0] == b]
                for idx in indices:
                    start = max(0, idx - window_size)
                    end = min(seq_len, idx + window_size + 1)
                    neighbors = data[b, start:end, f]
                    valid_neighbors = neighbors[neighbors != 0]
                    if len(valid_neighbors) > 0:
                        if type(data) is np.ndarray:
                            filled_data[b, idx, f] = np.mean(valid_neighbors)
                        elif type(data) is torch.Tensor:
                            filled_data[b, idx, f] = torch.mean(valid_neighbors)
                        else:
                            raise ValueError(f"Unsupported data type: {type(data)}")
        return filled_data


    def forward_fill(self, data, fill_indices):
        if type(data) is np.ndarray:
            filled_data = data.copy()
        elif type(data) is torch.Tensor:
            filled_data = data.clone()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        batch_size, seq_len, feature_dim = data.shape
        for b in range(batch_size):
            for f in range(feature_dim):
                indices = fill_indices[1][fill_indices[0] == b]
                for idx in indices:
                    if idx > 0:
                        filled_data[b, idx, f] = filled_data[b, idx - 1, f]
                    else:
                        filled_data[b, idx, f] = filled_data[b, idx + 1, f]
        return filled_data


    def backward_fill(self, data, fill_indices):
        if type(data) is np.ndarray:
            filled_data = data.copy()
        elif type(data) is torch.Tensor:
            filled_data = data.clone()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        batch_size, seq_len, feature_dim = data.shape
        for b in range(batch_size):
            for f in range(feature_dim):
                indices = fill_indices[1][fill_indices[0] == b]
                for idx in indices[::-1]:
                    if idx < seq_len - 1:
                        filled_data[b, idx, f] = filled_data[b, idx + 1, f]
                    else:
                        filled_data[b, idx, f] = filled_data[b, idx - 1, f]
        return filled_data
