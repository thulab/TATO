import logging
import os
from math import floor
from typing import Optional, Tuple, List, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
import torch
from torch.utils.data import Dataset
import itertools
from transformation.library.trimmer import Transformation as Trimmer


# Data directory constant
DATA_LIB = './DATASET/'


class CustomDataset(Dataset):
    """Custom PyTorch Dataset for univariate time series forecasting.
    
    This dataset loads time series data and prepares it for training, validation, or testing.
    It supports sampling a subset of data for faster experimentation.
    
    Attributes:
        dataset: Instance of dataset class (e.g., EttHour, Weather)
        mode: One of 'train', 'val', or 'test'
        target_column: Target column name (default: 'OT')
        max_seq_len: Maximum sequence length for input
        pred_len: Prediction length (forecast horizon)
        augmentor: Optional data augmentation module
        indices: List of valid indices for the specified mode
    """
    
    def __init__(self, dataset: Any, mode: str, target_column: str, max_seq_len: int, 
                 pred_len: int, augmentor: Optional[Any] = None, num_sample: Union[str, int] = 'all'):
        """Initialize the CustomDataset.
        
        Args:
            dataset: Dataset instance containing the time series data
            mode: Data split mode ('train', 'val', or 'test')
            target_column: Name of the target column to predict
            max_seq_len: Length of input sequence (history)
            pred_len: Length of prediction sequence (forecast horizon)
            augmentor: Optional augmentation module for data augmentation
            num_sample: Number of samples to use ('all' for all samples, or integer for subset)
        """
        self.dataset = dataset
        self.mode = mode
        self.target_column = target_column
        self.max_seq_len = max_seq_len
        self.pred_len = pred_len
        self.augmentor = augmentor
        
        # Get all available indices for this mode
        self.indices = self.dataset.get_available_idx_list(mode, max_seq_len, pred_len)
        
        # Sample a subset if requested (for faster experimentation)
        if num_sample != 'all':
            # Convert num_sample to integer if it's a string representation
            sample_count = int(num_sample) if isinstance(num_sample, str) else num_sample
            selected_indices_indexes = np.linspace(0, len(self.indices) - 1, sample_count).astype(int)
            self.indices = [self.indices[i] for i in selected_indices_indexes]
        
        logging.info(f'{mode} dataset size: {len(self)}')

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[int, np.ndarray, np.ndarray]:
        """Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple containing:
                - real_idx: Actual index in the raw dataset
                - history: Input sequence of shape (max_seq_len, 1)
                - label: Target sequence of shape (pred_len, 1)
        """
        # Get the actual index in the raw dataset
        real_idx = self.indices[idx]
        
        # Retrieve history and label sequences
        history_with_label = self.dataset.get_history_with_label(
            self.target_column, self.mode, real_idx, self.max_seq_len, self.pred_len)
        
        # Reshape to (sequence_length, 1) for univariate forecasting
        _history_with_label = history_with_label.reshape((-1, 1))  # shape: (max_seq_len + pred_len, 1)
        
        # Note: Data augmentation is currently disabled
        # aug_method = 'none'
        # if self.mode != 'test':
        #     aug_method = self.augmentor.get_aug_method()
        #     _history_with_label = self.augmentor.apply_augmentation(_history_with_label)
        
        # Validate shape
        assert _history_with_label.shape[1] == 1 and _history_with_label.shape[0] == self.max_seq_len + self.pred_len, \
            f'Invalid history_with_label shape: {_history_with_label.shape}'
        
        # Split into history (input) and label (target)
        history = _history_with_label[:-self.pred_len, :]   # First max_seq_len points
        label = _history_with_label[-self.pred_len:, :]     # Last pred_len points
        
        return real_idx, history, label

class CustomDatasetCov(Dataset):
    """Custom PyTorch Dataset for time series forecasting with covariates.
    
    This dataset extends CustomDataset to include covariate features alongside
    the target time series. Covariates are additional features that may help
    improve forecasting accuracy.
    
    Attributes:
        dataset: Instance of dataset class
        mode: One of 'train', 'val', or 'test'
        target_column: Target column name
        max_seq_len: Maximum sequence length for input
        pred_len: Prediction length
        augmentor: Data augmentation module
        indices: List of valid indices for the specified mode
    """
    
    def __init__(self, dataset: Any, mode: str, target_column: str, max_seq_len: int, 
                 pred_len: int, augmentor: Any, num_sample: Union[str, int]):
        """Initialize the CustomDatasetCov.
        
        Args:
            dataset: Dataset instance containing the time series data
            mode: Data split mode ('train', 'val', or 'test')
            target_column: Name of the target column to predict
            max_seq_len: Length of input sequence (history)
            pred_len: Length of prediction sequence (forecast horizon)
            augmentor: Augmentation module for data augmentation
            num_sample: Number of samples to use ('all' or integer)
        """
        self.dataset = dataset
        self.mode = mode
        self.target_column = target_column
        self.max_seq_len = max_seq_len
        self.pred_len = pred_len
        self.augmentor = augmentor
        
        # Get all available indices for this mode
        self.indices = self.dataset.get_available_idx_list(mode, max_seq_len, pred_len)
        
        # Sample a subset if requested
        if num_sample != 'all':
            # Convert num_sample to integer if it's a string representation
            sample_count = int(num_sample) if isinstance(num_sample, str) else num_sample
            selected_indices_indexes = np.linspace(0, len(self.indices) - 1, sample_count).astype(int)
            self.indices = [self.indices[i] for i in selected_indices_indexes]
        
        logging.info(f'{mode} dataset size: {len(self)}')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        history_with_label, cov_history_with_label = self.dataset.get_cov_history_with_label(
            self.target_column, self.mode, real_idx, self.max_seq_len, self.pred_len)
        _history_with_label = history_with_label.reshape((1, -1, 1))  # batch time feature
        S,C = cov_history_with_label.shape
        _cov_history_with_label = cov_history_with_label.reshape((1, S, C))
        aug_method = 'none'
        if self.mode != 'test':
            aug_method = self.augmentor.get_aug_method()
            _history_with_label = self.augmentor.apply_augmentation(_history_with_label)
        assert _history_with_label.shape[2] == 1 and _history_with_label.shape[1] == self.max_seq_len + self.pred_len, \
            f'Invalid history_with_label shape: {_history_with_label.shape}'
        history, label = _history_with_label[:, :-self.pred_len, :], _history_with_label[:, -self.pred_len:, :]
        cov_history, cov_label = _cov_history_with_label[:, :-self.pred_len, :], _cov_history_with_label[:, -self.pred_len:, :]
        return real_idx, aug_method, history, label, cov_history, cov_label


def split721(train_len, val_len, test_len):
    total_len = train_len + val_len + test_len
    ratios = [0.7, 0.2, 0.1]
    train_len = floor(total_len * ratios[0])
    val_len = floor(total_len * ratios[1])
    test_len = total_len - train_len - val_len
    return train_len, val_len, test_len


def get_dataset(data_name, fast_split=None):
    fast_split = fast_split if fast_split is not None else False

    if data_name == 'ETTh1':
        dataset = EttHour(DATA_LIB + 'ETT-small', 'ETTh1.csv', fast_split)
    elif data_name == 'ETTh2':
        dataset = EttHour(DATA_LIB + 'ETT-small/', 'ETTh2.csv', fast_split)
    elif data_name == 'ETTm1':
        dataset = EttMinute(DATA_LIB + 'ETT-small/', 'ETTm1.csv', fast_split)
    elif data_name == 'ETTm2':
        dataset = EttMinute(DATA_LIB + 'ETT-small/', 'ETTm2.csv', fast_split)
    elif data_name == 'Exchange' or data_name == 'exchange_rate':
        dataset = Exchange(DATA_LIB + 'exchange_rate/', 'exchange_rate.csv', fast_split)
    elif data_name == 'Weather' or data_name == 'weather':
        dataset = Weather(DATA_LIB + 'weather/', 'weather.csv', fast_split)
    elif data_name == 'Electricity' or data_name == 'electricity':
        dataset = Electricity(DATA_LIB + 'electricity/', 'electricity.csv', fast_split)
    elif data_name == 'Traffic' or data_name == 'traffic':
        dataset = Traffic(DATA_LIB + 'traffic/', 'traffic.csv', fast_split)
    else:
        raise ValueError(f"Unknown data_name: {data_name}")
    return dataset


class MyDataBase:
    column_names = ['OT']
    def __init__(self, root_path, data_path, train_start, train_end, val_start, val_end, test_start, test_end):
        self.root_path = root_path
        self.data_path = data_path
        self.train_start, self.train_end = train_start, train_end
        self.val_start, self.val_end = val_start, val_end
        self.test_start, self.test_end = test_start, test_end
        self.train_len = self.train_end - self.train_start
        self.val_len = self.val_end - self.val_start
        self.test_len = self.test_end - self.test_start
        self.total_len = self.train_len + self.val_len + self.test_len

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw = df_raw.iloc[:, 1:]
        self.df_data = df_raw
        self.feature_dim = 8 if 'exchange' in self.root_path else 21 if 'weather' in self.root_path else 7
        self.indices_dim = list(range(0, self.feature_dim))
        if 'electricity' in self.data_path:
            self.indices_dim = [314, 315, 316, 317, 318, 319, 320]
        elif 'traffic' in self.data_path:
            self.indices_dim = [855, 856, 857, 858, 859, 860, 861]
        if "shanxi" in self.data_path: # xiexin
            self.feature_dim = len([14, 15, 16, 17, 5, 10, 6, 11, 3, 4, 7, 40, 41, 42, 85, 80, 33, 84, 66, 68, 56, 60, 2, 64, 1])
            self.indices_dim = [14, 15, 16, 17, 5, 10, 6, 11, 3, 4, 7, 40, 41, 42, 85, 80, 33, 84, 66, 68, 56, 60, 2, 64, 1]
        print(f"feature_dim: {self.feature_dim}")

        self.np_data_dict = {col: np.array(df_raw[col].values).reshape(-1) for col in df_raw.columns}

        self.scalers = {}
        self.mul_scalers = {}
        # self.train_statistics = {}

    def get_history_with_label(self, target, flag, real_idx, max_seq_len, pred_len):
        # flag: train, test, val
        # target: column:   HUFL	HULL	MUFL	MULL	LUFL	LULL	OT
        assert flag in ['train', 'val', 'test'], \
            f'Invalid flag: {flag}'
        assert target in self.column_names, \
            f'Invalid target: {target}'
        # real_idx = idx + self.__getattribute__(flag + '_start')
        assert real_idx - max_seq_len >= 0, \
            f'Invalid real_idx: {real_idx}, max_seq_len: {max_seq_len}'
        assert real_idx + pred_len < self.__getattribute__(flag + '_end'), \
            f'Invalid real_idx: {real_idx}, pred_len: {pred_len}'
        # history = self.np_data_dict[target][real_idx - max_seq_len: real_idx]
        # label = self.np_data_dict[target][real_idx: real_idx + pred_len]
        history_with_label = self.np_data_dict[target][real_idx - max_seq_len: real_idx + pred_len]
        
        return history_with_label
    
    def get_cov_history_with_label(self, target, flag, real_idx, max_seq_len, pred_len):
        # flag: train, test, val
        # target: column:   HUFL	HULL	MUFL	MULL	LUFL	LULL	OT
        assert flag in ['train', 'val', 'test'], \
            f'Invalid flag: {flag}'
        assert target in self.column_names, \
            f'Invalid target: {target}'
        # real_idx = idx + self.__getattribute__(flag + '_start')
        assert real_idx - max_seq_len >= 0, \
            f'Invalid real_idx: {real_idx}, max_seq_len: {max_seq_len}'
        assert real_idx + pred_len < self.__getattribute__(flag + '_end'), \
            f'Invalid real_idx: {real_idx}, pred_len: {pred_len}'
        # history = self.np_data_dict[target][real_idx - max_seq_len: real_idx]
        # label = self.np_data_dict[target][real_idx: real_idx + pred_len]
        history_with_label = self.np_data_dict[target][real_idx - max_seq_len: real_idx + pred_len]
        cov_history_with_label = self.df_data.values[real_idx - max_seq_len: real_idx + pred_len, :-1]
        return history_with_label, cov_history_with_label

    def get_available_idx_list(self, mode, max_seq_len, pred_len):
        assert mode in ['train', 'val', 'test'], \
            f'Invalid flag: {mode}'
        assert max_seq_len <= self.train_len / 2, \
            f'Invalid max_seq_len: {max_seq_len}, train_len: {self.train_len}'
        if mode == 'train':
            start = self.train_start + max_seq_len
            end = self.train_end - pred_len
            # start = end - self.num_sample if end - start > self.num_sample else start
        elif mode == 'val':
            start = self.val_start
            end = self.val_end - pred_len
        else:
            start = self.test_start
            end = self.test_end - pred_len
        assert start < end, \
            f'Invalid start: {start}, end: {end}'
        # real split idx in raw dataset!
        return list(range(start, end))

    def get_mode_scaler(self, mode, method, target):
        assert target in self.column_names
        assert mode in ['train', 'val', 'test']
        # assert method in ['none', 'standard', 'minmax', 'maxabs', 'robust']
        assert method in ['none', 'standard', 'robust']

        if method == 'none':
            return None

        if (mode, target, method) in self.scalers:
            return self.scalers[(mode, target, method)]

        # Create and fit new scaler only if not already computed
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError('Invalid scaler method')
        # Fit scaler to training data for the specified target
        if mode == 'train':
            mode_data = self.np_data_dict[target][self.train_start:self.train_end].reshape(-1, 1)
        elif mode == 'val':
            mode_data = self.np_data_dict[target][self.val_start:self.val_end].reshape(-1, 1)
        elif mode == 'test':
            mode_data = self.np_data_dict[target][self.test_start:self.test_end].reshape(-1, 1)
        else:
            raise ValueError('Invalid mode')
        scaler.fit(mode_data)#! Is fitting scaler on validate and test data applicable?
        # Store the scaler for future use
        self.scalers[(mode, target, method)] = scaler
        return scaler

    def get_scaler(self, method, target):
        assert target in self.column_names
        assert method in ['none', 'standard', 'robust']

        if method == 'none':
            return None

        if (target, method) in self.scalers:
            return self.scalers[(target, method)]

        # Create and fit new scaler only if not already computed
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError('Invalid scaler method')
        # Fit scaler to training data
        mode_data = self.np_data_dict[target][self.train_start:self.train_end].reshape(-1, 1)
        scaler.fit(mode_data)#! Is fitting scaler on validate and test data applicable?
        # Store the scaler for future use
        self.scalers[(target, method)] = scaler
        return scaler

class EttMinute(MyDataBase):
    # column_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    column_names = ['OT']
    
    def __init__(self, root_path, data_path, fast_test_split):
        train_len, val_len, test_len = 34465, 11521, 11521
        if fast_test_split:
            train_len, val_len, test_len = split721(train_len, val_len, test_len)

        train_start, train_end = 0, train_len
        val_start, val_end = train_len, train_len + val_len
        test_start, test_end = train_len + val_len, train_len + val_len + test_len
        super(EttMinute, self).__init__(root_path, data_path,
                                        train_start, train_end,
                                        val_start, val_end,
                                        test_start, test_end)


class EttHour(MyDataBase):
    # column_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    column_names = ['OT']

    def __init__(self, root_path, data_path, fast_test_split):
        train_len, val_len, test_len = 8545, 2881, 2881
        if fast_test_split:
            train_len, val_len, test_len = split721(train_len, val_len, test_len)

        train_start, train_end = 0, train_len
        val_start, val_end = train_len, train_len + val_len
        test_start, test_end = train_len + val_len, train_len + val_len + test_len
        super(EttHour, self).__init__(root_path, data_path,
                                      train_start, train_end,
                                      val_start, val_end,
                                      test_start, test_end)


class Exchange(MyDataBase):
    # column_names = ['0', '1', '2', '3', '4', '5', '6', 'OT']
    column_names = ['OT']

    def __init__(self, root_path, data_path, fast_test_split):
        train_len, val_len, test_len = 4343, 1442, 1442
        if fast_test_split:
            train_len, val_len, test_len = split721(train_len, val_len, test_len)

        train_start, train_end = 0, train_len
        val_start, val_end = train_len, train_len + val_len
        test_start, test_end = train_len + val_len, train_len + val_len + test_len
        super(Exchange, self).__init__(root_path, data_path,
                                       train_start, train_end,
                                       val_start, val_end,
                                       test_start, test_end)


class Weather(MyDataBase):
    column_names = ['OT']

    def __init__(self, root_path, data_path, fast_test_split, clean=False):
        train_len, val_len, test_len = 36792, 5271, 10540
        if fast_test_split:
            train_len, val_len, test_len = split721(train_len, val_len, test_len)

        train_start, train_end = 0, train_len
        val_start, val_end = train_len, train_len + val_len
        test_start, test_end = train_len + val_len, train_len + val_len + test_len
        super(Weather, self).__init__(root_path, data_path,
                                      train_start, train_end,
                                      val_start, val_end,
                                      test_start, test_end)


class Traffic(MyDataBase):
    column_names = ['OT']

    def __init__(self, root_path, data_path, fast_test_split):
        train_len, val_len, test_len = 12185, 1757, 3509
        if fast_test_split:
            train_len, val_len, test_len = split721(train_len, val_len, test_len)

        train_start, train_end = 0, train_len
        val_start, val_end = train_len, train_len + val_len
        test_start, test_end = train_len + val_len, train_len + val_len + test_len
        super(Traffic, self).__init__(root_path, data_path,
                                      train_start, train_end,
                                      val_start, val_end,
                                      test_start, test_end)


class Electricity(MyDataBase):
    # 0  1	2	3	4	5	6	OT
    # column_names = ['0', '1', '2', '3', '4', '5', '6', 'OT']
    column_names = ['OT']

    def __init__(self, root_path, data_path, fast_test_split):
        # Exchange (1000, 1000, 1000)
        train_len, val_len, test_len = 18317, 2633, 5261
        if fast_test_split:
            train_len, val_len, test_len = split721(train_len, val_len, test_len)

        train_start, train_end = 0, train_len
        val_start, val_end = train_len, train_len + val_len
        test_start, test_end = train_len + val_len, train_len + val_len + test_len
        super(Electricity, self).__init__(root_path, data_path,
                                          train_start, train_end,
                                          val_start, val_end,
                                          test_start, test_end)


class Xiexin(MyDataBase):
    column_names = ['OT']
    def __init__(self, root_path, data_path, fast_test_split):
        train_len, val_len, test_len = 8700, 2900, 2900
        if fast_test_split:
            train_len, val_len, test_len = split721(train_len, val_len, test_len)

        train_start, train_end = 58674, 58674 + train_len
        val_start, val_end = train_len, train_len + val_len
        test_start, test_end = train_len + val_len, train_len + val_len + test_len
        super(Xiexin, self).__init__(root_path, data_path,
                                          train_start, train_end,
                                          val_start, val_end,
                                          test_start, test_end)

if __name__ == '__main__':
    # dataset = get_dataset('Electricity', True)
    # dataset = get_dataset('Weather', True)
    # dataset = get_dataset('ETTm2', True)
    dataset = get_dataset('Traffic', True)
    import matplotlib
    import matplotlib.pyplot as plt


    def is_pycharm():
        for key, value in os.environ.items():
            if key == "PYCHARM_HOSTED":
                print(f"PYCHARM_HOSTED={value}")
                return True


    matplotlib.use('TkAgg') if is_pycharm() else None

    fig, axs = plt.subplots(4, 1, figsize=(20, 10))
    axs[0].plot(dataset.np_data_dict['OT'])
    axs[1].plot(dataset.np_data_dict['OT'][dataset.train_start:dataset.train_end])
    axs[2].plot(dataset.np_data_dict['OT'][dataset.val_start:dataset.val_end])
    axs[3].plot(dataset.np_data_dict['OT'][dataset.test_start:dataset.test_end])
    plt.show()
    # max_seq_len = 12 * 30 * 24
    # max_pred_len = 96
    # for idx in dataset.get_available_idx_list('train', max_seq_len, max_pred_len):
    #     max_seq, label = dataset.get_max_seq_and_label('OT', 'train', idx, max_seq_len, max_pred_len)
    #     print(max_seq.shape, label.shape)
    #     break
    # scaler = dataset.get_scaler('standard', 'OT')
    # max_seq = scaler.transform(max_seq.reshape(-1, 1)).reshape(-1)
    # label = scaler.transform(label.reshape(-1, 1)).reshape(-1)
    # print(max_seq.shape, label.shape)
    # max_seq = scaler.inverse_transform(max_seq.reshape(-1, 1)).reshape(-1)
    # label = scaler.inverse_transform(label.reshape(-1, 1)).reshape(-1)
    # print(max_seq.shape, label.shape)
    # print(max_seq, label)
    # pass
