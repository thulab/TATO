from typing import List, Dict, Any
import matplotlib.pyplot as plt
from scipy import stats
from math import ceil
import pandas as pd
import numpy as np
import logging
import random
import torch
import ast
import os

def set_seed(seed):
    logging.info(f"Set seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Make the computations deterministic on GPU (if applicable)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # scipy.random.seed(seed)
    # sklearn.utils.check_random_state(seed)
    # statsmodels.tools.check_random_state(seed)

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2, axis=(1, 2))) / np.sqrt(np.sum((true - true.mean()) ** 2, axis=(1, 2)))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true), axis=(1, 2))


def MSE(pred, true):
    return np.mean((pred - true) ** 2, axis=(1, 2))


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true), axis=(1, 2))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true), axis=(1, 2))


def NRMSE(pred, true):
    return np.sqrt(MSE(pred, true)) / np.mean(np.abs(true), axis=(1, 2))


def WAPE(pred, true):
    return np.mean(np.abs(pred - true), axis=(1, 2)) / np.mean(np.abs(true), axis=(1, 2))


def metric_five_dict(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    metric_dict = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'MSPE': mspe
    }
    return metric_dict


def cal_metric_stat(metrics_dict: Dict[str, List[float]], 
                              statistics_list: List[str]) -> Dict[str, float]:
    """Compute summary statistics for metric value lists."""
    result = {}

    for stat_name in statistics_list:
        for metric_name, values in metrics_dict.items():
            if not isinstance(values, list) or len(values) == 0:
                raise ValueError(f"Metric {metric_name} has no valid data")

            stat_key = f"{metric_name}_{stat_name}"        
            # tansform to numpy array
            values_array = np.array(values)
    
            try:
                if stat_name == 'mean':
                    result[stat_key] = np.mean(values_array)
                elif stat_name == 'std':
                    result[stat_key] = np.std(values_array, ddof=1)
                elif stat_name == 'median':
                    result[stat_key] = np.median(values_array)
                elif stat_name == 'iqr':
                    q75, q25 = np.percentile(values_array, [75, 25])
                    result[stat_key] = q75 - q25
                elif stat_name == 'max':
                    result[stat_key] = np.max(values_array)
                elif stat_name == 'min':
                    result[stat_key] = np.min(values_array)
                elif stat_name == 'var':
                    result[stat_key] = np.var(values_array, ddof=1)
                elif stat_name == 'range':
                    result[stat_key] = np.max(values_array) - np.min(values_array)
                elif stat_name == 'cv':
                    mean_val = np.mean(values_array)
                    if mean_val != 0:
                        result[stat_key] = np.std(values_array) / mean_val
                    else:
                        result[stat_key] = 0.0
                elif stat_name == 'skew':
                    result[stat_key] = stats.skew(values_array)
                elif stat_name == 'kurtosis':
                    result[stat_key] = stats.kurtosis(values_array)
                else:
                    print(f"Warning: unsupported statistic {stat_name}")
                    
            except Exception as e:
                print(f"Error computing {stat_key}: {e}")
                result[stat_key] = float('nan')

    return result


