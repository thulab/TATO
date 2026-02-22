import numpy as np


def get_statistics(history_seq):
    mean = np.mean(history_seq, axis=1, keepdims=True)
    std = np.std(history_seq, axis=1, keepdims=True)
    q1 = np.percentile(history_seq, 25, axis=1, keepdims=True)
    q3 = np.percentile(history_seq, 75, axis=1, keepdims=True)
    median = np.median(feature_data, axis=1, keepdims=True)
    return mean, std, q1, q3, median
