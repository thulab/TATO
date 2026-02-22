
import logging
import numpy as np
from scipy.interpolate import CubicSpline
from math import ceil

def moving_average_smooth(data):
    window_size = 3
    kernel = np.ones(window_size) / window_size
    smoothed_data = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=1, arr=data)
    return smoothed_data


def ewma_smooth(x):
    alpha = 0.3
    batch, time, feature = x.shape
    smoothed_data = np.zeros_like(x)
    smoothed_data[:, 0, :] = x[:, 0, :]  # Initialize with the first value

    for t in range(1, time):
        smoothed_data[:, t, :] = alpha * x[:, t, :] + (1 - alpha) * smoothed_data[:, t - 1, :]
    return smoothed_data


def fft_denoise(data, percentile=90):
    batch, time, feature = data.shape
    denoised_data = np.zeros_like(data)

    for b in range(batch):
        for f in range(feature):
            # Perform FFT
            fft_coeffs = np.fft.fft(data[b, :, f])
            # Get magnitudes and set coefficients below threshold to zero
            magnitudes = np.abs(fft_coeffs)
            upper_magnitude = np.percentile(magnitudes, percentile)
            fft_coeffs[magnitudes < upper_magnitude] = 0 + 0j
            # Perform inverse FFT
            denoised_data[b, :, f] = np.fft.ifft(fft_coeffs).real
    return denoised_data


def jitter(x):
    factor = 0.03
    x_new = np.zeros(x.shape)
    for i in range(x.shape[0]):
        range_values = np.max(x[i], axis=0) - np.min(x[i], axis=0)
        for j in range(x.shape[2]):
            x_new[i, :, j] = x[i, :, j] + np.random.normal(loc=0., scale=range_values[j] * factor, size=x.shape[1])
    return x_new


def outlier(x, pred_len=None, factor=2):
    x_new = x.copy()
    max_values = np.max(x, axis=1, keepdims=True)
    min_values = np.min(x, axis=1, keepdims=True)
    range_values = max_values - min_values
    for i in range(x.shape[0]):
        for j in range(x.shape[2]):
            time_len = x.shape[1]
            # time_idx = time_len - pred_len - 1

            time_idx = np.random.randint(time_len * 0.6, time_len * 0.99)
            time_idx = min(time_len - pred_len - 1, time_idx)  # Keep the outlier within the sequence

            # time_idx = np.random.randint(time_len * 0.6, time_len * 0.9)
            x_new[i, time_idx, j] = x[i, time_idx, j] - range_values[i, 0, j] * factor
    return x_new


def scaling(x, factor=1.5):
    return x * factor


def window_slice(x, reduce_ratio=0.9):
    target_len = np.ceil(reduce_ratio * x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(low=0, high=x.shape[1] - target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i, :, dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len),
                                       pat[starts[i]:ends[i], dim]).T
    return ret


def time_warp(x, sigma=0.02, knot=4):
    orig_steps = np.arange(x.shape[1])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:, dim], warp_steps[:, dim] * random_warps[i, :, dim])(orig_steps)
            scale = (x.shape[1] - 1) / time_warp[-1]
            ret[i, :, dim] = np.interp(orig_steps, np.clip(scale * time_warp, 0, x.shape[1] - 1), pat[:, dim]).T
    return ret


def magnitude_warp(x, sigma=0.2, knot=4):  # our:10
    orig_steps = np.arange(x.shape[1])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array(
            [CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper

    return ret


def vertical_translation(x):
    factor = np.random.choice([-2, 2])  # FIXME: ensure fairness across params/splits when computing MSE
    # factor = np.random.choice([-0.25, 0.25])  # FIXME: ensure fairness across params/splits when computing MSE
    mean = np.mean(x, axis=1, keepdims=True)
    return x + mean * factor


def horizontal_translation(x, factor=None):
    factor = 1 / 5 if factor is not None else factor
    shift = ceil(x.shape[1] * factor)
    return np.roll(x, shift, axis=1)  # FIXME: consider padding instead of roll?


# def translation_first(x, factor=1):
#     shift = x[0] * factor
#     return x + shift


def scale_up_around_mean(x):
    factor = np.random.choice([2, -2])
    mean = np.mean(x, axis=1, keepdims=True)
    return mean + (x - mean) * factor


def scale_down_around_mean(x):
    factor = np.random.choice([1 / 2, -1 / 2])
    mean = np.mean(x, axis=1, keepdims=True)
    return mean + (x - mean) * factor


def scaling_around_first(x, factor=2):
    # x0 = x[0]
    # return x0 + (x - x0) * factor

    factor = np.random.choice([0.6])
    # Data shape: (batch, time, feature)
    mean = np.mean(x, axis=1, keepdims=True)
    return mean + (x - mean) * factor

def slope_around_mean(x, angle=None):
    # Determine the angle if not provided
    angle = np.random.choice([15, -15]) if angle is None else angle
    # Convert angle to radians and calculate the slope
    radians = np.deg2rad(angle)
    slope = np.tan(radians)
    # Generate time indices
    time = np.arange(x.shape[1]).reshape(1, -1, 1)
    # Calculate the data range
    data_range = np.max(x, axis=1, keepdims=True) - np.min(x, axis=1, keepdims=True)
    # Normalize the slope to the data range
    normalized_slope = slope * (data_range / x.shape[1])
    # Create the slope effect to add to the data
    added = (time * normalized_slope).reshape(1, -1, x.shape[2])
    # Center the added slope effect around the mean
    added = added - np.mean(added, axis=1, keepdims=True)
    # Print the added slope effect for debugging
    # Add the slope effect to the original data
    return x + added


def slope_at_split(x, angle=None):
    # Determine the angle if not provided
    angle = np.random.choice([60, -60]) if angle is None else angle
    split_ratio = np.random.uniform(0.7, 0.9)
    # Convert angle to radians and calculate the slope
    radians = np.deg2rad(angle)
    slope = np.tan(radians)
    # Generate time indices
    time = np.arange(x.shape[1]).reshape(1, -1, 1)
    # Calculate the data range
    data_range = np.max(x, axis=1, keepdims=True) - np.min(x, axis=1, keepdims=True)
    # Normalize the slope to the data range
    normalized_slope = slope * (data_range / x.shape[1])
    # Create the slope effect to add to the data
    added = (time * normalized_slope).reshape(1, -1, 1)
    # Center the added slope effect around the mean
    added[:, :int(split_ratio * x.shape[1]), :] = 0
    # Add the slope effect to the original data

    return x + added


def turn_around_mean(x):
    angle = np.random.choice([100, -100])
    x1 = x
    x2 = slope_around_mean(x1, angle)
    split = np.random.randint(x.shape[1] * 0.7, x.shape[1] * 0.9)
    gap = x2[:, split:split + 1, :] - x1[:, split - 1:split, :]
    x3 = x2 - gap
    return np.concatenate([x1[:, :split], x3[:, split:]], axis=1)


def sin_around_mean(x, amplitude=None, frequency=None):
    # Determine the amplitude and frequency if not provided
    amplitude = np.random.uniform(0.5, 1.5) if amplitude is None else amplitude  # Default amplitude range
    frequency = np.random.uniform(0.1, 1.0) if frequency is None else frequency  # Default frequency range

    # Generate time indices
    time = np.arange(x.shape[1]).reshape(1, -1, 1)

    # Calculate the data range
    data_range = np.max(x, axis=1, keepdims=True) - np.min(x, axis=1, keepdims=True)

    # Normalize the amplitude to the data range
    normalized_amplitude = amplitude * (data_range / 2)

    # Create the sine effect to add to the data
    added = normalized_amplitude * np.sin(2 * np.pi * frequency * time / x.shape[1])

    # Center the added sine effect around the mean
    added = added - np.mean(added, axis=1, keepdims=True)

    # Add the sine effect to the original data
    return x + added


def magnitude_flip(x):
    mean = np.mean(x, axis=1, keepdims=True)
    residual = x - mean
    return mean - residual


class Augmentor:
    def __init__(self, aug_method, mode, pred_len):
        self.org_aug_method_dict = {
            'none': lambda x: x,
            'magnitude_flip': magnitude_flip,
            'time_flip': lambda x: x[:, ::-1, :].copy(),  # Stable way to generate new samples
            'window_slice': window_slice,
            'time_warp': time_warp,
            'magnitude_warp': magnitude_warp,
            'horizontal_translation1': lambda x: horizontal_translation(x, 1 / 5),
            'ewma_smooth': ewma_smooth,
            'jitter': jitter,
            'outlier': lambda x: outlier(x, pred_len, 2),
            'outlier2': lambda x: outlier(x, pred_len, 1),
            # 'outlier3': lambda x: outlier(x, pred_len),
            # 'outlier': lambda x: outlier(x, 192),
            # 'outlier2': lambda x: outlier(x, 96),
            # 'outlier3': lambda x: outlier(x, 48),
            # 'outlier4': lambda x: outlier(x, 24),
            # 'fft_denoise': fft_denoise,
            'turn_around_mean': turn_around_mean,
            'scaling_around_first': scaling_around_first,

        }
        self.mode = mode
        assert mode in ['fix', 'rotate', 'all']
        self.aug_method_dict = None
        if mode == 'fix':
            self.aug_method_dict = {aug_method: self.org_aug_method_dict[aug_method]}
        elif mode == 'rotate':  # In rotate mode, keep the proportion of 'none' at about half
            self.aug_method_dict = self.org_aug_method_dict.copy()
            aug_method_len = len(self.org_aug_method_dict.keys())
            for i in range(aug_method_len):
                self.aug_method_dict[f'none{i}'] = lambda x: x
        elif mode == 'all':
            self.aug_method_dict = self.org_aug_method_dict.copy()
        else:
            raise ValueError(f"Unknown mode: {mode}")
        self.aug_idx = None
        self.reset_aug_method(aug_method)

    def reset_aug_method(self, aug_method):
        assert aug_method in self.aug_method_dict.keys(), f"Unknown augmentation method: {aug_method}"
        self.aug_idx = list(self.aug_method_dict.keys()).index(aug_method)
        logging.info(f"Augmentation method reset to {aug_method}")

    def get_aug_method(self):
        return list(self.aug_method_dict.keys())[self.aug_idx]


    def apply_augmentation(self, data_x, data_y):
        # data_x, data_y: (batch, time, dim)
        assert data_x.ndim == 3 and data_y.ndim == 3, f"Expect 3D inputs, got {data_x.ndim} and {data_y.ndim}"
        assert data_x.shape[0] == data_y.shape[0] and data_x.shape[2] == data_y.shape[2], \
            "batch and feature dims must match between data_x and data_y"
        batch = data_x.shape[0]
        time_x = data_x.shape[1]
        data_full = np.concatenate([data_x, data_y], axis=1)  # (batch, time_x+time_y, dim)

        aug_keys = list(self.aug_method_dict.keys())
        n_methods = len(self.aug_method_dict)
        out_samples = []

        if self.mode == 'fix' or self.mode == 'rotate':
            # modes 'fix' and 'rotate' -> one augmented sample per input sample
            for i in range(batch):
                sample = data_full[i:i + 1]
                method_name = aug_keys[self.aug_idx]
                augmented = self.aug_method_dict[method_name](sample)
                assert augmented.ndim == 3 and augmented.shape == sample.shape, \
                    f"Augmented shape changed from {sample.shape} to {augmented.shape}, method: {method_name}"
                out_samples.append(augmented[0])
                if self.mode == 'rotate':
                    self.aug_idx = (self.aug_idx + 1) % n_methods
        elif self.mode == 'all':
            # each input sample -> produce n_methods augmented samples (batch expands to batch * n_methods)
            for i in range(batch):
                sample = data_full[i:i + 1]  # keep batch dim for augmentors
                for method_name in aug_keys:
                    augmented = self.aug_method_dict[method_name](sample)
                    # normalize to (1, T, D)
                    assert augmented.ndim == 3 and augmented.shape[1:] == sample.shape[1:], \
                        f"Augmented shape mismatch for method {method_name}: got {augmented.shape}, expected {(1,) + sample.shape[1:]}"
                    out_samples.append(augmented[0])
                # update aug_idx per original-sample as requested

        augmented_full = np.stack(out_samples, axis=0)  # (new_batch, time_full, dim)
        # split back to x and y
        augmented_x = augmented_full[:, :time_x, :]
        augmented_y = augmented_full[:, time_x:, :]
        return augmented_x, augmented_y