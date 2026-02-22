# clip_utils.py
import numpy as np
import torch
import logging


def my_clip(seq_in, seq_out, nan_inf_clip_factor=None, min_max_clip_factor=None):
    # nan_inf_clip_factor=3, min_max_clip_factor=2 ...
    # mean+1.5IQR-> max+0.25range
    if isinstance(seq_in, np.ndarray):
        max_values = np.max(seq_in, axis=1, keepdims=True)
        min_values = np.min(seq_in, axis=1, keepdims=True)
    elif isinstance(seq_in, torch.Tensor):
        max_values = torch.max(seq_in, dim=1, keepdim=True).values
        min_values = torch.min(seq_in, dim=1, keepdim=True).values
    else:
        raise ValueError(f"Unknown type: {type(seq_in)}")
    range_values = max_values - min_values

    assert nan_inf_clip_factor is not None or min_max_clip_factor is not None, \
        "nan_inf_clip_factor and min_max_clip_factor cannot be both None!"
    

    if isinstance(seq_out, np.ndarray):
        if nan_inf_clip_factor is not None and (np.isnan(seq_out).any() or np.isinf(seq_out).any()):
            max_allowed = max_values + nan_inf_clip_factor * range_values
            min_allowed = min_values - nan_inf_clip_factor * range_values
            logging.info(f"seq_out contains NaN values!!! \n")
            logging.debug(f"seq_out contains NaN values!!!: {seq_out}")
            # seq_out = np.nan_to_num(seq_out, nan=(max_values + min_values) / 2, posinf=max_allowed, neginf=min_allowed)
            seq_out = np.nan_to_num(seq_out, nan=max_allowed, posinf=max_allowed, neginf=min_allowed)  # nan hard punish
    elif isinstance(seq_out, torch.Tensor):
        if nan_inf_clip_factor is not None and (torch.isnan(seq_out).any() or torch.isinf(seq_out).any()):
            max_allowed = max_values + nan_inf_clip_factor * range_values
            min_allowed = min_values - nan_inf_clip_factor * range_values
            nan_count = torch.isnan(seq_out).sum().item()
            inf_count = torch.isinf(seq_out).sum().item()
            # logging.info(f"seq_out contains NaN values!!! \n")
            # logging.debug(f"seq_out contains NaN values!!!: {seq_out}")
            # torch.nan_to_num only accepts Python scalars for nan/posinf/neginf.
            # Here max_allowed/min_allowed are tensors shaped (B,1,C), so we must replace elementwise.
            seq_out = torch.where(torch.isnan(seq_out), max_allowed, seq_out)
            pos_inf_mask = torch.isinf(seq_out) & (seq_out > 0)
            neg_inf_mask = torch.isinf(seq_out) & (seq_out < 0)
            seq_out = torch.where(pos_inf_mask, max_allowed, seq_out)
            seq_out = torch.where(neg_inf_mask, min_allowed, seq_out)

    if min_max_clip_factor is not None:
        max_allowed = max_values + min_max_clip_factor * range_values
        min_allowed = min_values - min_max_clip_factor * range_values
        seq_in_last_values = seq_in[:, -1:, :]
        if (seq_out > max_allowed).any() or (seq_out < min_allowed).any():
            logging.info(f"seq_out out of range!!!: \n")
            logging.debug(f"seq_out out of range!!!: {seq_out}")
            # seq_out = np.clip(seq_out, min_allowed, max_allowed)
            # logging.warning(f"seq_out after cl
            # ipping: {seq_out}")
            # seq_out = smart_clip(seq_out, min_allowed, max_allowed, seq_in_last_values)
            if isinstance(seq_out, torch.Tensor) and torch.is_grad_enabled() and seq_out.requires_grad:
                # Training-time safety: smart_clip is piecewise + in-place + contains divisions.
                # To avoid NaN/Inf and unstable gradients during finetuning, use a stable clamp.
                seq_out = torch.clamp(seq_out, min=min_allowed, max=max_allowed)
            else:
                seq_out = smart_clip(seq_out, min_allowed, max_allowed, seq_in_last_values)
    return seq_out


def smart_clip(seq, min_allowed, max_allowed, seq_in_last_values):
    assert len(seq.shape) == 3, "Input sequence must be 3D: (batch, time, feature)"
    assert min_allowed.shape == max_allowed.shape == (seq.shape[0], 1, seq.shape[2]), \
        "Min and max must have shape (batch, 1, feature)"

    batch, time, feature = seq.shape
    first_elements = seq[:, 0:1, :]  # Preserve the first elements
    # assert np.all(first_elements < max_values) and np.all(first_elements > min_values), \
    #     f"The first elements must be within min and max values: \n" \
    #     f"first_elements:{first_elements}, \nmin_values:{min_values}, \nmax_values:{max_values}"
    if isinstance(seq, np.ndarray):
        if np.any(first_elements > max_allowed) or np.any(first_elements < min_allowed):
            logging.info(f"The first elements must be within min and max allowed!!!\n")
            logging.debug(f"The first elements must be within min and max allowed: \n"
                          f"first_elements:{first_elements}, \nmin_allowed:{min_allowed}, \nmax_allowed:{max_allowed}")
            # return np.clip(seq, min_allowed, max_allowed)
            # FIXME: shift first to match last, then scale within (first, last)
            seq = seq - first_elements + seq_in_last_values
        seq_max_values = np.max(seq, axis=1, keepdims=True)  # Include the first element
        seq_min_values = np.min(seq, axis=1, keepdims=True)  # Include the first element isinstance(seq_in, torch.Tensor):
    elif isinstance(seq, torch.Tensor):
        if torch.any(first_elements > max_allowed) or torch.any(first_elements < min_allowed):
            logging.info(f"The first elements must be within min and max allowed!!!\n")
            logging.debug(f"The first elements must be within min and max allowed: \n"
                          f"first_elements:{first_elements}, \nmin_allowed:{min_allowed}, \nmax_allowed:{max_allowed}")
            # return torch.clamp(seq, min=min_allowed, max=max_allowed)
            # FIXME: shift first to match last, then scale within (first, last)
            seq = seq - first_elements + seq_in_last_values
        seq_max_values = torch.max(seq, dim=1, keepdim=True).values
        seq_min_values = torch.min(seq, dim=1, keepdim=True).values
    else:
        raise ValueError(f"Unknown type: {type(seq)}")
    

    # Apply scaling to the sequences that exceed the max values
    for i in range(batch):
        for j in range(feature):
            tmp_seq = seq[i, :, j]
            first_value = first_elements[i, 0, j]
            max_value = max_allowed[i, 0, j]
            min_value = min_allowed[i, 0, j]
            seq_max_value = seq_max_values[i, 0, j]
            seq_min_value = seq_min_values[i, 0, j]
            if seq_max_value > max_value:
                scale = (max_value - first_value) / (seq_max_value - first_value)
                upper_mask = tmp_seq > first_value
                seq[i, upper_mask, j] = first_value + (seq[i, upper_mask, j] - first_value) * scale
            if seq_min_value < min_value:
                scale = (min_value - first_value) / (seq_min_value - first_value)
                lower_mask = tmp_seq < first_value
                seq[i, lower_mask, j] = first_value + (seq[i, lower_mask, j] - first_value) * scale
    return seq