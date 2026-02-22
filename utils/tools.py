from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import periodogram
from statsmodels.tsa.ar_model import AutoReg
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
import itertools


class LargeScheduler:
    """Copied from Timer/utils/tools.py (for local adjustment)."""

    def __init__(self, args, optimizer) -> None:
        super().__init__()
        self.learning_rate = args.learning_rate
        self.decay_fac = getattr(args, 'decay_fac', 0.75)
        self.lradj = getattr(args, 'lradj', 'type1')
        self.optimizer = optimizer
        self.args = args

    def schedule_epoch(self, epoch: int):
        if self.lradj == 'type1':
            lr_adjust = {epoch: self.learning_rate if epoch < 3 else self.learning_rate * (0.9 ** ((epoch - 3) // 1))}
        elif self.lradj == 'type2':
            lr_adjust = {epoch: self.learning_rate * (self.decay_fac ** ((epoch - 1) // 1))}
        elif self.lradj == 'type4':
            lr_adjust = {epoch: self.learning_rate * (self.decay_fac ** ((epoch) // 1))}
        else:
            return

        if epoch in lr_adjust:
            lr = lr_adjust[epoch]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    """Copied from Timer/utils/tools.py (for local adjustment).

    Stores best model weights in-memory as `best_model`.
    """

    def __init__(self, patience=3, verbose=False, delta=0, save=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save = save
        self.best_model = None

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        if self.save:
            torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))
        self.best_model = model.state_dict()
        self.val_loss_min = val_loss


def save_imp(metric_list, save_dir):    
    first_row = metric_list[0]  # origin param's result
    second_row = metric_list[1]  # TATO's param's result

    improvement_data = {}
    
    for name in first_row.keys():
        ori = first_row[name]
        new = second_row[name]
        
        improvement_pct = (ori - new) / (ori if ori != 0 else 1e-5) * 100 
        
        improvement_data[f'imp_{name}'] = improvement_pct
    
    if improvement_data:
        filen = f"metric_imp.csv"
        improvement_path = os.path.join(save_dir, filen)
        
        improvement_df = pd.DataFrame([improvement_data])
        improvement_df.to_csv(improvement_path, index=False, encoding='utf-8')
        
    return None

def save_as_csv(param_scores, save_dir, mode):
    csv_filename = f"{mode}_rank.csv"
    csv_path = os.path.join(save_dir, csv_filename)
    print("-----=====-----")

    data_rows = []
    for i, item in enumerate(param_scores):
        try:
            print(item)
            param_dict = ast.literal_eval(item['param_str'])
            row_data = {
                'rank': i + 1,
                'final_rank': item['final_rank'] if item.get('final_rank') else None
            }
            
            for param_key, param_value in param_dict.items():
                row_data[f'{param_key}'] = param_value
            
            for metric_name, value in item['original_metrics'].items():
                row_data[f'{metric_name}'] = value

            data_rows.append(row_data)
        except Exception as e:
            print(f"Error saving parameter {i + 1} to CSV: {e}")
            continue

    if data_rows:
        df = pd.DataFrame(data_rows)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"Saved results to CSV: {csv_path}")
        return csv_path
    return None


def _freeze_for_hash(obj: Any) -> Any:
    if isinstance(obj, dict):
        return tuple(sorted((k, _freeze_for_hash(v)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze_for_hash(v) for v in obj)
    return obj


def _make_param_dict_unique(param_dict_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique_list = []
    for d in param_dict_list:
        key = _freeze_for_hash(d)
        if key in seen:
            continue
        seen.add(key)
        unique_list.append(d)
    return unique_list


def pareto_select(
    paroto_dicts: Dict[str, Dict[str, Any]],
    metric_weight_dict: Dict[str, float],
    multi_pareto_mode: bool = False,
    metrics_for_pareto_must: List[str] | None = None,
    top_k: int = 5,
    save_dir: str = './results',
    mode: str = 'train',
    save: bool = False,
) -> List[Dict[str, Any]]:
    """Select top parameter sets using weighted ranks and metric-subset aggregation."""

    metric_names = list(metric_weight_dict.keys())
    if metrics_for_pareto_must is None:
        metrics_for_pareto_must = list(metric_names)

    missing = [m for m in metrics_for_pareto_must if m not in metric_weight_dict]
    if missing:
        raise ValueError(f"metrics_for_pareto_must contains unknown metrics: {missing}")

    # Compatibility mode: when multi_pareto_mode=False, keep EXACT behavior with topk_by_wrank
    # (same ranking implementation and tie handling).
    if multi_pareto_mode is False:
        metric_names_for_rank = list(metrics_for_pareto_must)

        param_scores = []
        metrics_for_compare = []

        for param_str, metrics_dict in paroto_dicts.items():
            try:
                metric_values = {}
                for metric_name in metric_names_for_rank:
                    if metric_name in metrics_dict:
                        values = metrics_dict[metric_name]
                        if isinstance(values, list) and len(values) > 0:
                            metric_values[metric_name] = np.mean(values)
                        else:
                            metric_values[metric_name] = values if not isinstance(values, list) else values[0]
                    else:
                        metric_values[metric_name] = float('inf')

                param_scores.append({
                    'param_str': param_str,
                    'metric_values': metric_values,
                    'original_metrics': metrics_dict,
                })

                if mode == 'test':
                    metrics_for_compare.append(metrics_dict)
            except Exception as e:
                print(f"Error processing params {param_str}: {e}")
                continue

        if mode == 'test':
            if len(metrics_for_compare) >= 2:
                save_imp(metrics_for_compare, save_dir)
            else:
                raise ValueError("Error")

        if not param_scores:
            print("Warning: no valid parameter combinations")
            return []

        for metric_name in metric_names_for_rank:
            metric_values = [item['metric_values'][metric_name] for item in param_scores]
            ranks = np.array(metric_values).argsort().argsort() + 1
            weight = metric_weight_dict[metric_name]
            weighted_ranks = ranks * weight
            for i, item in enumerate(param_scores):
                if 'weighted_ranks' not in item:
                    item['weighted_ranks'] = []
                item['weighted_ranks'].append(weighted_ranks[i])

        total_weight = sum(metric_weight_dict[m] for m in metric_names_for_rank)
        for item in param_scores:
            if 'weighted_ranks' in item and len(item['weighted_ranks']) > 0:
                item['final_rank'] = sum(item['weighted_ranks']) / total_weight
            else:
                item['final_rank'] = float('inf')

        if mode != 'test' and mode != 'finetune':
            param_scores.sort(key=lambda x: x['final_rank'])

        if save:
            save_as_csv(param_scores[:min(top_k, len(param_scores))], save_dir, mode)

        top_k_params = []
        for item in param_scores[:min(top_k, len(param_scores))]:
            try:
                top_k_params.append(ast.literal_eval(item['param_str']))
            except (ValueError, SyntaxError) as e:
                print(f"Failed to parse params: {item['param_str']}, error: {e}")
                continue

        return top_k_params

    print(f"Start processing {len(paroto_dicts)} parameter combinations...")
    print(f"metric_weight_dict={metric_weight_dict}")
    print(f"multi_pareto_mode={multi_pareto_mode}, metrics_for_pareto_must={metrics_for_pareto_must}, top_k={top_k}")

    param_scores: List[Dict[str, Any]] = []
    metrics_for_compare = []

    for param_str, metrics_dict in paroto_dicts.items():
        try:
            metric_values: Dict[str, float] = {}
            for metric_name in metric_names:
                if metric_name not in metrics_dict:
                    metric_values[metric_name] = float('inf')
                    continue

                values = metrics_dict[metric_name]
                if isinstance(values, list):
                    metric_values[metric_name] = float(np.mean(values)) if len(values) > 0 else float('inf')
                else:
                    metric_values[metric_name] = float(values)

            param_scores.append({
                'param_str': param_str,
                'metric_values': metric_values,
                'original_metrics': metrics_dict,
            })

            if mode == 'test':
                metrics_for_compare.append(metrics_dict)
        except Exception as e:
            print(f"Error processing params {param_str}: {e}")

    if mode == 'test':
        if len(metrics_for_compare) >= 2:
            save_imp(metrics_for_compare, save_dir)
        else:
            raise ValueError("test mode requires at least two results to compute improvement")

    if not param_scores:
        print("Warning: no valid parameter combinations")
        return []

    for metric_name in metric_names:
        metric_values = [item['metric_values'][metric_name] for item in param_scores]
        ranks = pd.Series(metric_values).rank(method='average', ascending=True).to_numpy()
        weight = float(metric_weight_dict.get(metric_name, 1.0))
        weighted_ranks = ranks * weight
        for i, item in enumerate(param_scores):
            item.setdefault('rank_by_metric', {})
            item['rank_by_metric'][metric_name] = float(weighted_ranks[i])

    if multi_pareto_mode is True:
        metric_names_list = [
            list(combo)
            for length in range(1, len(metric_names) + 1)
            for combo in itertools.combinations(metric_names, length)
        ]
        must_set = set(metrics_for_pareto_must)
        filtered = []
        for cur_metric_names in metric_names_list:
            cur_set = set(cur_metric_names)
            if must_set.issubset(cur_set):
                filtered.append(cur_metric_names)
            elif cur_set.issubset(must_set):
                filtered.append(cur_metric_names)
        metric_names_list = filtered
    else:
        metric_names_list = [list(metrics_for_pareto_must)]

    if not metric_names_list:
        print("Warning: no metric combinations available after filtering")
        return []

    all_best_param_dict_list: List[Dict[str, Any]] = []
    ranked_for_save: List[Dict[str, Any]] | None = None

    for cur_metric_names in metric_names_list:
        total_weight = sum(float(metric_weight_dict[m]) for m in cur_metric_names)
        if total_weight <= 0:
            raise ValueError(f"total_weight <= 0 for metrics={cur_metric_names}")

        ranked_list = []
        for item in param_scores:
            final_rank = sum(item['rank_by_metric'][m] for m in cur_metric_names) / total_weight
            ranked_list.append({**item, 'final_rank': float(final_rank)})

        if mode != 'test' and mode != 'finetune':
            ranked_list.sort(key=lambda x: x['final_rank'])

        selected = ranked_list[: min(top_k, len(ranked_list))]

        if cur_metric_names == metric_names_list[-1]:
            ranked_for_save = selected

        for sel in selected:
            try:
                all_best_param_dict_list.append(ast.literal_eval(sel['param_str']))
            except (ValueError, SyntaxError) as e:
                print(f"Failed to parse params: {sel['param_str']}, error: {e}")

    unique_params = _make_param_dict_unique(all_best_param_dict_list)

    if save and ranked_for_save is not None:
        save_as_csv(ranked_for_save, save_dir, mode)

    print(f"len(all_best_param_dict_list)={len(all_best_param_dict_list)}, len(unique)={len(unique_params)}")
    return unique_params


def topk_by_wrank(paroto_dicts: Dict[str, Dict[str, List[float]]], 
                                   metric_weight_dict: Dict[str, float],
                                   top_k: int = 5, save_dir: str = './results', 
                                   mode: str = 'train', save: bool = False) -> List[Dict[str, Any]]:
    """Select top-k parameter sets by weighted ranking across metrics."""

    param_scores = []
    metric_names = list(metric_weight_dict.keys())
    
    print(f"Start processing {len(paroto_dicts)} parameter combinations...")
    print(f"Metric weights: {metric_weight_dict}")

    metrics_for_compare = []
    
    for param_str, metrics_dict in paroto_dicts.items():
        try:
            metric_values = {}
            for metric_name in metric_names:
                if metric_name in metrics_dict:
                    values = metrics_dict[metric_name]
                    if isinstance(values, list) and len(values) > 0:
                        metric_values[metric_name] = np.mean(values)
                    else:
                        metric_values[metric_name] = values if not isinstance(values, list) else values[0]
                else:
                    metric_values[metric_name] = float('inf')
            
            param_scores.append({
                'param_str': param_str,
                'metric_values': metric_values,
                'original_metrics': metrics_dict
            })

            if mode == 'test':
                metrics_for_compare.append(metrics_dict) # metrics_dict

        except Exception as e:
            print(f"Error processing params {param_str}: {e}")
            continue
    
    if mode == 'test':
        if len(metrics_for_compare) >= 2:
            save_imp(metrics_for_compare, save_dir)
        else:
            raise ValueError("Error")
    
    if not param_scores:
        print("Warning: no valid parameter combinations")
        return []
    
    for metric_name in metric_names:
        metric_values = [item['metric_values'][metric_name] for item in param_scores]
        
        ranks = np.array(metric_values).argsort().argsort() + 1  # ranks: [1, 2, 3, ...]
        
        weight = metric_weight_dict[metric_name]
        weighted_ranks = ranks * weight
        
        for i, item in enumerate(param_scores):
            if f'weighted_ranks' not in item:
                item['weighted_ranks'] = []
            item['weighted_ranks'].append(weighted_ranks[i])
    
    total_weight = sum(metric_weight_dict.values())
    
    for item in param_scores:
        if 'weighted_ranks' in item and len(item['weighted_ranks']) > 0:
            final_rank = sum(item['weighted_ranks']) / total_weight
            item['final_rank'] = final_rank
        else:
            item['final_rank'] = float('inf')
    
    if mode != 'test' and mode != 'finetune':
        param_scores.sort(key=lambda x: x['final_rank'])

    if save:
        save_as_csv(param_scores[:min(top_k, len(param_scores))], save_dir, mode)
    
    top_k_params = []
    print(f"\n=== Top-{top_k} Parameter Ranking ===")
    print("Rule: lower score is better")
    print(f"{'Rank':<4} {'Score':<12} {'Params'}")
    print("-" * 60)
    
    for i, item in enumerate(param_scores[:min(top_k, len(param_scores))]):
        try:
            param_dict = ast.literal_eval(item['param_str'])
            final_rank = item['final_rank']
            
            print(f"{i+1:<4} {final_rank:<12.6f} {param_dict}")
            top_k_params.append(param_dict)
            
            if i < 3:
                print("     Metrics: ", end="")
                for metric_name in metric_names:
                    value = item['metric_values'].get(metric_name, 'N/A')
                    print(f"{metric_name}={value:.6f} ", end="")
                print()
                
        except (ValueError, SyntaxError) as e:
            print(f"Failed to parse params: {item['param_str']}, error: {e}")
            continue
    
    print(f"\nSelected {len(top_k_params)} best parameter sets")
    return top_k_params


def find_pareto_frontier(param_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find Pareto-optimal (non-dominated) parameter sets."""
    if not param_scores:
        return []
    
    n_samples = len(param_scores)
    metric_names = list(param_scores[0]['metric_values'].keys())
    
    metrics_matrix = np.zeros((n_samples, len(metric_names)))
    for i, item in enumerate(param_scores):
        for j, metric_name in enumerate(metric_names):
            metrics_matrix[i, j] = item['metric_values'][metric_name]
    
    pareto_indices = []
    
    for i in range(n_samples):
        dominated = False
        
        for j in range(n_samples):
            if i == j:
                continue
            
            j_dominates_i = True
            all_equal = True
            
            for k in range(len(metric_names)):
                if metrics_matrix[j, k] > metrics_matrix[i, k]:
                    j_dominates_i = False
                    break
                elif metrics_matrix[j, k] < metrics_matrix[i, k]:
                    all_equal = False
            
            if j_dominates_i and not all_equal:
                dominated = True
                break
        
        if not dominated:
            pareto_indices.append(i)
    
    return [param_scores[i] for i in pareto_indices]


def visualize_samples(original_batch, ground_truth_batch,
                              preprocessed_batch, original_pred_batch,
                              processed_pred_batch, save_dir,
                              k=5, mode=None, split_idx_batch=None, id=None):
    """Save per-sample visualization figures for a batch."""

    save_dir = os.path.join(save_dir, mode)
    os.makedirs(save_dir, exist_ok=True)
    
    batch_size = original_batch.shape[0]
    
    k = min(k, batch_size)
    
    pred_len = ground_truth_batch.shape[1]
    if k == 1:
        indices = [batch_size // 2]
    else:
        indices = np.linspace(0, batch_size - 1, k + 1, dtype=int)[: -1]
    
    ######
    indices = list(indices)
    target_indices = np.arange(45210 - 10, 45210 + 10 + 1)
    mask = np.isin(target_indices, split_idx_batch)
    existing_indices = target_indices[mask]
    for target_idx in existing_indices:
        batch_pos = np.where(split_idx_batch == target_idx)[0][0]
        if batch_pos not in indices:
            indices.append(batch_pos)
    target_indices = np.arange(45354 - 10, 45354 + 10 + 1)
    mask = np.isin(target_indices, split_idx_batch)
    existing_indices = target_indices[mask]
    for target_idx in existing_indices:
        batch_pos = np.where(split_idx_batch == target_idx)[0][0]
        if batch_pos not in indices:
            indices.append(batch_pos)

    if 7162 in split_idx_batch:
        indices.append(np.where(split_idx_batch == 7162)[0][0])

    if 7233 in split_idx_batch:
        indices.append(np.where(split_idx_batch == 7233)[0][0])

    indices = np.array(indices)
    #######
    for i, idx in enumerate(indices):
        original_seq = original_batch[idx].squeeze()
        index = split_idx_batch[idx]
        if ground_truth_batch is not None:
            ground_truth = ground_truth_batch[idx].squeeze()
        else:
            ground_truth = np.zeros(pred_len)
        
        preprocessed_seq = preprocessed_batch[idx].squeeze()
        original_pred = original_pred_batch[idx].squeeze()
        processed_pred = processed_pred_batch[idx].squeeze()
        
        front_len = 1440 # 672
        
        if len(original_seq) > front_len:
            original_seq = original_seq[-front_len:]
        if len(preprocessed_seq) > front_len:
            preprocessed_seq = preprocessed_seq[-front_len:]
        
        curve1 = np.concatenate((original_seq, ground_truth))
        curve2 = np.concatenate((original_seq, processed_pred))
        curve3 = np.concatenate((preprocessed_seq, original_pred))
        
        plt.figure(figsize=(ceil(len(curve1) / 96) * 5, 5))

        plt.plot(curve1, label="Original + Ground Truth", 
                linestyle='-', color='orange', linewidth=2.5, alpha=0.8)
        plt.plot(curve2, label="Original + Prediction", 
                linestyle='--', color='blue', linewidth=2.5, alpha=0.8)
        plt.legend() #(loc='upper left', fontsize=10)
        save_path = os.path.join(save_dir, f'{index:04d}_{id}-ori.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


        plt.figure(figsize=(ceil(len(curve3) / 96) * 5, 5))
        plt.plot(curve3, label="Preprocessed + Original Prediction", 
                linestyle='-', color='green', linewidth=2.5, alpha=0.8)
        plt.legend() #(loc='upper left', fontsize=10)
        save_path = os.path.join(save_dir, f'{index:04d}_{id}-trans.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
    return None


def loss_plot(save_dir, mode, id, mse_list, beg_split_idx):
    if not mse_list:
        return
    
    plot_dir = os.path.join(save_dir, 'loss_plot', mode)
    os.makedirs(plot_dir, exist_ok=True)
    
    filepath = os.path.join(plot_dir, f"{id}.png")
    
    plt.figure(figsize=(20, 6))
    
    inx = range(beg_split_idx, len(mse_list) + beg_split_idx)
    plt.plot(inx, mse_list, 'b-', linewidth=2, label='MSE Loss')
    
    mean_value = np.mean(mse_list)
    plt.axhline(y=mean_value, color='r', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_value:.4f}')
    
    # plt.xlabel('Epoch/Step')
    # plt.ylabel('MSE Loss')
    # plt.title(f'{mode} Loss Curve - {id}')
    # plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    
    return None
