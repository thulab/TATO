import logging
import time
import numpy as np
import torch
import os
import gc
import tempfile
from typing import Optional, Any, Dict, Tuple
from transformers import AutoModelForCausalLM
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from utils.tools import EarlyStopping, LargeScheduler
import torch.cuda.amp as amp
import argparse
import einops
from pathlib import Path


# Checkpoint directory constant
CKPT_LIB = './CKPT/'

# Optional imports for models (may not be available in all environments)
try:
    from chronos import ChronosPipeline, Chronos2Pipeline
except Exception:  # optional dependency / local submodule
    ChronosPipeline, Chronos2Pipeline = None, None

try:
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
except Exception:  # optional dependency / local submodule
    MoiraiForecast, MoiraiModule = None, None

try:
    from Timer.exp.exp_large_few_shot_roll_demo import Exp_Large_Few_Shot_Roll_Demo as ExpTimer
except Exception:  # optional dependency / local submodule
    ExpTimer = None

class TimerXL:
    """Lightweight Timer model variant for time series forecasting.
    
    This class implements a simplified version of the Timer model using
    Hugging Face's AutoModelForCausalLM for sequence generation.
    
    Attributes:
        model_name: Name identifier for the model
        patch_len: Patch length used by the model (default: 96)
        input_token_len: Input token length (default: 96)
        device: Computation device (e.g., 'cuda:0', 'cpu')
        args: Additional configuration arguments
        model: Hugging Face AutoModelForCausalLM instance
    """
    
    def __init__(self, model_name: str, ckpt_path: str, device: str, args: Optional[Any] = None):
        """Initialize the TimerXL model.
        
        Args:
            model_name: Name identifier for the model
            ckpt_path: Path to model checkpoint
            device: Computation device (e.g., 'cuda:0', 'cpu')
            args: Additional configuration arguments
        """
        self.model_name = model_name
        self.patch_len = 96
        self.input_token_len = 96
        self.device = device
        self.args = args
        print(f'self.device: {self.device}')
        
        # Load pre-trained model from checkpoint
        self.model = AutoModelForCausalLM.from_pretrained(
            ckpt_path,
            trust_remote_code=True,
        ).to(self.device)

    def forecast(self, data: np.ndarray, pred_len: int) -> np.ndarray:
        """Generate forecasts for input data.
        
        Args:
            data: Input time series data
                - Shape: (batch_size, seq_len, num_features) for 3D arrays
                - Shape: (seq_len,) or (batch_size, seq_len) for 1D/2D arrays
            pred_len: Number of time steps to forecast
            
        Returns:
            Forecasted values of shape (batch_size, pred_len, 1)
        """
        # Convert input data to torch tensor
        if len(data.shape) == 3:
            # Extract last feature for univariate forecasting: (B,S,C) -> (B,S)
            data = torch.tensor(data[:, :, -1], dtype=torch.float32).to(self.device)
        else:
            data = torch.tensor(data, dtype=torch.float32).to(self.device)
        
        # Ensure data has batch dimension
        if data.dim() == 1:
            data = data.unsqueeze(0)
        
        # Generate predictions using the model
        pred = self.model.generate(data, max_new_tokens=pred_len)
        
        # Add feature dimension and convert to numpy
        pred = pred.unsqueeze(2).detach().to('cpu').numpy()
        return pred


class Timer:
    """Full Timer model implementation for time series forecasting.
    
    This class wraps the Timer model from the Timer submodule, providing
    a unified interface for time series forecasting.
    
    Attributes:
        model_name: Name identifier for the model
        args: Configuration arguments for the Timer model
        exp: Timer experiment instance for inference
        patch_len: Patch length used by the model
    """
    
    def __init__(self, model_name: str, ckpt_path: str, device: str, args: Any):
        """Initialize the Timer model.
        
        Args:
            model_name: Name identifier for the model
            ckpt_path: Path to model checkpoint
            device: Computation device (e.g., 'cuda:0', 'cpu')
            args: Configuration arguments (will be updated with Timer defaults)
        """
        # Default configuration for Timer model
        defaults = {
            "task_name": 'large_finetune',
            "model_name": 'Timer',
            "ckpt_path": ckpt_path,
            "patch_len": 96,
            "d_model": 1024,
            "n_heads": 16,
            "e_layers": 8,
            "d_ff": 2048,
            "factor": 3,
            "dropout": 0.1,
            "activation": 'gelu',
            "output_attention": False,
            "use_gpu": True,
            "gpu": device,
            "use_multi_gpu": False,
        }

        # Update args with Timer defaults
        for attr, default_value in defaults.items():
            setattr(args, attr, default_value)

        # Validate and configure device settings
        assert 'cpu' == device or 'cuda' in device
        args.use_gpu = True if 'cuda' in device else False
        args.gpu = device.split(':')[-1] if 'cuda' in device else 0
        print(f'args.use_gpu={args.use_gpu}, args.gpu={args.gpu}')

        self.model_name = model_name
        self.args = args
        self.exp = ExpTimer(args)  # Initialize Timer experiment instance
        self.patch_len = self.args.patch_len

    def forecast(self, data: np.ndarray, pred_len: int) -> np.ndarray:
        """Generate forecasts for input data using Timer model.
        
        Args:
            data: Input time series data of shape (batch_size, seq_len, num_features)
            pred_len: Number of time steps to forecast
            
        Returns:
            Forecasted values of shape (batch_size, pred_len, num_features)
        """
        # Perform inference using Timer experiment
        _pred_total = self.exp.any_inference(data, pred_len)  # shape: (batch_size, total_len, num_features)
        
        # Extract only the prediction portion
        pred = _pred_total[:, -pred_len:, :]
        return pred


class MOIRAI:
    def __init__(self, model_name, ckpt_path, device):
        self.model_name = model_name
        self.dtype = torch.float32
        self.device = self.choose_device(device)
        self.num_samples = 20
        self.patch_size = 128
        self.patch_len = self.patch_size
        self.module = MoiraiModule.from_pretrained(ckpt_path, local_files_only=True)
        logging.info(f'num_samples={self.num_samples}, patch_size={self.patch_size}')

    def choose_device(self, device):
        if 'cpu' == device:
            return 'cpu'
        elif 'cuda' in device:
            idx = int(device.split(':')[-1])
            os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
            return 'cuda:0'
        else:
            raise ValueError(f'Unknown device: {device}')

    def forecast(self, data, pred_len):
        batch_size, seq_len, feature = data.shape
        real_seq_l = seq_len
        real_pred_l = pred_len
        _data = data.reshape(batch_size * real_seq_l * feature)
        seq_with_zero_pred = np.concatenate([_data, np.zeros(real_pred_l)])
        date_range = pd.date_range(start='1900-01-01', periods=len(seq_with_zero_pred), freq='s')
        data_pd = pd.DataFrame(seq_with_zero_pred, index=date_range, columns=['target'])
        ds = PandasDataset(dict(data=data_pd))
        train, test_template = split(ds, offset=real_seq_l)
        test_data = test_template.generate_instances(
            prediction_length=real_pred_l,
            windows=batch_size,
            distance=real_seq_l,
        )

        with torch.no_grad(), amp.autocast(dtype=self.dtype):
            predictor = MoiraiForecast(
                module=self.module,
                prediction_length=real_pred_l,
                context_length=real_seq_l,
                patch_size=self.patch_size,
                num_samples=self.num_samples,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            ).create_predictor(batch_size=batch_size, device=self.device)
            forecasts = predictor.predict(test_data.input)
            forecast_list = list(forecasts)

        assert len(forecast_list) == batch_size, f'len(forecast_list)={len(forecast_list)}'
        preds = np.array([forecast.quantile(0.5) for forecast in forecast_list])
        
        preds = preds.reshape((batch_size, real_pred_l, feature))
        return preds


class Chronos:
    def __init__(self, model_name,ckpt_path, device):
        if ChronosPipeline is None:
            raise ImportError(
                "ChronosPipeline is not available. Install/package `chronos` or add `Chronos/src` to PYTHONPATH."
            )
        self.model_name = model_name
        self.device = self.choose_device(device)
        self.org_device = self.device
        self.ckpt_path = ckpt_path
        self.dtype = torch.float16
        self.pipeline = ChronosPipeline.from_pretrained(
            self.ckpt_path,
            device_map=self.device,
            torch_dtype=self.dtype,
        )
        self.pipeline.model = self.pipeline.model.to(self.device)  # Ensure the model is on the correct device
        self.pipeline.model.eval()
        self.num_samples = 3
        self.patch_len = 512

    def reinit(self, device, dtype):
        self.device = self.choose_device(device)
        self.pipeline = None
        self.pipeline = ChronosPipeline.from_pretrained(
            self.ckpt_path,
            device_map=device,
            torch_dtype=dtype
        )
        self.pipeline.model.eval()

    def choose_device(self, device):
        if 'cpu' == device:
            return 'cpu'
        elif 'cuda' in device:
            idx = int(device.split(':')[-1])
            os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
            return 'cuda:0'
        else:
            raise ValueError(f'Unknown device: {device}')

    def forecast(self, data, pred_len):
        batch_size, seq_len, feature = data.shape
        assert feature == 1, f'feature={feature}'
        with torch.no_grad(), amp.autocast(dtype=self.dtype):
            max_repeat = 5
            while max_repeat > 0:
                try:
                    if self.device != self.org_device:
                        logging.info(f'Chronos device changed, reinit...')
                        self.reinit(self.org_device, self.dtype)
                    data = torch.Tensor(data.reshape(batch_size, seq_len))
                    forecast = self.pipeline.predict(
                        context=data,
                        prediction_length=pred_len,
                        num_samples=self.num_samples,
                        limit_prediction_length=False,
                    )
                    break
                except Exception as e:
                    logging.error(e)
                    logging.info(f'Chronos predict failed, max_repeat={max_repeat}, reinit...')
                    time.sleep(3)
                    # device = 'cuda:0' if max_repeat != 1 else 'cpu'
                    # dtype = random.choice([torch.float16, torch.float32, torch.float64])
                    device, dtype = self.device, self.dtype
                    logging.info(f'device={device}, dtype={dtype}')
                    try:
                        self.reinit(device, dtype)
                    except Exception as e:
                        logging.error(e)
                        logging.info(f'Chronos reinit failed, max_repeat={max_repeat}, reinit...')
                    max_repeat -= 1
                    if max_repeat == 0:
                        raise ValueError(f'Chronos predict failed, with error: {e}')
            assert forecast.shape == (batch_size, self.num_samples, pred_len), f'forecast.shape={forecast.shape}'
            preds = np.median(forecast.numpy(), axis=1).reshape((batch_size, pred_len, 1))
            return preds


class Sundial:
    def __init__(self, model_name, ckpt_path, device, args=None):
        self.model_name = model_name
        self.device = device
        self.ckpt_path = ckpt_path
        self.args = args

        self.model = AutoModelForCausalLM.from_pretrained(self.ckpt_path, trust_remote_code=True).to(self.device)
        self.dtype = torch.float32

        # Sundial uses patch/token length from config (default in ckpt: 16)
        self.input_token_len = int(getattr(self.model.config, 'input_token_len', 16))
        output_token_lens = getattr(self.model.config, 'output_token_lens', [720])
        self.max_output_token_len = int(output_token_lens[-1]) if len(output_token_lens) > 0 else 720


    def forecast(self, data, pred_len):
        _, _, feature = data.shape
        assert feature == 1, f'feature={feature}'
        self.model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=self.dtype):
            # Univariate forecasting
            data = torch.from_numpy(data).squeeze(dim=-1).to(torch.float32).to(self.device)
            # Note that Sundial can generate multiple probable predictions
            num_samples = 3
            output = self.model.generate(data, max_new_tokens=pred_len, num_samples=num_samples)
            output = np.median(output.detach().cpu().numpy(), axis=1).reshape(data.shape[0], pred_len, 1)
            return output


class Chronos2:
    def __init__(self, model_name, ckpt_path, device, args=None):
        if Chronos2Pipeline is None:
            raise ImportError(
                "Chronos2Pipeline is not available. Install/package `chronos` or add `Chronos/src` to PYTHONPATH."
            )
        self.model_name = model_name
        self.device = device
        self.org_device = self.device
        self.ckpt_path = ckpt_path
        self.args = args

        self.dtype = torch.float32
        self.pipeline = Chronos2Pipeline.from_pretrained(self.ckpt_path, device_map=self.device)
        self.pipeline.model = self.pipeline.model.to(self.device)  # Ensure the model is on the correct device
        self.pipeline.model.eval()
        self.num_samples = 3


    def forecast(self, data, pred_len):
        batch_size, seq_len, num_features = data.shape
        with torch.no_grad(), amp.autocast(dtype=self.dtype):
            data = torch.tensor(data, dtype=self.dtype)#.to(self.device)
            if num_features > 1:
                context = einops.rearrange(data, 'b t f -> (b f) t')
            else:
                context = data.squeeze(-1)

            context = context.to(self.device)
            
            if num_features > 1:
                group_ids = torch.repeat_interleave(
                    torch.arange(batch_size, device=self.device),
                    repeats=num_features
                )
            else:
                group_ids = torch.arange(batch_size, device=self.device)

            patch_size = self.pipeline.model.chronos_config.output_patch_size
            num_output_patches = (pred_len + patch_size - 1) // patch_size
            
            # predict
            with torch.no_grad():
                output = self.pipeline.model(
                    context=context,
                    group_ids=group_ids,
                    num_output_patches=num_output_patches
                )
            
            quantile_preds = output.quantile_preds  # (batch_size*num_features, num_quantiles, actual_pred_len)
            mean_preds = quantile_preds.mean(dim=1)  # (batch_size*num_features, actual_pred_len)
            
            mean_preds = mean_preds[:, :pred_len]  # (batch_size*num_features, pred_len)
            
            if num_features > 1:
                predictions = einops.rearrange(mean_preds, '(b f) t -> b t f', b=batch_size, f=num_features)
            else:
                predictions = mean_preds.unsqueeze(-1)

            return predictions.cpu().numpy()


import yaml
import importlib

class ModelFactory:
    _config = None
    _models_config = None
    
    @classmethod
    def _load_config(cls):
        """Load model configuration from YAML file."""
        if cls._config is None:
            config_path = Path(__file__).parent.parent / 'configs' / 'model_config.yaml'
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    cls._config = yaml.safe_load(f)
                cls._models_config = cls._config.get('models', {})
            except FileNotFoundError:
                raise FileNotFoundError(f"Model configuration file not found: {config_path}")
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing model configuration file: {e}")
        
        return cls._config
    
    @staticmethod
    def _ckpt(*parts: str) -> str:
        """Return a path under CKPT_LIB (relative to current working directory)."""
        return str(Path(CKPT_LIB) / Path(*parts))
    
    @classmethod
    def get_available_models(cls):
        """Get list of all available model names."""
        cls._load_config()
        return list(cls._models_config.keys())
    
    @classmethod
    def get_model_info(cls, model_name):
        """Get information about a specific model."""
        cls._load_config()
        if model_name not in cls._models_config:
            raise ValueError(f"Model '{model_name}' not found in configuration")
        
        info = cls._models_config[model_name].copy()
        info['name'] = model_name
        return info
    
    @classmethod
    def _get_model_class(cls, class_name):
        """Dynamically import and return model class."""
        cls._load_config()
        
        # Get class mapping from config
        class_mapping = cls._config.get('model_classes', {})
        if class_name not in class_mapping:
            raise ValueError(f"Class '{class_name}' not found in model_classes configuration")
        
        # Parse module and class name
        module_class_path = class_mapping[class_name]
        module_name, class_name_in_module = module_class_path.rsplit('.', 1)
        
        # Import module and get class
        try:
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name_in_module)
            return model_class
        except ImportError as e:
            raise ImportError(f"Failed to import module '{module_name}': {e}")
        except AttributeError as e:
            raise AttributeError(f"Class '{class_name_in_module}' not found in module '{module_name}': {e}")
    
    @classmethod
    def load_model(cls, model_name, device, args=None):
        """Load a model by name from configuration."""
        cls._load_config()
        
        # Check if model exists in configuration
        if model_name not in cls._models_config:
            available_models = cls.get_available_models()
            raise ValueError(
                f"Unknown model_name: '{model_name}'. "
                f"Available models: {', '.join(available_models)}"
            )
        
        # Get model configuration
        model_config = cls._models_config[model_name]
        class_name = model_config['class']
        ckpt_path = model_config['ckpt_path']
        requires_args = model_config.get('requires_args', False)
        
        # Get model class
        model_class = cls._get_model_class(class_name)
        
        # Create model instance
        try:
            if requires_args:
                model = model_class(model_name, ckpt_path, device, args)
            else:
                model = model_class(model_name, ckpt_path, device)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}': {e}")
    
    @classmethod
    def validate_config(cls):
        """Validate the model configuration."""
        cls._load_config()
        
        errors = []
        
        # Check required sections
        if 'models' not in cls._config:
            errors.append("Missing 'models' section in configuration")
        
        if 'model_classes' not in cls._config:
            errors.append("Missing 'model_classes' section in configuration")
        
        # Check each model configuration
        for model_name, model_config in cls._models_config.items():
            # Check required fields
            required_fields = ['class', 'ckpt_path']
            for field in required_fields:
                if field not in model_config:
                    errors.append(f"Model '{model_name}': missing required field '{field}'")
            
            # Check if class exists in model_classes
            if 'class' in model_config:
                class_name = model_config['class']
                if class_name not in cls._config.get('model_classes', {}):
                    errors.append(f"Model '{model_name}': class '{class_name}' not found in model_classes")
        
        if errors:
            error_msg = "Model configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ValueError(error_msg)
        
        return True
