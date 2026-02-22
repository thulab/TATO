# TATO: Adaptive Transformation Optimization for Domain-Shared Time Series Foundation Models

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**TATO** (Adaptive Transformation Optimization) is a novel framework that automatically optimizes data transformations to enhance the performance of time series foundation models across diverse domains.

## 📖 Paper Abstract

Time series foundation models have shown remarkable capabilities in various forecasting tasks. However, their performance can be significantly degraded when applied to datasets with different characteristics due to domain shift. TATO addresses this challenge by introducing an adaptive transformation optimization framework that automatically searches for the optimal combination of data transformations to adapt foundation models to new domains.

**Key Contributions:**
- **Adaptive Transformation Pipeline**: Automatically optimizes 8 types of data transformations
- **Model-Agnostic Design**: Compatible with various time series foundation models
- **Efficient Search**: Utilizes Optuna for hyperparameter optimization
- **Comprehensive Evaluation**: Supports multiple benchmark datasets

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/thulab/TATO.git
cd TATO
```

2. **Dependency Management**:
Different time series foundation models have different version requirements for the transformers library (especially Sundial requires 4.40.1, while other models use 4.41.0). We have adopted a modular dependency management solution:

```
TATO/
├── base_requirements.txt          # Base dependencies shared by all models
├── scripts/timer_runs/           # Timer model specific
│   └── timer_requirements.txt    # Timer model dependencies (transformers 4.41.0)
├── scripts/moirai_runs/          # MOIRAI model specific
│   └── moirai_requirements.txt   # MOIRAI model dependencies (transformers 4.41.0)
├── scripts/chronos_runs/         # Chronos model specific
│   └── chronos_requirements.txt  # Chronos model dependencies (transformers 4.41.0)
└── scripts/sundial_runs/         # Sundial model specific
    └── sundial_requirements.txt  # Sundial model dependencies (transformers 4.40.1)
```

3. **Set up model environments**:
```bash
# Create independent virtual environments for each model. It is noteworthy that conflicts exist among their envs. 
bash scripts/timer_runs/setup_timer.sh      # Timer model
bash scripts/moirai_runs/setup_moirai.sh    # MOIRAI model
bash scripts/chronos_runs/setup_chronos.sh  # Chronos model
bash scripts/sundial_runs/setup_sundial.sh  # Sundial model (requires transformers 4.40.1)
```

4. **Download pre-trained models and datasets**:

Install dependencies of the pre-trained models.
For *Timer*, please download it from *https://drive.google.com/file/d/1PFHMpa32dO8Y2fQ8N7vjsbTpFhd0fC8_/view?usp=drive_link* and add the decompressed directory to PYTHONPATH

```bash
# Download checkpoints (Timer, Chronos, etc.) and put them in CKPT
mkdir -p CKPT
# Download datasets (ETT, Exchange, etc.) and put them in DATASET
mkdir -p DATASET
# You can make soft link to the directory as needed.
```

### Model Configuration

TATO uses a configuration-driven approach for model management. All model definitions are stored in `configs/model_config.yaml`:

```yaml
# Model Configuration File
models:
  # Timer models
  Timer-UTSD:
    class: Timer
    ckpt_path: CKPT/Building_timegpt_d1024_l8_p96_n64_new_full.ckpt
    description: "Timer model with UTSD architecture"
    requires_args: true
  
  # MOIRAI models
  MOIRAI-small:
    class: MOIRAI
    ckpt_path: CKPT/MOIRAI-small
    description: "Small MOIRAI model"
    requires_args: false
  
  # Chronos models
  Chronos-tiny:
    class: Chronos
    ckpt_path: CKPT/Chronos-tiny
    description: "Tiny Chronos model"
    requires_args: false
  
  # Sundial models
  Sundial:
    class: Sundial
    ckpt_path: CKPT/sundial
    description: "Sundial model (requires transformers 4.40.1)"
    requires_args: true
```

#### Required Checkpoint Structure

After downloading pre-trained models, your `CKPT/` directory should have the following structure:

```
CKPT/
├── Building_timegpt_d1024_l8_p96_n64_new_full.ckpt/    # Timer-UTSD
├── Large_timegpt_d1024_l8_p96_n64_new_full.ckpt/       # Timer-LOTSA
├── MOIRAI-small/                                       # MOIRAI-small
├── MOIRAI-base/                                        # MOIRAI-base
├── MOIRAI-large/                                       # MOIRAI-large
├── Chronos-tiny/                                       # Chronos-tiny
└── sundial/                                            # Sundial
```

#### Adding New Models

To add a new model, simply update the configuration file:

1. Add model entry to `configs/model_config.yaml`:
```yaml
NewModel-Example:
  class: Timer  # or MOIRAI, Chronos, Sundial
  ckpt_path: CKPT/new_model_checkpoint
  description: "New example model"
  requires_args: true  # or false
```

2. Download the checkpoint to `CKPT/new_model_checkpoint/`

3. The model will be automatically available through `ModelFactory.load_model('NewModel-Example', device, args)`

### Basic Usage

Run a simple experiment with default settings:

```bash
python experiment/run.py \
  --device cuda:0 \
  --dataset ETTh1 \
  --model Timer-LOTSA \
  --pred_len 96 \
  --train_trials 100 \
  --num_samples 500
```

## 🏗️ Architecture

### Core Components

```
TATO/
├── data/              # Dataset loading and preprocessing
├── model/             # Foundation model implementations
├── transformation/    # Data transformation modules
├── pipeline/          # Transformation pipeline
├── tuner/            # Hyperparameter optimization
├── experiment/        # Experiment orchestration
└── utils/            # Utility functions
```

### Built-in Transformations

| Transformation | Purpose | Search Parameters |
|----------------|---------|-------------------|
| **Normalizer** | Scale normalization | method, mode |
| **Sampler** | Temporal sampling | factor |
| **Warper** | Time warping | method |
| **Differentiator** | Differencing | n |
| **Inputer** | Missing value imputation | detect_method, fill_method |
| **Denoiser** | Noise reduction | method |
| **Trimmer** | Sequence trimming | seq_l |
| **Aligner** | Sequence alignment | mode, method |

## 📊 Datasets

The following benchmark datasets are supported for now:

| Dataset | Frequency | Length | Features | Description |
|---------|-----------|--------|----------|-------------|
| **ETTh1** | Hourly | 17,420 | 7 | Electricity Transformer Temperature |
| **ETTh2** | Hourly | 17,420 | 7 | Electricity Transformer Temperature |
| **ETTm1** | 15-min | 69,680 | 7 | Electricity Transformer Temperature |
| **ETTm2** | 15-min | 69,680 | 7 | Electricity Transformer Temperature |
| **Exchange** | Daily | 7,588 | 8 | Exchange Rates |
| **Weather** | 10-min | 52,695 | 21 | Weather Data |
| **Electricity** | Hourly | 26,304 | 321 | Electricity Consumption |
| **Traffic** | Hourly | 17,544 | 862 | Traffic Flow |

## 🔧 Advanced Usage

### Custom Experiment Configuration

Create a custom experiment script:

```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment.run import main
import argparse

# Custom configuration
args = argparse.Namespace(
    seed=42,
    device='cuda:0',
    dataset='Weather',
    model='MOIRAI-base',
    pred_len=336,  # 2-week forecast
    patch_len=128,
    train_trials=200,
    num_samples=1000,
    top_k=10,
    save_dir='./custom_results',
    no_aug=False,
    plt=True
)

main(args)
```

### Adding Custom Transformations

1. Create a new transformation in `transformation/library/`:

```python
# transformation/library/custom_transform.py
from transformation.base import BaseTransformation

class Transformation(BaseTransformation):
    search_space = {
        'parameter': [1, 2, 3, 4, 5]
    }
    
    def __init__(self, parameter, **kwargs):
        self.parameter = parameter
    
    def pre_process(self, data):
        # Your transformation logic
        return transformed_data
    
    def post_process(self, data):
        # Inverse transformation
        return restored_data
```

2. The transformation will be automatically discovered by the factory.

### Integration with New Models

To add a new foundation model:

```python
# model/custom_model.py
from model.model_factory import ModelFactory

class CustomModel:
    def __init__(self, model_name, ckpt_path, device, args=None):
        self.model_name = model_name
        self.device = device
        # Initialize your model
    
    def forecast(self, data, pred_len):
        # Implement forecasting logic
        return predictions

# Register in model_factory.py
ModelFactory.register_model('custom-model', CustomModel)
```

## 📈 Results and Evaluation

### Performance Metrics

TATO evaluates models using multiple metrics:
- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)
- **MSPE** (Mean Squared Percentage Error)

### Optimization Process

The framework follows a three-stage optimization:

1. **Training Phase**: Search for optimal transformation combinations
2. **Validation Phase**: Evaluate candidate transformations
3. **Testing Phase**: Final performance assessment

### Visualization

Enable visualization with the `--plt` flag to generate:
- Transformation effect plots
- Prediction vs ground truth comparisons

## 🧪 Experiments

### Reproducing Paper Results

To reproduce the experiments from the paper, use the new modular script structure:

#### New Script Structure (Recommended)

Each model now has its own dedicated directory with setup and run scripts:

```
scripts/
├── timer_runs/                    # Timer model specific
│   ├── setup_timer.sh            # Environment setup script
│   ├── run_timer.sh              # Run script
│   └── timer_requirements.txt    # Dependency configuration
├── moirai_runs/                  # MOIRAI model specific
│   ├── setup_moirai.sh
│   ├── run_moirai.sh
│   └── moirai_requirements.txt
├── chronos_runs/                 # Chronos model specific
│   ├── setup_chronos.sh
│   ├── run_chronos.sh
│   └── chronos_requirements.txt
└── sundial_runs/                 # Sundial model specific
    ├── setup_sundial.sh
    ├── run_sundial.sh
    └── sundial_requirements.txt
```

#### Base Dependencies
- `base_requirements.txt` - Base dependencies shared by all models

#### Usage Instructions

1. **Set up model environments**:
```bash
# Timer model
bash scripts/timer_runs/setup_timer.sh

# Sundial model (requires transformers 4.40.1)
bash scripts/sundial_runs/setup_sundial.sh
```

2. **Run model experiments**:
```bash
# Run after activating environment
source venv_timer/bin/activate
bash scripts/timer_runs/run_timer.sh

# Or run directly (script will check environment)
bash scripts/timer_runs/run_timer.sh
```

## 🔬 Research Extensions

### Extending TATO for Your Research

1. **Novel Transformations**: Implement domain-specific transformations
2. **Multi-objective Optimization**: Extend the tuner for Pareto-optimal solutions
3. **Online Adaptation**: Implement streaming data adaptation
4. **Cross-domain Transfer**: Study transfer learning across domains

### Citation

If you use TATO in your research, please cite our paper:

```bibtex
@inproceedings{qiu2026adapt,
    title={Adapt Data to Model: Adaptive Transformation Optimization for Domain-shared Time Series Foundation Models},
    author={Yunzhong Qiu and Zhiyao Cen and Zhongyi Pei and Chen Wang and Jianmin Wang},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=uTK1SNgi1N}
}
```

### Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Code Structure Guidelines

- Follow PEP 8 style guide
- Use type hints for function signatures
- Add docstrings for all public functions and classes
- Write unit tests for new functionality


## 🐛 Troubleshooting

### Common Issues

1. **Transformers Version Conflicts**
   - **Issue**: Sundial requires transformers 4.40.1, while other models require 4.41.0
   - **Solution**: Use independent virtual environments
   ```bash
   # Check current transformers version
   python -c "import transformers; print(transformers.__version__)"
   
   # Create independent environment for Sundial
   bash scripts/sundial_runs/setup_sundial.sh
   
   # Verify Sundial environment version
   source venv_sundial/bin/activate
   python -c "import transformers; print(f'Sundial environment: {transformers.__version__}')"
   deactivate
   
   # Verify other model environment versions
   source venv_timer/bin/activate
   python -c "import transformers; print(f'Timer environment: {transformers.__version__}')"
   deactivate
   ```

2. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   --batch_size 256
   
   # Use smaller model
   --model MOIRAI-small
   ```

3. **Dataset Download Issues**
   ```bash
   # Manual download
   wget -P DATASET/ https://dataset-url.com/data.zip
   unzip DATASET/data.zip -d DATASET/
   ```

4. **Checkpoint Not Found**
   ```bash
   # Download pre-trained checkpoints
   python scripts/download_checkpoints.py
   ```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for model hosting infrastructure
- Optuna for hyperparameter optimization framework
- All contributors and users of TATO

## 📞 Contact

For questions and collaborations:
- Email: peizhyi@tsinghua.edu.cn
- GitHub Issues: [github.com/thulab/TATO/issues](https://github.com/your-username/TATO/issues)
---

**TATO** - Making time series foundation models adaptable across domains. 🚀
