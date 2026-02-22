# Contributing to TATO

Thank you for your interest in contributing to TATO (Adaptive Transformation Optimization for Domain-Shared Time Series Foundation Models)! This document provides guidelines and instructions for contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style and Standards](#code-style-and-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Adding New Features](#adding-new-features)
- [Reporting Issues](#reporting-issues)
- [Community](#community)

## Code of Conduct

We are committed to fostering an open and welcoming environment. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) in all interactions.

## Getting Started

### Prerequisites
- Python 3.8+
- Git
- Basic understanding of time series forecasting and foundation models

### Setting Up Development Environment

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/thulab/TATO.git
   cd TATO
   ```

2. **Set Up Base Environment**
   ```bash
   # Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install base dependencies
   pip install -r base_requirements.txt
   ```

3. **Set Up Model-Specific Environments**
   Due to dependency conflicts between different time series foundation models, TATO uses separate environments:
   ```bash
   # Timer model
   bash scripts/timer_runs/setup_timer.sh
   
   # MOIRAI model
   bash scripts/moirai_runs/setup_moirai.sh
   
   # Chronos model
   bash scripts/chronos_runs/setup_chronos.sh
   
   # Sundial model (requires transformers 4.40.1)
   bash scripts/sundial_runs/setup_sundial.sh
   ```

4. **Download Pre-trained Models and Datasets**
   ```bash
   # Create directories
   mkdir -p CKPT DATASET
   
   # Download checkpoints and datasets as described in README.md
   # Place them in the appropriate directories
   ```

## Development Workflow

### Branch Strategy
- `main`: Stable production code
- `develop`: Integration branch for features
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `hotfix/*`: Critical production fixes

### Creating a Feature Branch
```bash
# Sync with upstream
git fetch upstream
git checkout develop
git pull upstream develop

# Create feature branch
git checkout -b feature/your-feature-name
```

### Making Changes
1. Make your changes in the feature branch
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure code follows style guidelines

### Committing Changes
We follow conventional commits format:
```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Example:
```bash
git commit -m "feat(transformation): add wavelet denoiser transformation"
```

## Code Style and Standards

### Python Code Style
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines
- Use type hints for function signatures
- Maximum line length: 88 characters (Black formatting)
- Use meaningful variable and function names

### Import Organization
```python
# Standard library imports
import os
import sys
from typing import Dict, List, Optional

# Third-party imports
import numpy as np
import torch
from transformers import AutoModelForCausalLM

# Local imports
from transformation.base import BaseTransformation
from utils.tools import EarlyStopping
```

### Documentation
- Use Google-style docstrings for all public functions and classes
- Include type hints in docstrings
- Document complex algorithms and design decisions

Example:
```python
def forecast(self, data: np.ndarray, pred_len: int) -> np.ndarray:
    """Generate forecasts for input data.
    
    Args:
        data: Input time series data of shape (batch_size, seq_len, num_features)
        pred_len: Number of time steps to forecast
        
    Returns:
        Forecasted values of shape (batch_size, pred_len, num_features)
        
    Raises:
        ValueError: If input data has incorrect shape
    """
```

### Configuration Files
- Use YAML for configuration files
- Follow existing structure in `configs/model_config.yaml`
- Include comments explaining configuration options

## Testing

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test module
python -m pytest tests/test_model_factory.py

# Run with coverage
python -m pytest tests/ --cov=model --cov-report=html
```

### Writing Tests
- Place tests in `tests/` directory
- Use descriptive test names
- Test both normal and edge cases
- Mock external dependencies when appropriate

Example:
```python
def test_model_factory_load_valid_model():
    """Test loading a valid model from factory."""
    model = ModelFactory.load_model('Timer-UTSD', device='cpu', args=None)
    assert model is not None
    assert isinstance(model, Timer)
```

### Test Structure
```
tests/
├── __init__.py
├── test_model_factory.py
├── test_transformations.py
├── test_pipeline.py
└── test_utils.py
```

## Documentation

### Updating Documentation
- Update README.md for major changes
- Add docstrings for new functions and classes
- Update API documentation in `docs/` directory
- Create tutorials for new features

### Building Documentation
```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build documentation
cd docs
make html
```

## Pull Request Process

1. **Create a Pull Request**
   - Target the `develop` branch
   - Provide a clear description of changes
   - Reference related issues

2. **PR Checklist**
   - [ ] Code follows style guidelines
   - [ ] Tests pass
   - [ ] Documentation updated
   - [ ] No breaking changes
   - [ ] Dependencies updated if needed

3. **Code Review**
   - Address reviewer comments
   - Update PR as needed
   - Ensure CI passes

4. **Merge**
   - Squash commits if needed
   - Use descriptive merge message
   - Delete feature branch after merge

## Adding New Features

### Adding New Transformations
1. Create new transformation class in `transformation/library/`
2. Inherit from `BaseTransformation`
3. Define `search_space` class variable
4. Implement `pre_process` and `post_process` methods
5. Add to `transformation/transformation_factory.py`
6. Write tests

Example:
```python
# transformation/library/wavelet_denoiser.py
from transformation.base import BaseTransformation

class WaveletDenoiser(BaseTransformation):
    """Wavelet-based denoising transformation."""
    
    search_space = {
        'wavelet': ['db1', 'db2', 'sym4'],
        'level': [1, 2, 3],
        'threshold': [0.1, 0.5, 1.0]
    }
    
    def __init__(self, wavelet: str, level: int, threshold: float):
        self.wavelet = wavelet
        self.level = level
        self.threshold = threshold
    
    def pre_process(self, data: np.ndarray) -> np.ndarray:
        # Implementation
        return denoised_data
    
    def post_process(self, data: np.ndarray) -> np.ndarray:
        # Inverse transformation if needed
        return data
```

### Adding New Models
1. Create model class in `model/` directory
2. Implement `__init__` and `forecast` methods
3. Add to `model/model_factory.py`
4. Update `configs/model_config.yaml`
5. Add model-specific requirements
6. Write tests

### Adding New Datasets
1. Add dataset loading logic to `data/dataset.py`
2. Follow existing dataset interface
3. Update dataset documentation
4. Add to supported datasets list in README

## Reporting Issues

### Bug Reports
When reporting bugs, include:
1. Clear description of the issue
2. Steps to reproduce
3. Expected vs actual behavior
4. Environment details (Python version, OS, etc.)
5. Error messages and stack traces

### Feature Requests
For feature requests, include:
1. Use case description
2. Proposed solution
3. Alternative solutions considered
4. Impact on existing functionality

### Security Issues
Please report security issues privately to the maintainers.

## Community

### Getting Help
- Check the [documentation](README.md)
- Search existing issues
- Join our [Discord/Slack community]
- Attend community meetings

### Contributing Beyond Code
- Improve documentation
- Write tutorials
- Report bugs
- Suggest features
- Help other contributors
- Review pull requests

### Recognition
Contributors are recognized in:
- Project README
- Release notes
- Contributor hall of fame

## License
By contributing to TATO, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

## Acknowledgments
Thank you for contributing to making time series foundation models more adaptable across domains!

---

*This document was inspired by best practices from open source projects and adapted for TATO's specific needs.*
