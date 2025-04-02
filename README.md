# da6401_assignment2
Assignment to learn how to use CNNs: train from scratch and fine tune a pre-trained model as it is.


# Project Structure for Deep Learning CNN Project

## Directory Hierarchy
```
.
├── configs/
│   └── model_config.py          # Configuration for model hyperparameters
│
├── models/
│   ├── __init__.py
│   └── cnn_model.py             # Main model definition
│
├── optimizers/
│   ├── __init__.py
│   └── custom_optimizer.py      # Custom optimizer configurations
│
├── utils/
│   ├── __init__.py
│   └── parameter_counter.py     # Utility for parameter counting
│
└── train.py                     # Main training script
```

## Configuration Files
- `configs/model_config.py`: Centralized configuration management
  - Stores hyperparameters
  - Allows easy model configuration
  - Supports dynamic parameter modification

## Model Definition
- `models/cnn_model.py`: Core neural network architecture
  - Implements flexible CNN structure
  - Supports configurable layers and parameters
  - Inherits from PyTorch `nn.Module`

## Optimizer Configuration
- `optimizers/custom_optimizer.py`: Custom optimization strategies
  - Can define specialized optimization techniques
  - Provides flexibility in training approach

## Utility Functions
- `utils/parameter_counter.py`: Model analysis tools
  - Computes trainable parameter count
  - Supports model introspection

## Main Training Script
- `train.py`: Entry point for model training
  - Integrates configuration, model, and optimization
  - Manages training loop
  - Handles data loading and model execution