# utils/wandb_utils.py
import wandb
import torch
from typing import Dict, Any

def initialize_wandb(config: Dict[str, Any]):
    """
    Initialize Wandb logging
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        wandb.run: Wandb run object
    """
    return wandb.init(
        project=config.get('project', 'iNaturalist-CNN'),
        entity=config.get('entity', None),
        group=config.get('group', None),
        job_type=config.get('job_type', 'train'),
        tags=config.get('tags', None),
        config=config
    )

def log_model_summary(model, log_freq=True):
    """
    Log model summary to Wandb
    
    Args:
        model (torch.nn.Module): PyTorch model
        log_freq (bool): Whether to log layer-wise parameter frequency
    """
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    wandb.run.summary["total_params"] = total_params
    wandb.run.summary["trainable_params"] = trainable_params
    
    # Optional: Log parameter frequency if requested
    if log_freq:
        param_freq = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_freq[name] = param.numel()
        wandb.run.summary["param_frequency"] = param_freq

def create_sweep_configuration(
    model_config_space=None, 
    training_config_space=None
):
    """
    Create a flexible Wandb sweep configuration
    
    Args:
        model_config_space (dict): Hyperparameter search space for model
        training_config_space (dict): Hyperparameter search space for training
    
    Returns:
        dict: Wandb sweep configuration
    """
    if model_config_space is None:
        model_config_space = {
            'num_layers': {'values': [50, 100, 555]},
            'initial_filters': {'values': [32, 64, 128]},
            'dense_neurons': {'values': [256, 512, 1024]},
            'activation': {'values': ['relu', 'tanh', 'sigmoid']}
        }
    
    if training_config_space is None:
        training_config_space = {
            'learning_rate': {'values': [1e-2, 1e-3, 1e-4]},
            'batch_size': {'values': [32, 64, 128]},
            'optimizer': {'values': ['adam', 'sgd']}
        }
    
    sweep_config = {
        'method': 'random',  # or 'grid', 'bayes'
        'metric': {
            'name': 'validation_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            **model_config_space,
            **training_config_space
        }
    }
    
    return sweep_config