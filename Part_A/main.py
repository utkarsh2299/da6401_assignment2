# import os
# import torch
import argparse
import wandb
# import pytorch_lightning as pl
# from pytorch_lightning.loggers import WandbLogger

from model import CNN, ResNet50FineTuner  # Added ResNet50FineTuner import
from dataset import create_dataloaders
from trainer import train_model, LightningModel
from sweep_config import get_sweep_config

def parse_args():
    parser = argparse.ArgumentParser(description="Training a CNN or fine-tuning ResNet50 on the iNaturalist dataset")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--use_augmentation", action="store_true", help="Use data augmentation")
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224], 
                        help="Input image size (height, width)")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="cnn", choices=["cnn", "resnet50"],
                       help="Type of model to use (CNN from scratch or ResNet50 fine-tuning)")
    
    # CNN model parameters (used when model_type = "cnn")
    parser.add_argument("--num_blocks", type=int, default=5, help="Number of conv blocks")
    parser.add_argument("--base_filters", type=int, default=128, help="Base number of filters")
    parser.add_argument("--filter_config", type=str, default="fixed", 
                        choices=["fixed", "doubling", "halving"], 
                        help="Filter configuration across layers i.e., same number of filters in all layers, doubling in each subsequent layer, halving in each subsequent layer, etc")
    parser.add_argument("--filter_sizes", type=int, nargs="+", default=[3, 3, 5, 5, 7], 
                        help="List of kernel sizes for each convolutional layer")
    parser.add_argument("--activation", type=str, default="mish", 
                        choices=["relu", "gelu", "silu", "mish"], 
                        help="Activation function")
    parser.add_argument("--dense_activation", type=str, default="relu", 
                        choices=["relu", "gelu", "silu", "mish"], 
                        help="Activation function")
    parser.add_argument("--dense_neurons", type=int, default=512, help="Number of neurons in dense layer")
    parser.add_argument("--batch_norm", action="store_true", help="Use batch normalization")
    parser.add_argument("--dropout_rate", type=float, default=0, help="Dropout rate")
    
    # ResNet50 fine-tuning parameters (used when model_type = "resnet50")
    parser.add_argument("--freeze_option", type=int, default=1, 
                        choices=[0, 1, 2],
                        help="Freeze options: 0=unfreeze fc only, 1=unfreeze fc+last block, 2=unfreeze all")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=5, help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--use_mixed_precision", action="store_true", help="Use mixed precision training")
    
    # Experiment tracking
    parser.add_argument("--project_name", type=str, default="da6401_assignment2", help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory for save checkpoints")
    
    # Sweep configuration
    parser.add_argument("--run_sweep", action="store_true", help="Run hyperparameter sweep or not")
    parser.add_argument("--sweep_count", type=int, default=60, help="Number of sweep runs")
    
    return parser.parse_args()


def sweep_train():
    # This function is called by wandb sweep agent

    # wandb.init()
    print("Running as part of a sweep")
    
    # Get the config from wandb
    with wandb.init(name="utk") as run:
        args = parse_args()
        config = wandb.config
        
        # Create a descriptive run name based on model type
        if config.get('model_type', 'cnn') == 'cnn':
            model_name = f"cnn_{config.get('filter_config')}_bs_{config.get('batch_size')}_dn{config.get('dense_neurons')}_f{config.get('base_filters')}_ks{config.get('filter_sizes')}_act{config.get('activation')}_ep{config.get('max_epochs')}_lr{config.get('learning_rate')}"
        else:  # ResNet50
            model_name = f"resnet50_bs_{config.get('batch_size')}_dn{config.get('dense_neurons')}_fr{config.get('freeze_option')}_ep{config.get('max_epochs')}_lr{config.get('learning_rate')}"
        
        run.name = model_name
        run.save()

        # data loaders from dataset.py
        data_loaders = create_dataloaders(
            args.data_dir,
            batch_size=config.get('batch_size', 32),
            val_split=config.get('val_split', 0.2),
            use_augmentation=config.get('use_augmentation', True),
            image_size=tuple(config.get('image_size', (224, 224)))
        )
        
        # Create model based on model type
        if config.get('model_type', 'cnn') == 'cnn':
            # Create CNN model
            model = CNN(
                input_channels=3,
                input_size=data_loaders['image_size'],
                num_classes=data_loaders['num_classes'],
                num_blocks=config.get('num_blocks', 5),
                filter_config=config.get('filter_config', 'fixed'),
                base_filters=config.get('base_filters', 32),
                filter_sizes=config.get('filter_sizes', [3, 3, 3, 3, 3]),
                activation=config.get('activation', 'relu'),
                dense_neurons=config.get('dense_neurons', 128),
                batch_norm=config.get('batch_norm', False),
                dropout_rate=config.get('dropout_rate', 0),
                dense_activation=config.get('dense_activation', 'relu')
            )
            
            # Calculate the model complexity for CNN
            try:
                complexity = model.compute_complexity(
                    m=config.get('base_filters', 32),
                    k=config.get('filter_sizes', [3, 3, 3, 3, 3]),
                    n=config.get('dense_neurons', 128)
                )
                
                print(f"Total computations: {complexity['total_computations']:,}")
                print(f"Total parameters: {complexity['total_parameters']:,}")
            except Exception as e:
                print(f"Warning: Could not compute model complexity: {e}")
        else:
            # Create ResNet50 fine-tuning model
            model = ResNet50FineTuner(
                num_classes=data_loaders['num_classes'],
                dense_neurons=config.get('dense_neurons', 512),
                dropout_rate=config.get('dropout_rate', 0),
                freeze_option=config.get('freeze_option', 1)
            )
            
            # Print the number of trainable parameters for ResNet50
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Total parameters: {total_params:,}")
        
        print(wandb.run.name)
        # Train the model
        trained_model = train_model(model, data_loaders, config)
    wandb.finish()
    return trained_model


# Function for regular runs
def regular_train():
    # This is for standard training, not sweep
    print("Running as a standalone training job")
    
    # Parse arguments
    args = parse_args()
    config = vars(args)
    
    # Create a descriptive run name if one wasn't provided
    if config.get('run_name') is None:
        try:
            if config.get('model_type') == 'cnn':
                filter_sizes_str = "_".join(str(k) for k in config.get('filter_sizes', [3, 3, 3, 3, 3]))
                config['run_name'] = f"cnn_{config.get('filter_config')}_bs{config.get('batch_size')}_dn{config.get('dense_neurons')}_f{config.get('base_filters')}_ks{filter_sizes_str}_act{config.get('activation')}_ep{config.get('max_epochs')}_lr{config.get('learning_rate')}"
            else:  # ResNet50
                config['run_name'] = f"resnet50_bs{config.get('batch_size')}_dn{config.get('dense_neurons')}_fr{config.get('freeze_option')}_ep{config.get('max_epochs')}_lr{config.get('learning_rate')}"
        except:
            # Fallback if something goes wrong with the run name generation
            config['run_name'] = f"{config.get('model_type')}_training_run"
    
    # Initialize wandb for regular run
    wandb.init(
        project=config["project_name"],
        name=config["run_name"],
        config=config
    )
    
    # Update config from wandb
    config = wandb.config
    
    # data loaders from dataset.py
    data_loaders = create_dataloaders(
        config.get('data_dir'),
        batch_size=config.get('batch_size', 32),
        val_split=config.get('val_split', 0.2),
        use_augmentation=config.get('use_augmentation', True),
        image_size=tuple(config.get('image_size', (224, 224)))
    )
    
    # Create model based on model type
    if config.get('model_type') == 'cnn':
        # Create CNN model
        model = CNN(
            input_channels=3,
            input_size=data_loaders['image_size'],
            num_classes=data_loaders['num_classes'],
            num_blocks=config.get('num_blocks', 5),
            filter_config=config.get('filter_config', 'fixed'),
            base_filters=config.get('base_filters', 32),
            filter_sizes=config.get('filter_sizes', [3, 3, 3, 3, 3]),
            activation=config.get('activation', 'relu'),
            dense_neurons=config.get('dense_neurons', 128),
            batch_norm=config.get('batch_norm', False),
            dropout_rate=config.get('dropout_rate', 0),
            dense_activation=config.get('dense_activation', 'relu')
        )
        
        # Calculate the model complexity for CNN
        try:
            complexity = model.compute_complexity(
                m=config.get('base_filters', 32),
                k=config.get('filter_sizes', [3, 3, 3, 3, 3]),
                n=config.get('dense_neurons', 128)
            )
            
            print(f"Total computations: {complexity['total_computations']:,}")
            print(f"Total parameters: {complexity['total_parameters']:,}")
        except Exception as e:
            print(f"Warning: Could not compute model complexity: {e}")
    else:
        # Create ResNet50 fine-tuning model
        model = ResNet50FineTuner(
            num_classes=data_loaders['num_classes'],
            dense_neurons=config.get('dense_neurons', 512),
            dropout_rate=config.get('dropout_rate', 0),
            freeze_option=config.get('freeze_option', 1)
        )
        
        # Print the number of trainable parameters for ResNet50
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
    
    # Train the model
    trained_model = train_model(model, data_loaders, config)
    wandb.finish()
    return trained_model


def run_sweep():
    """Run a hyperparameter sweep using W&B."""
    wandb.login()
    args = parse_args()
    
    # Get sweep configuration - this should be updated to include ResNet50 parameters
    sweep_config = get_sweep_config()
    
    # Add ResNet50 specific parameters to sweep if they're not already there
    if 'parameters' in sweep_config:
        if 'model_type' not in sweep_config['parameters']:
            sweep_config['parameters']['model_type'] = {'values': ['cnn', 'resnet50']}
        if 'freeze_option' not in sweep_config['parameters']:
            sweep_config['parameters']['freeze_option'] = {'values': [0, 1, 2]}
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project=args.project_name)
    
    # Run the sweep - use sweep_train function specifically for sweeps
    wandb.agent(sweep_id, function=sweep_train, count=args.sweep_count)
    wandb.finish()

def main():
    args = parse_args()
    
    if args.run_sweep:
        run_sweep()
    else:
        # Run a single training run with the given configuration default
        regular_train()


if __name__ == "__main__":
    main()