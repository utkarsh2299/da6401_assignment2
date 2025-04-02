# train.py
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from configs.model_config import ModelConfig
from configs.wandb_config import WandbConfig
from models.cnn_model import FlexibleCNN
from utils.wandb_utils import initialize_wandb, log_model_summary, create_sweep_configuration

def train_step(model, dataloader, criterion, optimizer, device):
    """
    Single training epoch
    
    Args:
        model (nn.Module): PyTorch model
        dataloader (DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim): Optimizer
        device (torch.device): Computing device
    
    Returns:
        dict: Training metrics
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Log batch-level metrics to Wandb
        wandb.log({
            'batch_loss': loss.item(),
            'batch_accuracy': 100. * correct / total
        })
    
    return {
        'epoch_loss': total_loss / len(dataloader),
        'epoch_accuracy': 100. * correct / total
    }

def train_with_wandb_sweep():
    """
    Train model using Wandb sweep
    """
    # Initialize Wandb run
    wandb.init(config=wandb.config)
    
    # Create device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Extract configuration from Wandb
    config = wandb.config
    
    # Create model with swept hyperparameters
    model_config = ModelConfig(
        num_layers=config.num_layers,
        initial_filters=config.initial_filters,
        dense_neurons=config.dense_neurons,
        activation=config.activation
    )
    model = FlexibleCNN(model_config).to(device)
    
    # Log model summary
    log_model_summary(model)
    
    # Create optimizer and loss function based on sweep config
    if config.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    
    criterion = nn.CrossEntropyLoss()
    
    # TODO: Replace with actual data loading
    # This is a placeholder - you'll need to implement actual data loading
    dataloader = _get_dataloader(batch_size=config.batch_size)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        metrics = train_step(model, dataloader, criterion, optimizer, device)
        
        # Log epoch-level metrics to Wandb
        wandb.log({
            'epoch': epoch,
            'loss': metrics['epoch_loss'],
            'accuracy': metrics['epoch_accuracy']
        })
    
    # Final validation/test (placeholder)
    wandb.log({'validation_accuracy': metrics['epoch_accuracy']})

def main():
    # Create sweep configuration
    sweep_config = create_sweep_configuration()
    
    # Initialize Wandb sweep
    sweep_id = wandb.sweep(
        sweep_config, 
        project='iNaturalist-CNN'
    )
    
    # Run sweep agent
    wandb.agent(sweep_id, function=train_with_wandb_sweep, count=10)

def _get_dataloader(batch_size=64):
    """
    Placeholder for data loading
    In real implementation, replace with actual data loading
    """
    # This is a mock function - replace with actual data loading
    from torch.utils.data import TensorDataset, DataLoader
    
    # Mock data
    inputs = torch.randn(1000, 3, 224, 224)
    targets = torch.randint(0, 10, (1000,))
    
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    main()