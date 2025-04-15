# import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

class LightningModel(pl.LightningModule):
    """PyTorch Lightning wrapper for the CNN model."""
    def __init__(self, model, learning_rate=1e-3, weight_decay=1e-5):
        super(LightningModel, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()
        # os.getcwd()
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=5, 
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }


def train_model(model, data_loaders, config):
    """
    Train a model using PyTorch Lightning.
    
    Args:
        model: The PyTorch model to train
        data_loaders: Dictionary containing train, val, and test data loaders
        config: Dictionary containing training configuration
        
    Returns:
        trained_model: The trained PyTorch Lightning model
    """
    # Create PyTorch Lightning model
    lightning_model = LightningModel(
        model, 
        learning_rate=config.get('learning_rate', 1e-3),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    # Configure logger
    wandb_logger = WandbLogger(
        project=config.get('project_name', 'da6401_assignment2'),
        name=config.get('run_name', None),
        log_model=False
    )
    
    # Configure callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=config.get('checkpoint_dir', './checkpoints'),
        filename='inaturalist-cnn-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=config.get('patience', 10),
        mode='min'
    )
    
    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=config.get('max_epochs', 50),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        precision=16 if config.get('use_mixed_precision', True) else 32,
        log_every_n_steps=50,
        deterministic=True
    )
    
    # Train the model
    trainer.fit(
        lightning_model, 
        train_dataloaders=data_loaders['train'],
        val_dataloaders=data_loaders['val']
    )
    
    # Test the model
    trainer.test(dataloaders=data_loaders['test'])
    
    return lightning_model