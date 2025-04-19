# import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import math
class CNNBlock(nn.Module):
    """A single block consisting of convolution, activation, and max-pooling."""
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 activation='relu', pool_size=2, batch_norm=False, dropout_rate=0):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
        
        # Batch normalization
        self.batch_norm = None
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'mish':
            self.activation = nn.Mish()
        else:
            raise ValueError(f"Activation function {activation} not supported")
        
        # Max pooling
        self.pool = nn.MaxPool2d(pool_size)
        
        # Dropout
        self.dropout = None
        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(dropout_rate)
    
    def forward(self, x):
        x = self.conv(x)
        
        if self.batch_norm is not None:
            x = self.batch_norm(x)
            
        x = self.activation(x)
        x = self.pool(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
            
        return x

class CNN(nn.Module):
    """CNN model with configurable architecture."""
    
    
    def __init__(self, input_channels=3, input_size=(224, 224), num_classes=10, 
                 num_blocks=5, 
                 filter_config='fixed', base_filters=32, filter_sizes=[3, 3, 3, 3, 3], 
                 activation='relu',dense_activation='relu', dense_neurons=128, 
                 batch_norm=False, dropout_rate=0):
        super(CNN, self).__init__()
        
        
        #a list to store all the blocks with proper parameters, useful during backpropogation
        self.blocks = nn.ModuleList()
        self.input_size = input_size #image_ size
        # Configure how filters evolve across layers
        if filter_config == 'fixed':
            filter_counts = [base_filters] * num_blocks
        elif filter_config == 'doubling':
            filter_counts = [base_filters * (2**i) for i in range(num_blocks)]
        elif filter_config == 'halving':
            filter_counts = [base_filters * (2**(num_blocks-1-i)) for i in range(num_blocks)]
        else:
            raise ValueError(f"Filter configuration {filter_config} not supported")
        
        
        # Ensure filter_sizes has the right length
        if len(filter_sizes) < num_blocks:
            # Extend filter_sizes list if it's too short
            filter_sizes = filter_sizes + [filter_sizes[-1]] * (num_blocks - len(filter_sizes))
        elif len(filter_sizes) > num_blocks:
            # Truncate filter_sizes list if it's too long
            filter_sizes = filter_sizes[:num_blocks]
            
        # First block
        self.blocks.append(
            CNNBlock(input_channels, filter_counts[0], filter_sizes[0], 
                     activation, batch_norm=batch_norm, dropout_rate=dropout_rate)
        )
        
        # Remaining of blocks
        for i in range(1, num_blocks):
            self.blocks.append(
                CNNBlock(filter_counts[i-1], filter_counts[i], filter_sizes[i], 
                         activation, batch_norm=batch_norm, dropout_rate=dropout_rate)
            )
        
        
        # Calculate the output size after convolutions and pooling
       
        # Each pooling with pool_size=2 reduces the dimension by half
        height, width = self.input_size
        final_height = height // (2 ** num_blocks)
        final_width = width // (2 ** num_blocks)
        
        # Dense layer
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(filter_counts[-1] * final_height * final_width, dense_neurons)
        

        if dense_activation == 'relu':
            self.dense_activation = nn.ReLU()
        elif dense_activation == 'gelu':
            self.dense_activation = nn.GELU()
        elif dense_activation == 'silu':
            self.dense_activation = nn.SiLU()
        elif dense_activation == 'mish':
            self.dense_activation = nn.Mish()
        else:
            raise ValueError(f"Activation function {dense_activation} not supported")
        
        # Output layer
        self.output = nn.Linear(dense_neurons, num_classes)
    
    def forward(self, x):
        # Pass through all conv blocks
        for block in self.blocks:
            x = block(x)
        
        # Flatten and pass through dense layers
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dense_activation(x)
        x = self.output(x)
        
        return x
    
    def compute_complexity(self, input_size=None, m=None, k=None, n=None):
        """
        Calculate the computational complexity and number of parameters
        
        Args:
            input_size: Input tensor size (C, H, W)
            m: Number of filters in each layer (if None, uses the model's configuration)
            k: Kernel size (if None, uses the model's configuration)
            n: Number of neurons in the dense layer (if None, uses the model's configuration)
            
        Returns:
            dict: Containing computations and parameters count
        """
        if m is None or k is None or n is None:
            raise ValueError("Please provide values for m, k, and n")
        if input_size is None:
            channels = 3  # rgb
            height, width = self.input_size
        else:
            channels, height, width = input_size
        # channels, height, width = input_size
        print(channels, height, width)
        if not isinstance(k, list):
            k = [k] * 5  # 5 layers as per assignment
        
        # does k has enough values?
        if len(k) < 5:
            k = k + [k[-1]] * (5 - len(k))
            
        total_computations = 0
        total_parameters = 0
        
        # For each convolutional layer
        for i in range(5):  # We have 5 blocks as specified
            # Computations for conv layer: H*W*C_in*C_out*k*k
            # For the first layer, C_in = channels (3 for RGB)
            # For subsequent layers, C_in = m (number of filters in previous layer)
            c_in = channels if i == 0 else m
            
            kernel_size = k[i]
            
            conv_computations = height * width * c_in * m * kernel_size * kernel_size
            
            # Parameters for conv layer: C_in*C_out*k*k + C_out (weights + bias)
            conv_parameters = c_in * m * kernel_size * kernel_size + m
            
            # After pooling, height and width are halved
            height //= 2
            width //= 2
            
            total_computations += conv_computations
            total_parameters += conv_parameters
        
        # Final feature map size
        feature_map_size = height * width * m
        
        # Dense layer: input_size * output_size computations
        dense_computations = feature_map_size * n
        dense_parameters = feature_map_size * n + n  # weights + bias
        
        # Output layer: dense_neurons * num_classes computations
        output_computations = n * 10  # Assuming 10 classes for simplicity
        output_parameters = n * 10 + 10  # weights + bias
        
        total_computations += dense_computations + output_computations
        total_parameters += dense_parameters + output_parameters
        
        return {
            "total_computations": total_computations,
            "total_parameters": total_parameters
        }
        
class ResNet50FineTuner(nn.Module):
    """
    ResNet50 model with fine-tuning .
    This class implements pre-trained ResNet50 with customizable freezing options.
    """
    def __init__(self, num_classes=10, dense_neurons=512, dropout_rate=0.0, 
                 freeze_option=1, dense_activation="relu"):
        """
        Initialize the ResNet50 fine-tuning model.
        
        Args:
            num_classes: Number of output classes
            dense_neurons: Size of the intermediate dense layer (after ResNet50 features, before final classification)
            dropout_rate: Dropout rate for dense layer
            freeze_option: Freezing strategy:
                           0 = unfreeze fc only
                           1 = unfreeze fc + last conv block
                           2 = unfreeze all layers
            dense_activation: Activation function for dense layer
        """
        super(ResNet50FineTuner, self).__init__()
        
    
        self.dense_activation_name = dense_activation
        
        #pre-trained ResNet50
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Save the number of features in the final layer
        num_ftrs = self.model.fc.in_features
        
        # Replace the final fully connected layer with custom layers
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, dense_neurons),
            self._get_activation(self.dense_activation_name),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_neurons, num_classes)
        )
        
        # freezing strategy
        self._apply_freezing(freeze_option)
        
    def _get_activation(self, activation_name="relu"):
        """Get activation function by name"""
        activations = {
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(inplace=True),
            "mish": nn.Mish(inplace=True)
        }
        return activations.get(activation_name, nn.ReLU(inplace=True))
    
    def _apply_freezing(self, freeze_option):
        """
        Apply the specified freezing strategy.
        
        Args:
            freeze_option: 0 = unfreeze fc only
                          1 = unfreeze fc + last conv block
                          2 = unfreeze all layers
        """
        # First freeze all parameters
        if freeze_option < 2:  # No need to freeze if option is 2 (unfreeze everything)
            for param in self.model.parameters():
                param.requires_grad = False
        
        #Then unfreeze according to the option
        if freeze_option == 0:
            # Unfreeze only the fully connected layer
            for param in self.model.fc.parameters():
                param.requires_grad = True
                
        elif freeze_option == 1:
            # Unfreeze the fully connected layer and the last conv block
            for param in self.model.layer4.parameters():
                param.requires_grad = True
            for param in self.model.fc.parameters():
                param.requires_grad = True
                
        # If freeze_option is 2, all parameters are already unfrozen
    
    def forward(self, x):
        """Forward pass through the model"""
        return self.model(x)
    
    def count_parameters(self):
        """Count the number of trainable parameters"""
        return {
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "total": sum(p.numel() for p in self.parameters())
        }
