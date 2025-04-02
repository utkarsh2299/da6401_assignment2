import torch
import torch.nn as nn

class FlexibleCNN(nn.Module):
    def __init__(self, config):
        super(FlexibleCNN, self).__init__()
        
        # Activation function mapping
        activation_map = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
        }
        activation_func = activation_map.get(config.activation, nn.ReLU())
        
        # Create convolution blocks dynamically
        self.conv_blocks = nn.ModuleList()
        current_channels = config.input_channels
        current_filters = config.initial_filters
        
        for i in range(config.num_layers):
            # Convolution layer
            conv = nn.Conv2d(current_channels, current_filters, 
                             kernel_size=config.filter_size, 
                             padding=config.filter_size//2)
            
            # Max pooling layer
            pool = nn.MaxPool2d(kernel_size=2, stride=2)
            
            # Add conv block
            self.conv_blocks.append(
                nn.Sequential(
                    conv,
                    activation_func,
                    pool
                )
            )
            
            # Update channels for next layer
            current_channels = current_filters
            current_filters = min(current_filters * 2, 512)  # Cap at 512 filters
        
        # Estimate output size after convolutions
        dummy_input = torch.randn(1, config.input_channels, 224, 224)
        with torch.no_grad():
            x = dummy_input
            for block in self.conv_blocks:
                x = block(x)
            
            flattened_size = x.view(x.size(0), -1).size(1)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, config.dense_neurons),
            activation_func,
            nn.Linear(config.dense_neurons, config.num_classes)
        )
    
    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x