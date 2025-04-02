from dataclasses import dataclass

@dataclass
class ModelConfig:
    num_layers: int = 555
    input_channels: int = 3
    initial_filters: int = 64
    filter_size: int = 3
    activation: str = 'relu'
    dense_neurons: int = 512
    num_classes: int = 10
    
    