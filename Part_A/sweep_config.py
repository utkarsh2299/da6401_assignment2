def get_sweep_config():
    """
    Get the configuration for the W&B hyperparameter sweep.
    includes parameters for both CNN and ResNet50 models.
    Returns:
        dict: W&B sweep configuration
    """
    sweep_config = {
        'method': 'bayes',  # or 'grid', 'random'
        'metric': {
            'name': 'val_acc',
            'goal': 'maximize'
        },
        'parameters': {
            # Model architecture parameters
            'num_blocks': {
                'value': 5  # Fixed as per assignment requirement
            },
            'model_type': {
                # 'values': ['cnn', 'resnet50']
                'values': ['resnet50']
            },
            'base_filters': {
                'values': [32, 64, 128]
            },
            'filter_config': {
                'values': ['fixed', 'doubling', 'halving']
            },
            'filter_sizes': {
                'values': [
                    [7, 5, 5, 3, 3],  # Decreasing sizes
                    [3, 3, 5, 5, 7],  # Increasing sizes
                    [3, 3, 3, 3, 3],  # Uniform small
                    [5, 5, 5, 5, 5],  # Uniform medium
                    [7, 7, 7, 7, 7]   # Uniform large
                ]
            },
            'activation': {
                'values': ['relu', 'gelu', 'silu', 'mish']
            },
            'dense_activation':{
                'values': ['relu']
            },
            'dense_neurons': {
                'values': [1024, 256, 512]
            },
            
            # Regularization parameters
            'batch_norm': {
                'values': [True, False]
            },
            'dropout_rate': {
                'values': [0, 0.2, 0.3]
            },
            
            # Data augmentation
            'use_augmentation': {
                'values': [True, False]
            },
            
            # Training parameters
             'image_size': {
                'value': [224, 224]  
            },
            'batch_size': {
                'values': [32, 64,16,128]
            },
            'learning_rate': {
                 'values': [1e-3,1e-4]
            },
            'weight_decay': {
                 'values': [0,0.5,0.005]
            },
            'max_epochs': {
                # 'values': [10,15]
                'values': [10,5]
            },
            'use_mixed_precision': {
                'value': True
            },
            # ResNet50-specific parameters
            'freeze_option': {
                'values': [0, 1, 2]
            }
        }
    }
    
    return sweep_config
