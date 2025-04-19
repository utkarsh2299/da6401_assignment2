# Part A | training CNN for iNaturalist dataset using pytorch-lightning

This repository contains code for training and fine-tuning deep learning models on the iNaturalist dataset. 

## Project Structure

```
.
├── main.py             # Main script for training models
├── model.py            # Model architecture definitions (CNN and ResNet50FineTuner)
├── dataset.py          # Data loading and augmentation utilities
├── trainer.py          # PyTorch Lightning training implementation
├── test.py             # Evaluation script for trained models
├── sweep_config.py     # Configuration for hyperparameter sweeps
└── requirements.txt    # Python dependencies
```


## Training Models

### Part A: Training a CNN from Scratch

```bash
python main.py \
  --data_dir path/to/inaturalist_dataset \
  --model_type cnn \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --max_epochs 15 \
  --num_blocks 5 \
  --base_filters 128 \
  --filter_config fixed \
  --filter_sizes 3 3 5 5 7 \
  --activation mish \
  --dense_neurons 512 \
  --use_augmentation
```

## Hyperparameter Tuning

To run a hyperparameter sweep using Weights & Biases:

```bash
python main.py \
  --data_dir path/to/dataset \
  --run_sweep \
  --model_type "cnn" \
  --sweep_count 60 \
  --project_name your_project_name
```

The sweep configuration is defined in `sweep_config.py` and can be modified to search different hyperparameter spaces.

The best parameters are here:

- 'base_filters':[128] 

- 'filter_config':['fixed']

- 'filter_sizes':[3, 3, 5, 5, 7]  # Increasing sizes
  
- 'activation': ['mish']
            
- 'dense_activation':['relu']

- 'dense_neurons':[512]

####### Regularization parameters

- 'batch_norm':[True]
  
- 'dropout_rate':[0]
            
####### Data augmentation

- 'use_augmentation': [True]
            
####### Training parameters
- 'image_size': [[224, 224]] 
      
- 'batch_size': [16]

- 'learning_rate': [1e-4]

- 'weight_decay': [0.005]

- 'max_epochs': [15]
                
- 'use_mixed_precision': True


## Model Architectures

### CNN

The custom CNN architecture supports various configurations:

- Variable number of convolutional blocks
- Different filter configurations (fixed, doubling, halving)
- Custom filter sizes per layer
- Choice of activation functions
- Optional batch normalization


## Note

Work done for the course Introduction to Deep Learning. Please raise an issue if the code doesn't work propely for any case.
