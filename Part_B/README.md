# Part B | Finetuning ResNet for iNaturalist dataset

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

### Part B: Fine-tuning a Pre-trained ResNet50

```bash
python main.py \
  --data_dir path/to/dataset \
  --model_type resnet50 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --max_epochs 10 \
  --dense_neurons 512 \
  --freeze_option 2 \
  --dropout_rate 0.2 \
  --use_augmentation
```

### Fine-tuning Strategies

The `--freeze_option` parameter controls which parts of the ResNet50 model are fine-tuned:

- **0**: Fine-tune only the fully connected (FC) layer
- **1**: Fine-tune FC layer + last convolutional block
- **2**: Fine-tune all layers (full fine-tuning)

## Hyperparameter Tuning

To run a hyperparameter sweep using Weights & Biases:

```bash
python main.py \
  --data_dir path/to/dataset \
  --run_sweep \
  --sweep_count 60 \
  --project_name your_project_name
```

The sweep configuration is defined in `sweep_config.py` and can be modified to search different hyperparameter spaces.

Sweep Configurations that gave the best valid accuracy score for ResNet50:  (Run: resnet50_bs_128_dn512_fr2_ep10_lr0.0001)

- 'base_filters':[32] 

- 'filter_config':['halving']

- 'filter_sizes':[3, 3, 5, 5, 7]  # Increasing sizes
  
- 'activation': ['relu']
            
- 'dense_activation':['relu']

- 'dense_neurons':[512]

####### Regularization parameters

- 'batch_norm':[True]
  
- 'dropout_rate':[0.2]
            
####### Data augmentation

- 'use_augmentation': [false]
            
####### Training parameters
- 'image_size': [[224, 224]] 
      
- 'batch_size': [128]

- 'learning_rate': [1e-4]

- 'weight_decay': [0.005]

- 'max_epochs': [10]
                
- 'use_mixed_precision': True

- 'freeze_option': 2

## Model Architectures

### ResNet50

The ResNet50 fine-tuning implementation:

- Uses pre-trained weights from ImageNet
- Allows different freezing strategies
- Replaces the final classification layer
- Supports customizable dense layer sizes and dropout

## Experiment Tracking

The project uses Weights & Biases for experiment tracking. Each run logs:

- Training/validation/test metrics
- Model architecture details
- Hyperparameters
- Example predictions and visualizations

## Note

Work done for the course Introduction to Deep Learning. Please raise an issue if the code doesn't work propely for any case.
