import os
import torch
import argparse
import wandb
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
# from sklearn.metrics import confusion_matrix
# import seaborn as sns

from dataset import create_dataloaders
from model import CNN, ResNet50FineTuner  # Import both model types

def parse_args():
    parser = argparse.ArgumentParser(description="Test your best model configuration on the test set")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./test_results", help="Directory to save results")
    parser.add_argument("--wandb_project", type=str, default="da6401_assignment2", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default="best_model_test", help="W&B run name")
    parser.add_argument("--checkpoint", type=str, help="Path to best model checkpoint (optional)")
    
    # Model type selection
    parser.add_argument("--model_type", type=str, default="cnn", choices=["cnn", "resnet50"],
                       help="Type of model to test (CNN or ResNet50)")
    
    # CNN model configuration params
    parser.add_argument("--num_blocks", type=int, default=5, help="Number of conv blocks")
    parser.add_argument("--base_filters", type=int, default=128, help="Base number of filters from best model")
    parser.add_argument("--filter_config", type=str,  default="fixed", help="Filter configuration from best model")
    parser.add_argument("--filter_sizes", type=int, nargs="+", default=[3, 3, 5, 5, 7], help="Filter sizes from best model")
    parser.add_argument("--activation", type=str, default="mish", help="Activation function from best model")
    parser.add_argument("--dense_activation", type=str, default="relu", help="Dense activation from best model")
    parser.add_argument("--dense_neurons", type=int, default=512, help="Number of dense neurons from best model")
    parser.add_argument("--batch_norm", action="store_true", help="Use batch normalization if in best model")
    parser.add_argument("--dropout_rate", type=float, default=0, help="Dropout rate from best model")
    
    # ResNet50 specific parameters
    parser.add_argument("--freeze_option", type=int, default=1, choices=[0, 1, 2],
                      help="Freeze option for ResNet50 (0=fc only, 1=fc+last block, 2=all)")
    
    # Other params
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224], help="Image size (height, width)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for testing")
    
    return parser.parse_args()

def test_model(model, test_loader, device):
    """Test the model on the test set and collect results"""
    model.eval()
    
    all_preds = []
    all_labels = []
    sample_images = []
    
    correct = 0
    total = 0
    test_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Testing")):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs, 1)
            
            # Update accuracy count
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and true labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Store some sample images for visualization (up to 30 for a 10x3 grid)
            if len(sample_images) < 30:
                for i in range(min(len(images), 30 - len(sample_images))):
                    # Store the image, true label, and prediction
                    sample_images.append({
                        'image': images[i].cpu().clone(),
                        'true': labels[i].item(),
                        'pred': predicted[i].item()
                    })
    
    accuracy = 100.0 * correct / total
    avg_loss = test_loss / total
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f"Test Loss: {avg_loss:.4f}")
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'sample_images': sample_images
    }

def create_visualization_grid(results, class_names, output_dir):
    """Create a 10x3 grid of test images with predictions"""
    # Create the figure
    fig, axes = plt.subplots(10, 3, figsize=(15, 30))
    plt.subplots_adjust(hspace=0.3)
    
    # Prepare grid of images
    sample_images = results['sample_images']
    
    # Normalization parameters used in dataset
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Plot each image
    for i, sample in enumerate(sample_images):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        # Get image and denormalize it
        img = sample['image']
        img = img * std + mean  # Denormalize
        img = img.permute(1, 2, 0).numpy()  # Convert to HWC format
        img = np.clip(img, 0, 1)  # Clip values to valid range
        
        # Get true and predicted labels
        true_label = sample['true']
        pred_label = sample['pred']
        is_correct = true_label == pred_label
        
        # Get class names if available
        true_name = class_names[true_label] if class_names and true_label < len(class_names) else f"Class {true_label}"
        pred_name = class_names[pred_label] if class_names and pred_label < len(class_names) else f"Class {pred_label}"
        
        # Plot the image
        ax.imshow(img)
        
        # Set title with color indicating correctness
        color = 'green' if is_correct else 'red'
        title = f"True: {true_name}\nPred: {pred_name}"
        ax.set_title(title, color=color)
        
        # Add colored border
        for spine in ax.spines.values():
            spine.set_linewidth(3)
            spine.set_color(color)
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Handle any remaining empty plots
    for i in range(len(sample_images), 30):
        row, col = i // 3, i % 3
        axes[row, col].axis('off')
    
    # Add a big title
    plt.suptitle("Test Set Predictions (Green = Correct, Red = Incorrect)", fontsize=16, y=0.995)
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    grid_path = os.path.join(output_dir, "prediction_grid.png")
    plt.savefig(grid_path, bbox_inches='tight')
    
    # Create a wandb compatible figure for direct logging
    wandb_img = wandb.Image(
        grid_path,
        caption="Prediction Grid (Green = Correct, Red = Incorrect)"
    )
    
    print(f"Saved visualization grid to {grid_path}")
    return grid_path, wandb_img

def get_class_names(data_dir):
    """Get class names from dataset directory"""
    test_dir = os.path.join(data_dir, 'test')
    if os.path.exists(test_dir):
        class_names = sorted([d for d in os.listdir(test_dir) 
                             if os.path.isdir(os.path.join(test_dir, d))])
        return class_names
    return None

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize W&B with configuration based on model type
    config = {
        "model_type": args.model_type,
        "dense_neurons": args.dense_neurons,
        "dropout_rate": args.dropout_rate,
        "image_size": args.image_size,
        "evaluation": "test_set"
    }
    
    # Add model-specific parameters
    if args.model_type == "cnn":
        config.update({
            "num_blocks": args.num_blocks,
            "base_filters": args.base_filters,
            "filter_config": args.filter_config,
            "filter_sizes": args.filter_sizes,
            "activation": args.activation,
            "dense_activation": args.dense_activation,
            "batch_norm": args.batch_norm
        })
    else:  # ResNet50
        config.update({
            "freeze_option": args.freeze_option,
            "dense_activation": args.dense_activation
        })
    
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=config
    )
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get class names
    class_names = get_class_names(args.data_dir)
    if class_names:
        print(f"Found {len(class_names)} classes")
    
    # Create dataloaders - we only need the test loader
    data_loaders = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        val_split=0.2,  # Not used for testing
        use_augmentation=False,  # No augmentation for test set
        image_size=tuple(args.image_size)
    )
    
    test_loader = data_loaders['test']
    num_classes = data_loaders['num_classes']
    
    # Create model based on model type
    if args.model_type == "cnn":
        print("Creating CNN model...")
        model = CNN(
            input_channels=3,
            input_size=tuple(args.image_size),
            num_classes=num_classes,
            num_blocks=args.num_blocks,
            filter_config=args.filter_config,
            base_filters=args.base_filters,
            filter_sizes=args.filter_sizes,
            activation=args.activation,
            dense_neurons=args.dense_neurons,
            batch_norm=args.batch_norm,
            dropout_rate=args.dropout_rate,
            dense_activation=args.dense_activation
        )
    else:  # ResNet50
        print("Creating ResNet50 model...")
        model = ResNet50FineTuner(
            num_classes=num_classes,
            dense_neurons=args.dense_neurons,
            dropout_rate=args.dropout_rate,
            freeze_option=args.freeze_option,
            dense_activation=args.dense_activation
        )
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading weights from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            # PyTorch Lightning format
            state_dict = checkpoint['state_dict']
            
            # Handle ResNet50 model's structure if needed
            if args.model_type == "resnet50":
                # If state_dict keys have 'model.' prefix but model doesn't
                if any(k.startswith('model.') for k in state_dict.keys()):
                    if not hasattr(model, 'model'):
                        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
                
                # If checkpoint has 'model.model' but our ResNet50FineTuner has just 'model'
                if any(k.startswith('model.model.') for k in state_dict.keys()):
                    state_dict = {k.replace('model.model.', 'model.'): v for k, v in state_dict.items()}
            
            # For both models, remove 'model.' prefix if needed
            if hasattr(model, 'model') and not any(k.startswith('model.') for k in state_dict.keys()):
                state_dict = {f"model.{k}": v for k, v in state_dict.items()}
            
            try:
                model.load_state_dict(state_dict, strict=False)
                print("Successfully loaded checkpoint with state_dict")
            except Exception as e:
                print(f"Warning: Error loading checkpoint: {e}")
                print("Trying alternative loading method...")
                
                # Try alternative loading method
                try:
                    # Create a new state dict by matching parameter shapes
                    model_dict = model.state_dict()
                    pretrained_dict = {k: v for k, v in state_dict.items() 
                                      if k in model_dict and v.shape == model_dict[k].shape}
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict)
                    print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters using shape matching")
                except Exception as e2:
                    print(f"Error with alternative loading method: {e2}")
                    print("Continuing with initialized weights")
        else:
            # Direct model state dict
            try:
                model.load_state_dict(checkpoint, strict=False)
                print("Successfully loaded checkpoint")
            except Exception as e:
                print(f"Warning: Error loading checkpoint: {e}")
                print("Continuing with initialized weights")
    else:
        print("No checkpoint provided. Testing with initialized weights.")
    
    model = model.to(device)
    
    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test the model
    print("\nEvaluating model on test set...")
    results = test_model(model, test_loader, device)
    
    # Create visualizations with direct wandb logging
    grid_path, grid_wandb = create_visualization_grid(results, class_names, args.output_dir)
    
    # Calculate per-class accuracy
    per_class_acc = {}
    class_accuracies = {}
    
    for c in np.unique(results['labels']):
        indices = results['labels'] == c
        correct = np.sum(results['predictions'][indices] == c)
        total = np.sum(indices)
        accuracy = 100.0 * correct / total
        
        class_name = class_names[c] if class_names and c < len(class_names) else f"Class {c}"
        per_class_acc[class_name] = accuracy
        class_accuracies[f"class_{c}_accuracy"] = accuracy
    
    # Print per-class accuracy
    print("\nPer-class accuracy:")
    for class_name, accuracy in per_class_acc.items():
        print(f"{class_name}: {accuracy:.2f}%")
    
    # Create a table for class accuracies
    class_table = wandb.Table(columns=["Class", "Accuracy (%)"])
    for class_name, accuracy in per_class_acc.items():
        class_table.add_data(class_name, accuracy)
    
    # Log everything to W&B
    wandb.log({
        "test_accuracy": results['accuracy'],
        "test_loss": results['loss'],
        "prediction_grid": grid_wandb,
        "class_accuracies": class_table,
        **class_accuracies  # Log individual class accuracies
    })
    
    # Save test results to a file
    with open(os.path.join(args.output_dir, "test_results.txt"), "w") as f:
        f.write(f"Model type: {args.model_type}\n")
        f.write(f"Test Accuracy: {results['accuracy']:.2f}%\n")
        f.write(f"Test Loss: {results['loss']:.4f}\n\n")
        f.write("Per-class accuracy:\n")
        for class_name, accuracy in per_class_acc.items():
            f.write(f"{class_name}: {accuracy:.2f}%\n")
    
    print(f"\nAll results saved to {args.output_dir}")
    print(f"Test accuracy: {results['accuracy']:.2f}%")
    print(f"Test loss: {results['loss']:.4f}")
    print(f"Results also logged to W&B project: {args.wandb_project}, run: {wandb.run.name}")
    
    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()