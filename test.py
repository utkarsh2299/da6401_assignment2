import os
import torch
import argparse
import wandb
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from collections import defaultdict

from dataset import create_dataloaders
from model import CNN, ResNet50FineTuner  # Import both model types

def parse_args():
    parser = argparse.ArgumentParser(description="Test your best model configuration on the test set")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./test_results", help="Directory to save results")
    parser.add_argument("--wandb_project", type=str, default="da6401_assignment2", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default="best_model_test", help="W&B run name")
    parser.add_argument("--checkpoint", type=str, help="Path to best model checkpoint (optional)")
    parser.add_argument("--wandb_log", action="store_true", help="Whether to log to W&B")
    
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
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for testing")
    
    return parser.parse_args()

def test_model(model, test_loader, device, max_samples_per_class=10):
    """Test the model on the test set and collect results with samples from each class"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # Store samples grouped by class
    class_samples = defaultdict(list)
    
    correct = 0
    total = 0
    test_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    # First pass: get prediction stats and collect samples
    print("Evaluating model and collecting sample images...")
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
            
            # Store sample images from each class
            for i in range(len(images)):
                true_label = labels[i].item()
                # Only store up to max_samples_per_class per class
                if len(class_samples[true_label]) < max_samples_per_class:
                    class_samples[true_label].append({
                        'image': images[i].cpu().clone(),
                        'true': true_label,
                        'pred': predicted[i].item()
                    })
    
    # Print class distribution in the collected samples
    print("\nCollected samples by class:")
    for class_id, samples in class_samples.items():
        print(f"  Class {class_id}: {len(samples)} samples")
    
    # Aggregate samples for visualization
    sample_images = []
    for samples in class_samples.values():
        sample_images.extend(samples)
    
    accuracy = 100.0 * correct / total
    avg_loss = test_loss / total
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f"Test Loss: {avg_loss:.4f}")
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'sample_images': sample_images,
        'class_samples': class_samples
    }

def create_aligned_class_grid(results, class_names, output_dir):
    """Create a properly aligned grid where each row contains samples from a different class"""
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import matplotlib.gridspec as gridspec
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get class samples
    class_samples = results['class_samples']
    
    # Normalization parameters used in dataset
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Select classes to display (up to 10)
    classes_to_display = sorted(list(class_samples.keys()))[:10]
    
    # Create figure with GridSpec for better alignment
    fig = plt.figure(figsize=(15, 30))
    
    # Create a GridSpec with 10 rows and 4 columns
    # First column for class labels, remaining 3 for images
    gs = gridspec.GridSpec(len(classes_to_display), 4, figure=fig)
    
    # Plot each class in a separate row
    for row, class_id in enumerate(classes_to_display):
        samples = class_samples[class_id][:3]  # Get up to 3 samples
        
        # Add class name in its own cell
        class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"Class {class_id}"
        ax_label = fig.add_subplot(gs[row, 0])
        ax_label.text(0.5, 0.5, class_name, 
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=12, fontweight='bold')
        ax_label.axis('off')
        
        # Plot samples for this class in the remaining cells
        for col, sample in enumerate(samples):
            ax = fig.add_subplot(gs[row, col+1])  # +1 because first column is for labels
            
            # Get image and denormalize it
            img = sample['image']
            img = img * std + mean  # Denormalize
            img = img.permute(1, 2, 0).numpy()  # Convert to HWC format
            img = np.clip(img, 0, 1)  # Clip values to valid range
            
            # Get prediction information
            true_label = sample['true']
            pred_label = sample['pred']
            is_correct = true_label == pred_label
            
            # Get predicted class name
            pred_name = class_names[pred_label] if class_names and pred_label < len(class_names) else f"Class {pred_label}"
            
            # Plot the image
            ax.imshow(img)
            
            # Set title with color indicating correctness
            color = 'green' if is_correct else 'red'
            title = f"Pred: {pred_name}"
            ax.set_title(title, color=color)
            
            # Add colored border
            for spine in ax.spines.values():
                spine.set_linewidth(3)
                spine.set_color(color)
            
            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Handle any remaining empty plots in this row
        for col in range(len(samples), 3):
            ax = fig.add_subplot(gs[row, col+1])
            ax.axis('off')
    
    # Add a big title
    plt.suptitle("Test Set Predictions by Class (Green = Correct, Red = Incorrect)", fontsize=16, y=0.995)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Make room for suptitle
    
    # Save the figure
    grid_path = os.path.join(output_dir, "class_prediction_grid.png")
    plt.savefig(grid_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved aligned class-based visualization grid to {grid_path}")
    return grid_path

def create_confusion_matrix(results, class_names, num_classes, output_dir):
    """Create and save a confusion matrix visualization"""
    labels = results['labels']
    predictions = results['predictions']
    
    # Create confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # For visualization, limit to first 20 classes if there are many
    max_classes = min(20, num_classes)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm[:max_classes, :max_classes],
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=range(max_classes),
        yticklabels=range(max_classes)
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Save the figure
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved confusion matrix to {cm_path}")
    return cm_path

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
    
    # Initialize W&B if logging is enabled
    if args.wandb_log:
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
        print(f"Found {len(class_names)} classes: {class_names}")
    else:
        print("No class names found")
    
    # Create dataloaders
    data_loaders = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        val_split=0.2,  # Not used for testing
        use_augmentation=False,  # No augmentation for test set
        image_size=tuple(args.image_size)
    )
    
    test_loader = data_loaders['test']
    num_classes = data_loaders['num_classes']
    
    print(f"Dataset loaded with {num_classes} classes")
    print(f"Test loader has {len(test_loader.dataset)} samples")
    
    # Analyze the test set class distribution
    all_labels = []
    for _, labels in test_loader:
        all_labels.extend(labels.numpy())
    
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    print("\nTest set class distribution:")
    for i, (label, count) in enumerate(zip(unique_labels, counts)):
        class_name = class_names[label] if class_names and label < len(class_names) else f"Class {label}"
        print(f"  {class_name}: {count} samples")
    
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
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                # PyTorch Lightning format
                state_dict = checkpoint['state_dict']
                
                # Remove 'model.' prefix if needed
                if any(k.startswith('model.') for k in state_dict.keys()):
                    if not hasattr(model, 'model'):
                        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
                
                model.load_state_dict(state_dict, strict=False)
                print("Successfully loaded checkpoint with state_dict")
            else:
                # Direct model state dict
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
    
    # Create visualizations
    grid_path = create_aligned_class_grid(results, class_names, args.output_dir)

    cm_path = create_confusion_matrix(results, class_names, num_classes, args.output_dir)
    
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
    
    # Log to wandb if enabled
    if args.wandb_log:
        # Create a table for class accuracies
        class_table = wandb.Table(columns=["Class", "Accuracy (%)"])
        for class_name, accuracy in per_class_acc.items():
            class_table.add_data(class_name, accuracy)
        
        # Log metrics and visualizations
        wandb.log({
            "test_accuracy": results['accuracy'],
            "test_loss": results['loss'],
            "prediction_grid": wandb.Image(grid_path),
            "confusion_matrix": wandb.Image(cm_path),
            "class_accuracies": class_table,
            **class_accuracies  # Log individual class accuracies
        })
        
        # Finish the wandb run
        wandb.finish()
    
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

if __name__ == "__main__":
    main()