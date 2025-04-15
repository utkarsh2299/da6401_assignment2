
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
from model import CNN  # Replace with your actual model import

def parse_args():
    parser = argparse.ArgumentParser(description="Test your best model configuration on the test set")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./test_results", help="Directory to save results")
    parser.add_argument("--wandb_project", type=str, default="da6401_assignment2", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default="best_model_test", help="W&B run name")
    parser.add_argument("--checkpoint", type=str, help="Path to best model checkpoint (optional)")
    
    # Best model configuration params
    parser.add_argument("--num_blocks", type=int, default=5, help="Number of conv blocks")
    parser.add_argument("--base_filters", type=int, default=32, help="Base number of filters from best model")
    parser.add_argument("--filter_config", type=str,  default="fixed", help="Filter configuration from best model")
    parser.add_argument("--filter_sizes", type=int, nargs="+", default=[3, 3, 3, 3, 3], help="Filter sizes from best model")
    parser.add_argument("--activation", type=str, default="relu", help="Activation function from best model")
    parser.add_argument("--dense_activation", type=str, default="relu", help="Dense activation from best model")
    parser.add_argument("--dense_neurons", type=int, default=128, help="Number of dense neurons from best model")
    parser.add_argument("--batch_norm", action="store_true", help="Use batch normalization if in best model")
    parser.add_argument("--dropout_rate", type=float, default=0, help="Dropout rate from best model")
    
    # Other params
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224], help="Image size (height, width)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for testing")
    
    return parser.parse_args()

def test_model(model, test_loader, device):
    """Test the model on the test set and collect results"""
    model.eval()
    
    all_preds = []
    all_labels = []
    sample_images = []
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Testing")):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
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
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    
    return {
        'accuracy': accuracy,
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

# def create_confusion_matrix(results, num_classes, class_names, output_dir):
#     """Create and save a confusion matrix"""
#     labels = results['labels']
#     predictions = results['predictions']
    
#     # Create confusion matrix
#     cm = confusion_matrix(labels, predictions)
    
#     # For visualization, limit to first 20 classes if there are many
#     max_classes = min(20, num_classes)
    
#     plt.figure(figsize=(12, 10))
#     sns.heatmap(
#         cm[:max_classes, :max_classes],
#         annot=True,
#         fmt='d',
#         cmap='Blues',
#         xticklabels=range(max_classes),
#         yticklabels=range(max_classes)
#     )
#     plt.title('Confusion Matrix')
#     plt.xlabel('Predicted Label')
#     plt.ylabel('True Label')
    
#     # Save the figure
#     cm_path = os.path.join(output_dir, "confusion_matrix.png")
#     plt.savefig(cm_path, bbox_inches='tight')
#     plt.close()
    
#     # Create a wandb compatible figure for direct logging
#     wandb_img = wandb.Image(
#         cm_path,
#         caption="Confusion Matrix"
#     )
    
#     print(f"Saved confusion matrix to {cm_path}")
#     return cm_path, wandb_img

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
    
    # Initialize W&B directly with your best model configuration
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "num_blocks": args.num_blocks,
            "base_filters": args.base_filters,
            "filter_config": args.filter_config,
            "filter_sizes": args.filter_sizes,
            "activation": args.activation,
            "dense_activation": args.dense_activation,
            "dense_neurons": args.dense_neurons,
            "batch_norm": args.batch_norm,
            "dropout_rate": args.dropout_rate,
            "image_size": args.image_size,
            "evaluation": "test_set"  # Indicate this is test set evaluation
        }
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
    
    # Create model with best configuration
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
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading weights from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            # PyTorch Lightning format
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if needed
            if all(k.startswith('model.') for k in state_dict.keys()):
                state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
        else:
            # Direct model state dict
            model.load_state_dict(checkpoint)
    else:
        print("No checkpoint provided. Testing with randomly initialized weights.")
        # Initialize with Xavier/Glorot initialization
        def init_weights(m):
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        model.apply(init_weights)
    
    model = model.to(device)
    
    # Test the model
    print("\nEvaluating model on test set...")
    results = test_model(model, test_loader, device)
    
    # Create visualizations with direct wandb logging
    grid_path, grid_wandb = create_visualization_grid(results, class_names, args.output_dir)
    # cm_path, cm_wandb = create_confusion_matrix(results, num_classes, class_names, args.output_dir)
    
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
        "prediction_grid": grid_wandb,
        "class_accuracies": class_table,
        **class_accuracies  # Log individual class accuracies
    })
    
    # Save test results to a file
    with open(os.path.join(args.output_dir, "test_results.txt"), "w") as f:
        f.write(f"Test Accuracy: {results['accuracy']:.2f}%\n\n")
        f.write("Per-class accuracy:\n")
        for class_name, accuracy in per_class_acc.items():
            f.write(f"{class_name}: {accuracy:.2f}%\n")
    
    print(f"\nAll results saved to {args.output_dir}")
    print(f"Test accuracy: {results['accuracy']:.2f}%")
    print(f"Results also logged to W&B project: {args.wandb_project}, run: {wandb.run.name}")
    
    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()