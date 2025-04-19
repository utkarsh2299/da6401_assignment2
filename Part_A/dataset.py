import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import numpy as np
import torch

class INaturalistDataset(Dataset):
    """Custom dataset for iNaturalist dataset."""
    def __init__(self, root_dir, transform=None, split='train'):
        """
        Args:
            root_dir (string): Directory with all the images organized in class folders
            transform (callable, optional): Optional transform to be applied on a sample
            split (string): 'train', 'val', or 'test'
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.split = split
        
        # Getting all class folders
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Getting all image paths and labels
        self.image_paths = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_dataloaders(data_dir, batch_size=32, val_split=0.2, 
                       use_augmentation=True, num_workers=4, image_size=(224, 224)):
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir (string): Root directory for the dataset
        batch_size (int): Batch size for training
        val_split (float): Proportion of training data to use for validation
        use_augmentation (bool): Whether to use data augmentation
        num_workers (int): Number of worker  for loading data
        image_size (tuple): Desired image size (height, width)
    Returns:
        dict: Contains train, val, and test dataloaders
    """
    # Let's define various transforms  using torch vision's  transforms
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((int(image_size[0]*1.14), int(image_size[1]*1.14))),  # Resize to slightly larger
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((int(image_size[0]*1.14), int(image_size[1]*1.14))),  # Resize to slightly larger
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset instances
    train_dataset = INaturalistDataset(data_dir, transform=train_transform, split='train')
    test_dataset = INaturalistDataset(data_dir, transform=val_transform, split='test')
    
    # Create a stratified validation split to ensure each class in dataset is proportionally represented in both the training and testing sets
    train_indices, val_indices = stratified_split(train_dataset.labels, val_split)
    
    #Since we have split train in to Train and valid set, we will refer the same location based on indices instead of creating any copy for val
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'num_classes': len(train_dataset.classes), 'image_size': image_size}


def stratified_split(labels, val_split=0.2):
    """
    Create a stratified split, ensuring each class is equally represented.
    
    Args:
        labels (list): List of class labels
        val_split (float): Proportion to use for validation
        
    Returns:
        tuple: (train_indices, val_indices)
    """
    labels = np.array(labels)
    train_indices, val_indices = train_test_split(
        np.arange(len(labels)),
        test_size=val_split,
        stratify=labels,
        random_state=42
    )
    return train_indices, val_indices