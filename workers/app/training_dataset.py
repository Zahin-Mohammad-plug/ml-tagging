"""
PyTorch Dataset for Multi-Label Image Classification
Loads images from manifest and applies preprocessing transforms
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class MultiLabelImageDataset(Dataset):
    """Dataset for multi-label image classification with preprocessing"""
    
    def __init__(
        self,
        manifest_path: str,
        labels_path: str,
        split_indices: List[int],
        transform: Optional[transforms.Compose] = None,
        image_size: int = 224,
        augment: bool = False
    ):
        """
        Args:
            manifest_path: Path to dataset_manifest.csv
            labels_path: Path to labels.npy
            split_indices: List of image indices for this split (train/val/test)
            transform: Custom transform (if None, uses default)
            image_size: Target image size (default 224 for EfficientNet-B0)
            augment: Apply data augmentation (for training only)
        """
        self.manifest_path = Path(manifest_path)
        self.labels_path = Path(labels_path)
        self.split_indices = split_indices
        self.image_size = image_size
        self.augment = augment
        
        # Load manifest
        self.images = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.images.append(row)
        
        # Load labels
        self.all_labels = np.load(labels_path)
        
        # Filter to split
        self.images = [self.images[i] for i in split_indices]
        self.labels = self.all_labels[split_indices]
        
        # Set up transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transform(augment)
    
    def _get_default_transform(self, augment: bool) -> transforms.Compose:
        """Get default preprocessing transforms"""
        if augment:
            # Training augmentation
            return transforms.Compose([
                transforms.Resize(int(self.image_size * 1.15)),  # Slightly larger for random crop
                transforms.RandomCrop(self.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.1,
                    hue=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            # Validation/test (no augmentation)
            return transforms.Compose([
                transforms.Resize(int(self.image_size * 1.15)),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get image and label tensor
        
        Returns:
            image: Tensor of shape (3, image_size, image_size)
            label: Binary label vector of shape (num_tags,)
        """
        img_data = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        img_path = Path(img_data['original_path'])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (self.image_size, self.image_size), color='black')
        
        # Apply transforms
        image = self.transform(image)
        
        # Convert label to tensor
        label = torch.from_numpy(label).float()
        
        return image, label


def get_data_loaders(
    dataset_dir: str,
    batch_size: int = 8,
    num_workers: int = 2,
    image_size: int = 224
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test data loaders
    
    Args:
        dataset_dir: Directory containing dataset files
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        image_size: Target image size
        
    Returns:
        train_loader, val_loader, test_loader
    """
    dataset_dir = Path(dataset_dir)
    
    # Load splits
    with open(dataset_dir / 'splits.json', 'r') as f:
        splits = json.load(f)
    
    manifest_path = dataset_dir / 'dataset_manifest.csv'
    labels_path = dataset_dir / 'labels.npy'
    
    # Create datasets
    train_dataset = MultiLabelImageDataset(
        manifest_path=manifest_path,
        labels_path=labels_path,
        split_indices=splits['train'],
        image_size=image_size,
        augment=True  # Training augmentation
    )
    
    val_dataset = MultiLabelImageDataset(
        manifest_path=manifest_path,
        labels_path=labels_path,
        split_indices=splits['val'],
        image_size=image_size,
        augment=False
    )
    
    test_dataset = MultiLabelImageDataset(
        manifest_path=manifest_path,
        labels_path=labels_path,
        split_indices=splits['test'],
        image_size=image_size,
        augment=False
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch for consistent batch size
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"✅ Created data loaders:")
    print(f"   Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"   Val:   {len(val_dataset)} images, {len(val_loader)} batches")
    print(f"   Test:  {len(test_dataset)} images, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test dataset loading
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset_dir='../../trainData',
        batch_size=8,
        num_workers=0  # Use 0 for testing
    )
    
    # Test loading a batch
    for images, labels in train_loader:
        print(f"\nBatch shapes:")
        print(f"  Images: {images.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  Labels per image: {labels.sum(dim=1).mean():.2f}")
        break
