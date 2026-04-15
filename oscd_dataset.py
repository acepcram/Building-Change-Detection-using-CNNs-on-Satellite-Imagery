"""
OSCD Dataset Loader for Building Change Detection
Handles loading, preprocessing, and patch extraction from the OSCD dataset
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import rasterio
from sklearn.model_selection import train_test_split

class OSCDDataset(Dataset):
    """
    Dataset class for OSCD (Onera Satellite Change Detection)
    
    Args:
        root_dir: Path to OSCD dataset root directory
        split: 'train' or 'test'
        patch_size: Size of patches (default: 15)
        stride: Stride for patch extraction (default: 5 for training, 15 for testing)
        use_augmentation: Whether to use data augmentation (default: True for training)
        rgb_only: Use only RGB bands (default: True)
    """
    
    def __init__(self, root_dir, split='train', patch_size=15, stride=None, 
                 use_augmentation=True, rgb_only=True):
        self.root_dir = root_dir
        self.split = split
        self.patch_size = patch_size
        self.stride = stride if stride is not None else (5 if split == 'train' else 15)
        self.use_augmentation = use_augmentation and (split == 'train')
        self.rgb_only = rgb_only
        
        # OSCD train/test split as defined in the paper
        self.train_cities = [
            'abudhabi', 'aguasclaras', 'beihai', 'beirut', 'bercy',
            'bordeaux', 'nantes', 'paris', 'rennes', 'saclay_e',
            'pisa', 'rennes', 'rio', 'saclay_w'
        ]
        
        self.test_cities = [
            'brasilia', 'chongqing', 'cupertino', 'dubai', 'hongkong',
            'lasvegas', 'milano', 'montpellier', 'mumbai', 'norcia'
        ]
        
        # Select cities based on split
        if split == 'train':
            self.cities = self.train_cities
        else:
            self.cities = self.test_cities
            
        # Load all patches
        self.patches = []
        self._load_patches()
        
        print(f"Loaded {len(self.patches)} patches for {split} split")
        
    def _load_patches(self):
        """Extract patches from all images"""
        images_dir = os.path.join(self.root_dir, 'images')
        labels_dir = os.path.join(self.root_dir, 'labels')
        
        for city in self.cities:
            # Look for image files
            img1_path = os.path.join(images_dir, f"{city}_imgs_1.tif")
            img2_path = os.path.join(images_dir, f"{city}_imgs_2.tif")
            label_path = os.path.join(labels_dir, f"{city}_cm.tif")
            
            # Alternative naming conventions
            if not os.path.exists(img1_path):
                img1_path = os.path.join(images_dir, city, "imgs_1.tif")
                img2_path = os.path.join(images_dir, city, "imgs_2.tif")
                label_path = os.path.join(labels_dir, city, "cm.tif")
            
            if not os.path.exists(img1_path):
                print(f"Warning: Could not find images for {city}, skipping...")
                continue
                
            # Load images
            try:
                with rasterio.open(img1_path) as src:
                    img1 = src.read()  # Shape: (bands, height, width)
                with rasterio.open(img2_path) as src:
                    img2 = src.read()
                with rasterio.open(label_path) as src:
                    labels = src.read(1)  # Shape: (height, width)
            except Exception as e:
                print(f"Error loading {city}: {e}")
                continue
            
            # Use only RGB bands if specified (bands 2, 3, 4 for Sentinel-2)
            if self.rgb_only and img1.shape[0] >= 4:
                img1 = img1[1:4, :, :]  # Bands 2, 3, 4 (RGB)
                img2 = img2[1:4, :, :]
            
            # Normalize to [0, 1]
            img1 = img1.astype(np.float32) / 10000.0  # Sentinel-2 scale
            img2 = img2.astype(np.float32) / 10000.0
            
            # Clip values to [0, 1]
            img1 = np.clip(img1, 0, 1)
            img2 = np.clip(img2, 0, 1)
            
            # Convert labels to binary (0: no change, 1: change)
            labels = (labels > 0).astype(np.int64)
            
            # Extract patches
            height, width = labels.shape
            for y in range(0, height - self.patch_size + 1, self.stride):
                for x in range(0, width - self.patch_size + 1, self.stride):
                    # Extract patch
                    patch1 = img1[:, y:y+self.patch_size, x:x+self.patch_size]
                    patch2 = img2[:, y:y+self.patch_size, x:x+self.patch_size]
                    label = labels[y+self.patch_size//2, x+self.patch_size//2]  # Center pixel
                    
                    self.patches.append({
                        'patch1': patch1,
                        'patch2': patch2,
                        'label': label,
                        'city': city
                    })
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        """Get a single patch pair"""
        patch_data = self.patches[idx]
        
        patch1 = torch.from_numpy(patch_data['patch1'].copy())
        patch2 = torch.from_numpy(patch_data['patch2'].copy())
        label = torch.tensor(patch_data['label'], dtype=torch.long)
        
        # Data augmentation (only during training)
        if self.use_augmentation:
            # Random flip
            if np.random.rand() > 0.5:
                patch1 = torch.flip(patch1, [1])  # Horizontal flip
                patch2 = torch.flip(patch2, [1])
            if np.random.rand() > 0.5:
                patch1 = torch.flip(patch1, [2])  # Vertical flip
                patch2 = torch.flip(patch2, [2])
            
            # Random rotation (0, 90, 180, 270 degrees)
            k = np.random.randint(0, 4)
            if k > 0:
                patch1 = torch.rot90(patch1, k, [1, 2])
                patch2 = torch.rot90(patch2, k, [1, 2])
        
        return patch1, patch2, label


class OSCDDatasetSimple(Dataset):
    """
    Simplified OSCD Dataset loader that works with common file structures
    Assumes structure:
        root_dir/
            train/
                images/
                    city1_t1.tif
                    city1_t2.tif
                    ...
                labels/
                    city1.tif
                    ...
            test/
                images/
                labels/
    """
    
    def __init__(self, root_dir, split='train', patch_size=15, stride=None, 
                 use_augmentation=True):
        self.root_dir = root_dir
        self.split = split
        self.patch_size = patch_size
        self.stride = stride if stride is not None else (5 if split == 'train' else 15)
        self.use_augmentation = use_augmentation and (split == 'train')
        
        self.patches = []
        self._load_patches_simple()
        
        print(f"Loaded {len(self.patches)} patches for {split} split")
    
    def _load_patches_simple(self):
        """Simple patch extraction assuming preprocessed numpy files"""
        split_dir = os.path.join(self.root_dir, self.split)
        
        # Look for .npy files (preprocessed)
        if os.path.exists(os.path.join(split_dir, 'patches.npy')):
            data = np.load(os.path.join(split_dir, 'patches.npy'), allow_pickle=True)
            self.patches = list(data)
            return
        
        # Otherwise, process from images
        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')
        
        if not os.path.exists(images_dir):
            raise ValueError(f"Images directory not found: {images_dir}")
        
        # Find all image pairs
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.tif') or f.endswith('.png')])
        
        # Process each city
        cities = set([f.split('_')[0] for f in image_files])
        
        for city in cities:
            self._process_city(city, images_dir, labels_dir)
    
    def _process_city(self, city, images_dir, labels_dir):
        """Process a single city's images"""
        # This is a placeholder - adapt based on your actual file structure
        pass
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch_data = self.patches[idx]
        
        patch1 = torch.from_numpy(patch_data['patch1'].copy()).float()
        patch2 = torch.from_numpy(patch_data['patch2'].copy()).float()
        label = torch.tensor(patch_data['label'], dtype=torch.long)
        
        # Data augmentation
        if self.use_augmentation:
            if np.random.rand() > 0.5:
                patch1 = torch.flip(patch1, [1])
                patch2 = torch.flip(patch2, [1])
            if np.random.rand() > 0.5:
                patch1 = torch.flip(patch1, [2])
                patch2 = torch.flip(patch2, [2])
            k = np.random.randint(0, 4)
            if k > 0:
                patch1 = torch.rot90(patch1, k, [1, 2])
                patch2 = torch.rot90(patch2, k, [1, 2])
        
        return patch1, patch2, label


def get_class_weights(dataset):
    """
    Calculate class weights for handling imbalanced dataset
    Returns weights for cross-entropy loss
    """
    labels = [patch['label'] for patch in dataset.patches]
    unique, counts = np.unique(labels, return_counts=True)
    
    # Inverse frequency weighting
    total = len(labels)
    weights = total / (len(unique) * counts)
    
    print(f"Class distribution: {dict(zip(unique, counts))}")
    print(f"Class weights: {dict(zip(unique, weights))}")
    
    return torch.FloatTensor(weights)
