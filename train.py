"""
Training Script for Building Change Detection using Early Fusion Architecture
Based on: Daudt et al., "Urban change detection for multispectral earth observation 
          using convolutional neural networks" IGARSS 2018
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Import the model and dataset
from TwoChNet_15 import TwoChNet_15
from oscd_dataset import OSCDDataset, get_class_weights


class Trainer:
    """Trainer class for change detection model"""
    
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 device, save_dir='checkpoints'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for batch_idx, (patch1, patch2, labels) in enumerate(pbar):
            # Move to device
            patch1 = patch1.to(self.device)
            patch2 = patch2.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(patch1, patch2)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Per-class accuracy
        class_correct = [0, 0]
        class_total = [0, 0]
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            for batch_idx, (patch1, patch2, labels) in enumerate(pbar):
                # Move to device
                patch1 = patch1.to(self.device)
                patch2 = patch2.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(patch1, patch2)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Per-class accuracy
                for i in range(2):
                    mask = (labels == i)
                    class_correct[i] += (predicted[mask] == i).sum().item()
                    class_total[i] += mask.sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        # Calculate per-class accuracies
        no_change_acc = 100. * class_correct[0] / max(class_total[0], 1)
        change_acc = 100. * class_correct[1] / max(class_total[1], 1)
        
        print(f'Validation - Overall Acc: {epoch_acc:.2f}%, '
              f'No-Change Acc: {no_change_acc:.2f}%, '
              f'Change Acc: {change_acc:.2f}%')
        
        return epoch_loss, epoch_acc, no_change_acc, change_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
        }
        
        # Save latest checkpoint
        path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, path)
        
        # Save best checkpoint
        if is_best:
            path = os.path.join(self.save_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, path)
            print(f'Saved best model with validation accuracy: {self.best_val_acc:.2f}%')
    
    def plot_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accs, label='Train Acc')
        ax2.plot(self.val_accs, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'))
        plt.close()
    
    def train(self, num_epochs):
        """Main training loop"""
        print(f'Starting training for {num_epochs} epochs...')
        print(f'Device: {self.device}')
        print(f'Model: Early Fusion (TwoChNet_15)')
        print(f'Training samples: {len(self.train_loader.dataset)}')
        print(f'Validation samples: {len(self.val_loader.dataset)}')
        print('-' * 60)
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc, no_change_acc, change_acc = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            self.save_checkpoint(epoch, is_best)
            
            # Plot history
            self.plot_history()
            
            print(f'Epoch {epoch}/{num_epochs} - '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print('-' * 60)
        
        print(f'Training completed! Best validation accuracy: {self.best_val_acc:.2f}%')


def main():
    parser = argparse.ArgumentParser(description='Train Early Fusion model for change detection')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to OSCD dataset directory')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay (default: 0.0001)')
    parser.add_argument('--patch_size', type=int, default=15,
                        help='Patch size (default: 15)')
    parser.add_argument('--stride_train', type=int, default=5,
                        help='Stride for training patch extraction (default: 5)')
    parser.add_argument('--stride_val', type=int, default=15,
                        help='Stride for validation patch extraction (default: 15)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create datasets
    print('Loading datasets...')
    train_dataset = OSCDDataset(
        root_dir=args.data_dir,
        split='train',
        patch_size=args.patch_size,
        stride=args.stride_train,
        use_augmentation=True,
        rgb_only=True
    )
    
    val_dataset = OSCDDataset(
        root_dir=args.data_dir,
        split='test',  # Using test set for validation during training
        patch_size=args.patch_size,
        stride=args.stride_val,
        use_augmentation=False,
        rgb_only=True
    )
    
    # Get class weights for handling imbalance
    class_weights = get_class_weights(train_dataset).to(device)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model (Early Fusion with 6 input channels: 3 from each image)
    print('Creating model...')
    model = TwoChNet_15(n_in=6)  # 3 channels per image, concatenated = 6
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Create trainer and start training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir=args.save_dir
    )
    
    trainer.train(num_epochs=args.epochs)


if __name__ == '__main__':
    main()
