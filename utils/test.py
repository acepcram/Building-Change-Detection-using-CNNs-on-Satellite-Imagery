"""
Testing and Evaluation Script for Building Change Detection
Generates change detection maps and computes performance metrics
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import rasterio
from scipy.ndimage import gaussian_filter

# Import the model
from TwoChNet_15 import TwoChNet_15
from oscd_dataset import OSCDDataset


class ChangeDetectionEvaluator:
    """Evaluator for change detection model"""
    
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.model.eval()
        
    def evaluate(self):
        """Evaluate model on test set"""
        all_preds = []
        all_labels = []
        all_probs = []
        
        print('Evaluating model...')
        with torch.no_grad():
            for patch1, patch2, labels in tqdm(self.test_loader):
                patch1 = patch1.to(self.device)
                patch2 = patch2.to(self.device)
                
                # Forward pass
                outputs = self.model(patch1, patch2)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(outputs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        return all_preds, all_labels, all_probs
    
    def compute_metrics(self, predictions, labels):
        """Compute evaluation metrics"""
        # Overall accuracy
        overall_acc = 100. * np.mean(predictions == labels)
        
        # Per-class accuracy
        no_change_mask = labels == 0
        change_mask = labels == 1
        
        no_change_acc = 100. * np.mean(predictions[no_change_mask] == 0) if no_change_mask.sum() > 0 else 0
        change_acc = 100. * np.mean(predictions[change_mask] == 1) if change_mask.sum() > 0 else 0
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Classification report
        report = classification_report(labels, predictions, 
                                       target_names=['No Change', 'Change'],
                                       digits=3)
        
        print('\n' + '='*60)
        print('EVALUATION RESULTS')
        print('='*60)
        print(f'Overall Accuracy: {overall_acc:.2f}%')
        print(f'No Change Accuracy: {no_change_acc:.2f}%')
        print(f'Change Accuracy: {change_acc:.2f}%')
        print('\nConfusion Matrix:')
        print(cm)
        print('\nClassification Report:')
        print(report)
        print('='*60)
        
        return {
            'overall_acc': overall_acc,
            'no_change_acc': no_change_acc,
            'change_acc': change_acc,
            'confusion_matrix': cm
        }
    
    def plot_confusion_matrix(self, cm, save_path='confusion_matrix.png'):
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=['No Change', 'Change'],
               yticklabels=['No Change', 'Change'],
               title='Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=20)
        
        fig.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Confusion matrix saved to {save_path}')


class FullImagePredictor:
    """Generate change detection maps for full images"""
    
    def __init__(self, model, device, patch_size=15, stride=5):
        self.model = model
        self.device = device
        self.patch_size = patch_size
        self.stride = stride
        self.model.eval()
    
    def predict_image(self, img1, img2, use_gaussian_voting=True):
        """
        Predict change map for full image pair
        
        Args:
            img1: First image, shape (C, H, W)
            img2: Second image, shape (C, H, W)
            use_gaussian_voting: Use Gaussian weighted voting (default: True)
        
        Returns:
            change_map: Binary change map, shape (H, W)
            probability_map: Change probability map, shape (H, W)
        """
        C, H, W = img1.shape
        
        # Initialize output maps
        if use_gaussian_voting:
            # Accumulator for weighted voting
            votes_sum = np.zeros((2, H, W), dtype=np.float32)
            weights_sum = np.zeros((H, W), dtype=np.float32)
            
            # Create Gaussian weight kernel
            gaussian_kernel = self._create_gaussian_kernel(self.patch_size)
        else:
            # Simple voting without weighting
            votes = np.zeros((2, H, W), dtype=np.int32)
        
        # Extract and classify patches
        with torch.no_grad():
            for y in range(0, H - self.patch_size + 1, self.stride):
                for x in range(0, W - self.patch_size + 1, self.stride):
                    # Extract patch
                    patch1 = img1[:, y:y+self.patch_size, x:x+self.patch_size]
                    patch2 = img2[:, y:y+self.patch_size, x:x+self.patch_size]
                    
                    # Convert to tensor and add batch dimension
                    patch1 = torch.from_numpy(patch1).unsqueeze(0).float().to(self.device)
                    patch2 = torch.from_numpy(patch2).unsqueeze(0).float().to(self.device)
                    
                    # Predict
                    output = self.model(patch1, patch2)
                    probs = output.cpu().numpy()[0]  # Shape: (2,)
                    
                    # Vote
                    if use_gaussian_voting:
                        # Add weighted votes
                        for c in range(2):
                            votes_sum[c, y:y+self.patch_size, x:x+self.patch_size] += \
                                probs[c] * gaussian_kernel
                        weights_sum[y:y+self.patch_size, x:x+self.patch_size] += gaussian_kernel
                    else:
                        # Simple voting
                        pred_class = np.argmax(probs)
                        votes[pred_class, y:y+self.patch_size, x:x+self.patch_size] += 1
        
        # Generate final predictions
        if use_gaussian_voting:
            # Normalize by total weights
            probability_map = np.zeros((H, W), dtype=np.float32)
            for c in range(2):
                mask = weights_sum > 0
                probability_map[mask] = votes_sum[1, mask] / weights_sum[mask]
            
            change_map = (probability_map > 0.5).astype(np.uint8)
        else:
            # Take class with most votes
            change_map = np.argmax(votes, axis=0).astype(np.uint8)
            probability_map = votes[1] / (votes[0] + votes[1] + 1e-10)
        
        return change_map, probability_map
    
    def _create_gaussian_kernel(self, size, sigma=None):
        """Create 2D Gaussian kernel"""
        if sigma is None:
            sigma = size / 6.0
        
        kernel = np.zeros((size, size))
        center = size // 2
        
        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        # Normalize
        kernel = kernel / kernel.sum()
        
        return kernel
    
    def visualize_results(self, img1_rgb, img2_rgb, change_map, ground_truth=None, 
                         save_path='change_detection_result.png'):
        """Visualize change detection results"""
        if ground_truth is not None:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes = [axes]
        
        # First row: Images and prediction
        axes[0][0].imshow(np.transpose(img1_rgb, (1, 2, 0)))
        axes[0][0].set_title('Image T1')
        axes[0][0].axis('off')
        
        axes[0][1].imshow(np.transpose(img2_rgb, (1, 2, 0)))
        axes[0][1].set_title('Image T2')
        axes[0][1].axis('off')
        
        axes[0][2].imshow(change_map, cmap='gray')
        axes[0][2].set_title('Predicted Change Map')
        axes[0][2].axis('off')
        
        # Second row: Ground truth comparison
        if ground_truth is not None:
            axes[1][0].imshow(ground_truth, cmap='gray')
            axes[1][0].set_title('Ground Truth')
            axes[1][0].axis('off')
            
            # Difference map
            diff = np.zeros((*change_map.shape, 3))
            diff[np.logical_and(change_map == 1, ground_truth == 1)] = [0, 1, 0]  # True positive (green)
            diff[np.logical_and(change_map == 0, ground_truth == 0)] = [0, 0, 0]  # True negative (black)
            diff[np.logical_and(change_map == 1, ground_truth == 0)] = [1, 0, 0]  # False positive (red)
            diff[np.logical_and(change_map == 0, ground_truth == 1)] = [1, 1, 0]  # False negative (yellow)
            
            axes[1][1].imshow(diff)
            axes[1][1].set_title('Difference Map\n(Green=TP, Red=FP, Yellow=FN)')
            axes[1][1].axis('off')
            
            # Accuracy
            acc = 100. * np.mean(change_map == ground_truth)
            axes[1][2].text(0.5, 0.5, f'Accuracy: {acc:.2f}%', 
                          ha='center', va='center', fontsize=20)
            axes[1][2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Visualization saved to {save_path}')


def main():
    parser = argparse.ArgumentParser(description='Test change detection model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to OSCD dataset directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for testing (default: 256)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results (default: results)')
    parser.add_argument('--visualize_cities', nargs='+', default=['rio', 'montpellier'],
                        help='Cities to visualize (default: rio montpellier)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print('Loading model...')
    model = TwoChNet_15(n_in=6)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print('Model loaded successfully!')
    
    # Create test dataset
    print('Loading test dataset...')
    test_dataset = OSCDDataset(
        root_dir=args.data_dir,
        split='test',
        patch_size=15,
        stride=15,
        use_augmentation=False,
        rgb_only=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Evaluate on patches
    evaluator = ChangeDetectionEvaluator(model, test_loader, device)
    predictions, labels, probs = evaluator.evaluate()
    metrics = evaluator.compute_metrics(predictions, labels)
    
    # Save confusion matrix
    evaluator.plot_confusion_matrix(
        metrics['confusion_matrix'],
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    # Generate full image predictions for selected cities
    print('\nGenerating full image change maps...')
    predictor = FullImagePredictor(model, device, patch_size=15, stride=5)
    
    # This would require loading full images - placeholder for now
    print('Full image prediction functionality ready.')
    print('To use: load city images and call predictor.predict_image(img1, img2)')
    
    # Save metrics to file
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write('EVALUATION RESULTS\n')
        f.write('='*60 + '\n')
        f.write(f'Overall Accuracy: {metrics["overall_acc"]:.2f}%\n')
        f.write(f'No Change Accuracy: {metrics["no_change_acc"]:.2f}%\n')
        f.write(f'Change Accuracy: {metrics["change_acc"]:.2f}%\n')
        f.write('\nTarget (from paper):\n')
        f.write('Overall Accuracy: 83.63%\n')
        f.write('No Change Accuracy: 83.71%\n')
        f.write('Change Accuracy: 82.14%\n')
    
    print(f'\nResults saved to {args.output_dir}')


if __name__ == '__main__':
    main()
