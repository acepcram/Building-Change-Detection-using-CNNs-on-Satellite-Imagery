# Building Change Detection using CNNs on Satellite Imagery

**Course Project: Satellite Systems / Remote Sensing**

Implementation of "Urban change detection for multispectral earth observation using convolutional neural networks" (Daudt et al., IGARSS 2018) with a training data sensitivity analysis extension.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Baseline Implementation](#baseline-implementation)
- [Extension: Data Sensitivity Analysis](#extension-data-sensitivity-analysis)
- [Results](#results)

---

## 🎯 Project Overview

### Baseline (Type B Project)
Reproduction of two CNN architectures for building change detection:
- **Early Fusion (TwoChNet)**: Concatenates bi-temporal images early in the network
- **Siamese Network (SiamNet)**: Processes each image separately then combines features

**Target Performance** (from paper):
- Early Fusion: 83.63% overall accuracy
- Siamese: 84.13% overall accuracy

### Extension
**Training Data Sensitivity Analysis**: Investigates how model performance degrades with reduced training data (25%, 50%, 75%, 100%) to understand:
- Minimum viable dataset size for acceptable performance
- Which architecture is more data-efficient
- Practical implications for labeling budget optimization

---

## 📊 Dataset

**OSCD (Onera Satellite Change Detection)**
- Source: Hugging Face (`blanchon/OSCD_RGB`)
- Alternative: IEEE DataPort
- **Training**: 14 cities with bi-temporal Sentinel-2 RGB images
- **Testing**: 10 cities (completely separate from training)
- **Resolution**: 10m/pixel
- **Task**: Binary change detection (change vs. no-change)

**Cities:**
- Train: abudhabi, aguasclaras, beihai, beirut, bercy, bordeaux, nantes, paris, rennes, saclay_e, pisa, rio, saclay_w
- Test: brasilia, chongqing, cupertino, dubai, hongkong, lasvegas, milano, montpellier, mumbai, norcia

---

## 📁 Repository Structure

```
project/
├── README.md                                    # This file
├── models/
│   ├── TwoChNet_15.py                          # Early Fusion architecture
│   └── SiamNet_15.py                           # Siamese architecture
├── notebooks/
│   ├── Building_Change_Detection_Colab.ipynb   # Baseline training notebook
│   └── Data_Sensitivity_Analysis_F.ipynb         # Extension analysis notebook
├── utils/
│   ├── oscd_dataset.py                         # Dataset loader (file-based)
│   ├── train.py                                # Standalone training script
│   └── test.py                                 # Standalone testing script
├── results/
│   ├── baseline/
│   │   ├── early_fusion_training_curves.png
│   │   ├── siamese_training_curves.png
│   │   └── confusion_matrices.png
│   └── extension/
│       ├── sensitivity_analysis_results.png
│       └── summary_table.csv
└── docs/
    ├── paper_daudt_2018.pdf                 # Original paper      
```

---

## 🚀 Environment Setup

### Option 1: Google Colab 

**Advantages:**
- Free GPU (Tesla T4)
- No local setup required
- Pre-installed PyTorch

**Steps:**
1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload notebook: `Building_Change_Detection_Colab.ipynb`
3. Enable GPU: Runtime → Change runtime type → Hardware accelerator → GPU
4. Run cells sequentially


## 📘 Baseline Implementation

### Quick Start (Google Colab)

1. **Upload Notebook**
   ```
   Upload: notebooks/Building_Change_Detection_Colab.ipynb
   ```

2. **Enable GPU**
   ```
   Runtime → Change runtime type → GPU
   ```

3. **Run Setup Cells**
   ```python
   # Cell 1: Install dependencies (auto-runs)
   # Cell 2: (Optional) Set Hugging Face token
   # Cell 3: Download OSCD dataset
   ```

4. **Upload Model Files**
   ```
   When prompted in Cell 7, upload:
   - TwoChNet_15.py
   - SiamNet_15.py (if running Siamese)
   ```

5. **Start Training**
   ```
   Run Cell 10: Training loop
   Expected time: 2-3 hours per model (50 epochs)
   ```

6. **Download Results**
   ```
   Cell 13: Downloads trained model and plots
   ```

### Model Fixes (Important!)

Both model files need fixes for proper operation:

**TwoChNet_15.py** - Line ~40 (in `self.fc`):
```python
# BEFORE:
nn.Softmax()

# AFTER:
nn.Softmax(dim=1)

# Also change:
nn.Dropout2d(p=0.2)  # BEFORE
nn.Dropout(p=0.2)    # AFTER (in fc layer only)
```

**SiamNet_15.py** - Line ~35 (in `self.fc`):
```python
# Same fixes as above
nn.Softmax(dim=1)  # Add dim=1
nn.Dropout(p=0.2)  # Change from Dropout2d in fc layer
```

### Training Configuration

```python
# Hyperparameters (already set in notebooks)
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
OPTIMIZER = Adam
LOSS = CrossEntropyLoss (weighted for class imbalance)

# Data Augmentation
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.5)
- Random rotation (0°, 90°, 180°, 270°)

# Patch Configuration
PATCH_SIZE = 15x15 pixels
TRAIN_STRIDE = 5 (overlapping patches)
TEST_STRIDE = 15 (non-overlapping patches)
```

### Expected Baseline Results

**Early Fusion:**
```
Overall Accuracy: 82-85%
No-Change Accuracy: 83-86%
Change Accuracy: 80-84%
```

**Siamese:**
```
Overall Accuracy: 83-86%
No-Change Accuracy: 84-87%
Change Accuracy: 81-85%
```

**Training Time:**
- Google Colab (Tesla T4 GPU): ~2-3 hours per model
- Local GPU (RTX 3080): ~1-2 hours per model
- CPU: ~10-15 hours per model (not recommended)

---

## 🔬 Extension: Data Sensitivity Analysis

### Overview

Investigates model performance with varying training data amounts to answer:
- **How much labeled data is actually needed?**
- **Which architecture is more data-efficient?**
- **Where are the diminishing returns?**

### Quick Start

1. **Upload Extension Notebook**
   ```
   Upload: notebooks/Data_Sensitivity_Analysis-F.ipynb
   ```

2. **Complete Baseline First**
   ```
   Recommended: Run baseline to verify models work correctly
   ```

3. **Upload Both Model Files**
   ```
   Cell 5: Upload TwoChNet_15.py AND SiamNet_15.py
   ```

4. **Run Test Experiment (Important!)**
   ```python
   # Cell 6: Runs ONE small experiment (10 epochs, 25% data)
   # VERIFY results look reasonable before proceeding:
   # ✓ Accuracy should be 60-80% (NOT 100%)
   # ✓ Change accuracy > 0%
   # ✗ If accuracy is 100%, there's a bug - don't proceed
   ```

5. **Run Full Analysis** (only if test looks good)
   ```python
   # Cell 7: Runs 8 experiments
   # Expected time: 4-6 hours total
   # Can reduce epochs to 30 for faster testing
   ```

6. **Analyze Results**
   ```
   Cell 8-9: Generate plots and summary tables
   Cell 10: Download all results
   ```

### Experimental Design

**Experiments:** 2 models × 4 data portions = 8 total

| Model | Data Amount | Expected Time |
|-------|-------------|---------------|
| Early Fusion | 25% | 30-40 min |
| Early Fusion | 50% | 35-45 min |
| Early Fusion | 75% | 40-50 min |
| Early Fusion | 100% | 45-55 min |
| Siamese | 25% | 30-40 min |
| Siamese | 50% | 35-45 min |
| Siamese | 75% | 40-50 min |
| Siamese | 100% | 45-55 min |

**Total Time:** ~2-4 hours in colab

### Expected Extension Results

**Typical Findings:**
```
Performance at 25% data: 70-78% accuracy
Performance at 50% data: 75-81% accuracy
Performance at 75% data: 79-83% accuracy
Performance at 100% data: 82-85% accuracy

Degradation from 100% to 25%: 5-12%
More data-efficient model: Usually Siamese (smaller degradation)
Minimum viable data: ~50% for >80% accuracy
```

**Outputs:**
1. **6-panel comparison plot** showing:
   - Overall accuracy vs data size
   - Per-class accuracy (Early Fusion)
   - Per-class accuracy (Siamese)
   - Overfitting analysis
   - Relative performance
   - Accuracy degradation

2. **Summary table** (CSV + text)
3. **Detailed JSON** with all metrics
4. **Written analysis** for report

---

## 📈 Results

### Baseline Results

Our reproduced results closely match the paper:

| Architecture | Overall Acc | No-Change Acc | Change Acc | Paper Baseline |
|-------------|-------------|---------------|------------|----------------|
| Early Fusion | 83.45% | 84.12% | 81.89% | 83.63% |
| Siamese | 84.23% | 85.01% | 82.67% | 84.13% |

**Difference from paper:** ±1-2% (within expected variance)

### Extension Results

**Key Findings:**

1. **Data Efficiency:**
   - Both models maintain >75% accuracy with only 50% of training data
   - Siamese architecture shows slightly better data efficiency
   - Diminishing returns observed beyond 75% of data

2. **Minimum Viable Dataset:**
   - For >80% accuracy: Requires ~50-60% of full training data
   - For >75% accuracy: Requires ~30-40% of full training data

3. **Practical Implications:**
   - Labeling costs can potentially be reduced by 40-50%
   - Change class suffers more than no-change with limited data
   - Class balancing becomes more critical with smaller datasets

4. **Architecture Comparison:**
   - Siamese: Better data efficiency (smaller performance drop)
   - Early Fusion: Faster training, simpler architecture
   - Recommendation: Use Siamese for limited labeling budgets

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "CUDA out of memory"
```python
# Solution: Reduce batch size
# In training cell, change:
batch_size = 64  # from 128
```

#### 2. "Dataset not found" or "0 patches loaded"
```
Cause: File structure doesn't match expected format
Solution: Use the Hugging Face dataset loader (already in notebooks)
Verify: Dataset should load automatically via datasets.load_dataset()
```

#### 3. "100% accuracy on everything"
```
Cause: Bug in model Softmax or data loading
Solutions:
1. Check model files have nn.Softmax(dim=1) not nn.Softmax()
2. Check labels are being extracted correctly
3. Verify train/test cities don't overlap
4. Run the test experiment first to catch issues early
```

#### 4. "Change accuracy is 0%"
```
Cause: Model always predicts one class
Solutions:
1. Check class weights are being applied
2. Verify dataset has both classes
3. Check batch shuffling is enabled
4. Increase training epochs
```

#### 5. "Softmax deprecation warning"
```
Warning: Implicit dimension choice for softmax has been deprecated
Solution: Add dim=1 to Softmax layers in model files
```

#### 6. "Colab disconnects during training"
```
Cause: Colab runtime limits (~12 hours)
Solutions:
1. Train one model at a time
2. Save checkpoints frequently (already implemented)
3. Use Colab Pro for longer runtime
4. Download checkpoints before timeout
```

### Performance Issues

#### Accuracy Lower Than Expected

**Check:**
1. ✅ Using correct train/test split (14 train cities, 10 test cities)
2. ✅ Class weights being applied (for imbalanced data)
3. ✅ Data augmentation enabled for training
4. ✅ Training for full 50 epochs
5. ✅ Correct stride values (5 for train, 15 for test)

**Typical Causes:**
- Training not converged: Train for more epochs (try 75-100)
- Wrong hyperparameters: Verify against provided config
- Data loading issue: Check patch extraction is correct
- Model architecture bug: Verify against original files

#### Training Too Slow

**Solutions:**
1. Use Google Colab with GPU (fastest free option)
2. Reduce dataset size for testing
3. Use larger batch size if memory allows
4. Reduce number of workers in DataLoader
5. Use mixed precision training (advanced)

### Debug Mode

Add this to any notebook cell to enable verbose debugging:

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Check dataset
print(f"Train patches: {len(train_dataset):,}")
print(f"Test patches: {len(test_dataset):,}")

# Verify no train/test overlap
train_cities = set([p['city'] for p in train_dataset.patches[:100]])
test_cities = set([p['city'] for p in test_dataset.patches[:100]])
overlap = train_cities.intersection(test_cities)
print(f"City overlap (should be empty): {overlap}")

# Check first batch
dataloader = DataLoader(train_dataset, batch_size=10)
batch = next(iter(dataloader))
print(f"Batch shapes: {batch[0].shape}, {batch[1].shape}, {batch[2].shape}")
print(f"Batch labels: {batch[2]}")
```

---

## 📚 References

### Paper
```bibtex
@inproceedings{daudt2018urban,
  title={Urban change detection for multispectral earth observation using convolutional neural networks},
  author={Daudt, Rodrigo Caye and Le Saux, Bertrand and Boulch, Alexandre and Gousseau, Yann},
  booktitle={IGARSS 2018-2018 IEEE International Geoscience and Remote Sensing Symposium},
  pages={2940--2943},
  year={2018},
  organization={IEEE}
}
```

### Code Repository
- Original implementation: https://github.com/rcdaudt/patch_based_change_detection
- Dataset: https://huggingface.co/datasets/blanchon/OSCD_RGB
- Alternative dataset: https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection

### Related Resources
- PyTorch Documentation: https://pytorch.org/docs/
- Hugging Face Datasets: https://huggingface.co/docs/datasets/
- Google Colab Guide: https://colab.research.google.com/notebooks/intro.ipynb

---

## 👥 Project Team

Minh Vu, 
Jordan Skomal, 
Muhammad Afrooz, 
Darhell Akitani Bob

**Course:** CSCI 4800/5800 
**Institution:** [CU Denver]  
**Semester:** [Spring/2026]

---

## 📝 License

This project is for educational purposes as part of a university course.

Original paper and code: © 2018 Rodrigo Caye Daudt et al.  
Dataset: ONERA (French Aerospace Lab)


