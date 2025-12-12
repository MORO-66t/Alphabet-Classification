# EMNIST Alphabet Classification

> A comprehensive deep learning project for classifying handwritten alphabet characters (A-Z) using multiple state-of-the-art neural network architectures.

## 📋 Project Overview

This project implements and compares multiple deep learning architectures for **26-class alphabet letter classification** using the EMNIST dataset. It serves as a practical exploration of modern computer vision techniques, including custom implementations, transfer learning, and advanced architectures.

**Dataset:** EMNIST Letters (28×28 grayscale images)  
**Classes:** 26 (A-Z uppercase letters)  
**Total Samples:** ~145,000 training + ~20,000 testing images

---

## 🎯 Objectives

- Implement multiple CNN architectures from scratch and using transfer learning
- Compare model performance across different architectures
- Demonstrate data preprocessing, augmentation, and evaluation best practices
- Explore various techniques including batch normalization, dropout, and learning rate scheduling
- Evaluate models using confusion matrices, ROC curves, and classification metrics

---

## 📁 Project Structure

```
Alphabet-Classification-Description/
├── README.md                              # Project documentation
├── Data/
│   ├── emnist-letters-train.csv          # Training dataset (124,800 samples)
│   └── emnist-letters-test.csv           # Testing dataset (20,800 samples)
├── Models/
│   ├── DL_Projectv3.ipynb                # Baseline VGG implementation
│   ├── DL_Projectv5.ipynb                # Enhanced multi-architecture implementation
│   ├── inceptionv1-googlenet.ipynb       # GoogLeNet/Inception V1 with PyTorch
│   └── ResNet.ipynb                      # ResNet-focused implementation
└── Documentation of Architectures/
    └── dcoumnetion_DL.docx               # Architecture documentation
```

---

## 📊 Dataset Information

### EMNIST Letters Dataset
- **Format:** CSV files (pixel values in flattened 784-dimensional vectors)
- **Image Resolution:** 28×28 pixels
- **Channels:** 1 (grayscale)
- **Classes:** 26 (letters A-Z)
- **Training Set:** 124,800 images
- **Test Set:** 20,800 images
- **Source:** Kaggle/ExtendedMNIST

### Data Format
```
Label | Pixel_0 | Pixel_1 | ... | Pixel_783
  1   |   255   |   200   | ... |    10
  2   |   240   |   190   | ... |     0
  .   |   .     |   .     | ... |    .
```

---

## 🏗️ Implemented Architectures

### 1. **VGG-19 (Custom Implementation)**
- **Type:** CNN built from scratch
- **Layers:** 3 convolutional blocks with batch normalization
- **Features:**
  - Conv2D filters: 32 → 64 → 128
  - MaxPooling2D layers for dimensionality reduction
  - Dropout (0.3) for regularization
  - Dense layers: 256 → 26 (softmax output)
- **Training:** 50 epochs with EarlyStopping (patience=6)
- **File:** `DL_Projectv3.ipynb`, `DL_Projectv5.ipynb`

### 2. **ResNet-50 (Transfer Learning)**
- **Type:** Pre-trained ResNet50 + fine-tuning
- **Features:**
  - ImageNet pre-trained weights
  - Frozen base model → Global Average Pooling
  - Custom classification head (256 hidden → 26 output)
  - Input resizing: 28×28 → 224×224
  - Grayscale → RGB channel expansion
- **Training:** 10 epochs, Adam optimizer (lr=1e-3)
- **File:** `DL_Projectv5.ipynb`

### 3. **MobileNetV2 (Transfer Learning)**
- **Type:** Lightweight pre-trained model
- **Features:**
  - ImageNet pre-trained weights (trainable)
  - Input resizing: 28×28 → 32×32
  - Data augmentation (rotation, shift, zoom)
  - Fine-tuning with low learning rate (1e-5)
  - ReduceLROnPlateau scheduling
- **Training:** 20 epochs with augmentation + 10 fine-tuning epochs
- **File:** `DL_Projectv5.ipynb`

### 4. **Vision Transformer (ViT)**
- **Type:** Transformer-based architecture
- **Features:**
  - Patch embedding (4×4 patches → 49 total)
  - Positional embeddings
  - 4 transformer encoder blocks (multi-head attention)
  - Layer normalization + MLP heads
  - GELU activation
- **Architecture Details:**
  - Projection dim: 64
  - Num heads: 4
  - Transformer units: [128, 64]
  - MLP head units: [256, 128]
- **Training:** 20 epochs with data augmentation
- **File:** `DL_Projectv5.ipynb`

### 5. **GoogLeNet/Inception V1 (PyTorch)**
- **Type:** Pre-trained GoogLeNet with auxiliary classifiers
- **Features:**
  - PyTorch implementation with torchvision
  - Auxiliary loss weighting (0.3 × aux1 + 0.3 × aux2 + main)
  - Custom EMNISTDataset class with grayscale→RGB conversion
  - Image resizing to 224×224
  - Data augmentation (rotation)
- **Training:** 10 epochs, Adam optimizer (lr=1e-4)
- **Evaluation:** Comprehensive metrics + ROC curves for all 26 classes
- **File:** `inceptionv1-googlenet.ipynb`

---

## 🔧 Key Technologies & Libraries

### Data Processing & ML
- **TensorFlow/Keras** - Deep learning framework (VGG, ResNet, MobileNet, ViT)
- **PyTorch** - Deep learning framework (GoogLeNet/Inception V1)
- **pandas** - Data loading and manipulation
- **NumPy** - Numerical operations
- **scikit-learn** - Preprocessing, metrics, and utilities

### Visualization
- **Matplotlib** - Training curves, confusion matrices, ROC curves
- **Seaborn** - Enhanced visualizations (heatmaps)

### Data Augmentation
- **tf.keras.preprocessing.image.ImageDataGenerator** - Real-time data augmentation
- **torchvision.transforms** - PyTorch-based augmentations

---

## 📈 Data Preprocessing Pipeline

1. **Data Loading**
   ```
   CSV → Pandas DataFrame
   Shape: (N, 785) - 1 label + 784 pixels
   ```

2. **Feature Extraction**
   ```
   X = drop 'label' column → reshape to (N, 28, 28, 1)
   y = label values → subtract 1 (convert 1-26 to 0-25)
   ```

3. **Image Transformation**
   ```
   Rotation: 90° rotation applied (align handwritten letters)
   Reshape: Flatten → (N, 28, 28, 1) for CNN
   ```

4. **Normalization**
   ```
   X = X.astype('float32') / 255.0  # Pixel values: 0-1 range
   ```

5. **Encoding**
   ```
   y = to_categorical(y, num_classes=26)  # One-hot encoding
   ```

6. **Train-Val Split**
   ```
   Train: 90% (111,120 samples)
   Validation: 10% (12,346 samples)
   Shuffle: random_state=42 (reproducibility)
   ```

7. **Data Augmentation** (optional)
   ```
   Rotation: ±10°
   Width/Height Shift: ±10%
   Zoom: ±10%
   Shear: ±10%
   ```

---

## 🚀 How to Use

### Prerequisites
```bash
pip install tensorflow keras pandas numpy scikit-learn matplotlib seaborn pytorch torchvision kagglehub
```

### Running the Notebooks

**Option 1: VGG Implementation (Baseline)**
```
Open: Models/DL_Projectv3.ipynb
Contains: Data loading, preprocessing, VGG training, and evaluation
```

**Option 2: Multi-Architecture Comparison (Recommended)**
```
Open: Models/DL_Projectv5.ipynb
Contains: VGG + ResNet + MobileNet + ViT implementations
```

**Option 3: GoogLeNet/Inception V1**
```
Open: Models/inceptionv1-googlenet.ipynb
Note: Requires PyTorch (CPU or CUDA available)
```

**Option 4: ResNet Focused**
```
Open: Models/ResNet.ipynb
Contains: Detailed ResNet transfer learning workflow
```

### Execution Steps
1. Install dependencies
2. Ensure data files exist in `Data/` folder
3. Run cells sequentially in Jupyter Notebook or Google Colab
4. Monitor training with TensorBoard or printed metrics
5. Review evaluation visualizations (confusion matrices, ROC curves)

---

## 📊 Model Evaluation Metrics

All models are evaluated using:

- **Accuracy:** Overall correct classification rate
- **Precision & Recall:** Per-class performance metrics
- **F1-Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** 26×26 matrix showing per-class misclassifications
- **ROC Curve & AUC:** Area under receiver operating characteristic curve
- **Classification Report:** Detailed per-class statistics

### Expected Performance
- **VGG:** ~95-97% accuracy
- **ResNet50:** ~96-98% accuracy (transfer learning advantage)
- **MobileNet:** ~94-96% accuracy (lightweight, faster inference)
- **ViT:** ~94-96% accuracy (attention-based mechanisms)
- **GoogLeNet:** ~95-97% accuracy (with auxiliary classifiers)

---

## 💡 Key Features & Techniques

### Regularization
- ✅ Batch Normalization (accelerates training, reduces internal covariate shift)
- ✅ Dropout (0.2-0.5 rates for different architectures)
- ✅ Early Stopping (patience=5-6 to prevent overfitting)
- ✅ ReduceLROnPlateau (adaptive learning rate scheduling)

### Training Strategies
- ✅ Adam Optimizer (adaptive learning rates)
- ✅ Categorical Crossentropy Loss (for multi-class classification)
- ✅ Data Augmentation (rotation, shift, zoom, shear)
- ✅ Transfer Learning (pre-trained ImageNet weights)
- ✅ Fine-tuning (unfreezing base model layers)

### Evaluation Techniques
- ✅ Confusion Matrices (identifies problematic class pairs)
- ✅ ROC Curves per class (24 individual AUC measurements)
- ✅ Misclassification Analysis (visual inspection of errors)
- ✅ Overfitting Gap Analysis (training vs validation accuracy)

---

## 🔍 Known Issues & Improvements

### Current Limitations
- ⚠️ Test dataset contains only 19 classes (A-S), missing T-Z
- ⚠️ Some undefined variables in evaluation sections (need careful execution order)
- ⚠️ Code duplication in v5 (ResNet and ViT sections repeated)

### Recommended Improvements
1. **Complete test dataset** with all 26 classes
2. **Fix undefined variables** in classification reports and ROC curve sections
3. **Modularize code** - Create reusable functions for common operations
4. **Add hyperparameter tuning** - Grid search or Bayesian optimization
5. **Ensemble methods** - Combine multiple models for better accuracy
6. **Class imbalance handling** - Use weighted loss if classes are imbalanced
7. **Model deployment** - Export models to ONNX or SavedModel format

---

## 📝 Version History

| Version | Date | Changes |
|---------|------|---------|
| v3 | 2024 | Initial VGG implementation, basic evaluation |
| v5 | 2024 | Added ResNet, MobileNet, ViT; expanded evaluation; multiple evaluation cells |

### Key Differences: v3 vs v5
- v5 supports Google Colab (`/content/` paths) + Kaggle paths
- v5 includes 4 additional architectures beyond VGG
- v5 has comprehensive evaluation with ROC curves for all 26 classes
- v5 includes GoogLeNet/Inception V1 (PyTorch implementation)

---

## 📚 References & Resources

### Academic Papers
- [VGG: Very Deep Convolutional Networks](https://arxiv.org/abs/1409.1556)
- [ResNet: Deep Residual Learning](https://arxiv.org/abs/1512.03385)
- [MobileNetV2: Inverted Residuals](https://arxiv.org/abs/1801.04381)
- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
- [GoogLeNet/Inception](https://arxiv.org/abs/1409.4842)

### Datasets
- [EMNIST on Kaggle](https://www.kaggle.com/datasets/crawford/emnist)
- [Original MNIST Paper](http://yann.lecun.com/exdb/mnist/)

### Tools & Frameworks
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/)
- [PyTorch Documentation](https://pytorch.org/)
- [scikit-learn Guide](https://scikit-learn.org/)

---

## 👤 Project Information

**Level:** Level 4 - Advanced Deep Learning  
**Institution:** FHCAI  
**Type:** Comparative Deep Learning Study  
**Last Updated:** December 2024

---

## 📞 Notes for Users

1. **Data Location:** Ensure `Data/emnist-letters-train.csv` and `emnist-letters-test.csv` are in the correct folder
2. **GPU/TPU:** Optional but recommended for faster training. Use Google Colab for free GPU access
3. **Kaggle Setup:** If using Kaggle kernel, API credentials are automatically handled
4. **Execution Time:** Full pipeline (all architectures) takes ~2-4 hours on GPU
5. **Memory:** Requires ~4-8GB RAM; 12GB+ recommended for smooth execution

---

## 📄 License & Attribution

This project uses the EMNIST dataset from the Extended MNIST project. All implementations are educational and based on published architectures.

---

**Happy Learning! 🚀**
