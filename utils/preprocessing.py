"""
Preprocessing utilities for EMNIST dataset loading and preparation.
"""

import numpy as np
import pandas as pd
import kagglehub
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def load_emnist_data():
    """
    Download and load EMNIST Balanced dataset from Kaggle.
    
    Returns:
        tuple: (train_data, test_data, label_map, num_classes)
            - train_data: pandas DataFrame with training data
            - test_data: pandas DataFrame with test data
            - label_map: list of character labels (e.g., ['0', '1', ..., 'A', ...])
            - num_classes: int, number of classes (47 for balanced dataset)
    """
    # Download dataset
    path = kagglehub.dataset_download("crawford/emnist")
    print("Dataset path:", path)
    
    # Load train and test CSVs
    train_data = pd.read_csv(f"{path}/emnist-balanced-train.csv")
    test_data = pd.read_csv(f"{path}/emnist-balanced-test.csv")
    
    # Load label mapping (index -> ASCII code)
    mapping = np.loadtxt(f"{path}/emnist-balanced-mapping.txt", delimiter=" ", dtype=int)
    label_map = [chr(m[1]) for m in mapping]  # e.g., ['0', '1', ..., 'A', ...]
    num_classes = len(label_map)
    
    # Set column names
    column_names = ['label'] + [f'pixel_{i}' for i in range(784)]
    train_data.columns = column_names
    test_data.columns = column_names
    
    print(f"Classes: {num_classes}")
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    return train_data, test_data, label_map, num_classes


def preprocess_emnist(train_data, test_data, num_classes, val_split=0.1, random_state=42):
    """
    Preprocess EMNIST data for model training.
    
    Steps:
        1. Separate features and labels
        2. Reshape to 28x28x1
        3. Rotate 90 degrees (EMNIST format correction)
        4. Normalize to [0, 1]
        5. Shuffle training data
        6. Train/validation split
        7. One-hot encode labels
    
    Args:
        train_data: pandas DataFrame with training data
        test_data: pandas DataFrame with test data
        num_classes: int, number of classes
        val_split: float, validation split ratio (default 0.1)
        random_state: int, random seed for reproducibility (default 42)
    
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
            All arrays are preprocessed and ready for training
    """
    # Separate features and labels
    X_train = train_data.drop('label', axis=1).values
    y_train = train_data['label'].values
    X_test = test_data.drop('label', axis=1).values
    y_test = test_data['label'].values
    
    print("Original shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Reshape to 28x28x1
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    # Rotate 90 degrees (EMNIST images are rotated)
    X_train = np.rot90(X_train, k=1, axes=(1, 2))
    X_test = np.rot90(X_test, k=1, axes=(1, 2))
    
    # Normalize to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Shuffle training data
    X_train, y_train = shuffle(X_train, y_train, random_state=random_state)
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=val_split, 
        random_state=random_state
    )
    
    print("\nAfter train/val split:")
    print(f"  Training set: {X_train.shape}, {y_train.shape}")
    print(f"  Validation set: {X_val.shape}, {y_val.shape}")
    
    # One-hot encode labels
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)
    
    print(f"\nPreprocessing complete! ✅")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def preprocess_for_transfer_learning(X_train, X_val, X_test, target_size=(224, 224)):
    """
    Resize images for transfer learning models (e.g., ResNet, VGG).
    
    Args:
        X_train, X_val, X_test: numpy arrays of shape (n, 28, 28, 1)
        target_size: tuple, target image size (default 224x224 for ImageNet models)
    
    Returns:
        tuple: (X_train_resized, X_val_resized, X_test_resized)
            All arrays resized and converted to RGB format
    """
    import tensorflow as tf
    
    def resize_images(images, size):
        # Resize and convert grayscale to RGB
        resized = tf.image.resize(images, size)
        rgb = tf.image.grayscale_to_rgb(resized)
        return rgb.numpy()
    
    X_train_resized = resize_images(X_train, target_size)
    X_val_resized = resize_images(X_val, target_size)
    X_test_resized = resize_images(X_test, target_size)
    
    print(f"Resized to {target_size} for transfer learning:")
    print(f"  X_train: {X_train_resized.shape}")
    print(f"  X_val: {X_val_resized.shape}")
    print(f"  X_test: {X_test_resized.shape}")
    
    return X_train_resized, X_val_resized, X_test_resized
