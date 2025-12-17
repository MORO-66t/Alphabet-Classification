"""
Visualization utilities for model analysis and predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


def plot_training_history(history, figsize=(12, 4), save_path=None):
    """
    Plot training and validation accuracy/loss curves.
    
    Args:
        history: Keras history object from model.fit()
        figsize: tuple, figure size
        save_path: optional path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels=None, figsize=(12, 10), 
                         normalize=False, cmap='Blues', save_path=None):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: true labels (integer encoded)
        y_pred: predicted labels (integer encoded)
        labels: list of class names (optional)
        figsize: tuple, figure size
        normalize: whether to normalize the confusion matrix
        cmap: colormap for heatmap
        save_path: optional path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=figsize)
    
    annot_kws = {"size": 10} if len(cm) < 20 else {"size": 6}
    fmt = '.2f' if normalize else 'd'
    
    sns.heatmap(cm, cmap=cmap, annot=True, fmt=fmt, 
                annot_kws=annot_kws, cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    if labels is not None and len(labels) < 30:
        plt.yticks(np.arange(len(labels)) + 0.5, labels, rotation=0)
        plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation=90)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_roc_curves(y_test, y_pred_proba, labels, num_classes, 
                   figsize=(18, 8), save_path=None):
    """
    Plot ROC curves for all classes.
    
    Args:
        y_test: true labels (one-hot encoded)
        y_pred_proba: predicted probabilities
        labels: list of class names
        num_classes: number of classes
        figsize: tuple, figure size
        save_path: optional path to save the figure
    """
    # Ensure y_test is one-hot encoded
    if y_test.ndim == 1:
        y_test_onehot = label_binarize(y_test, classes=np.arange(num_classes))
    else:
        y_test_onehot = y_test
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Compute ROC curve and AUC for each class
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_onehot[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot
    plt.figure(figsize=figsize)
    
    for i in range(num_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            label=f'{labels[i]} (AUC={roc_auc[i]:.2f})',
            linewidth=1.5 if num_classes < 20 else 1.0
        )
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves for All Classes', fontsize=16)
    plt.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        fontsize=9 if num_classes > 20 else 10
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_sample_predictions(X_data, y_true, y_pred, labels, n_samples=9, 
                           figsize=(8, 8), random_state=None, save_path=None):
    """
    Plot sample predictions with true and predicted labels.
    
    Args:
        X_data: input images
        y_true: true labels (integer encoded)
        y_pred: predicted labels (integer encoded)
        labels: list of class names
        n_samples: number of samples to show
        figsize: tuple, figure size
        random_state: random seed for reproducibility
        save_path: optional path to save the figure
    """
    import random
    
    if random_state is not None:
        random.seed(random_state)
    
    indices = random.sample(range(len(X_data)), min(n_samples, len(X_data)))
    
    rows = int(np.ceil(np.sqrt(n_samples)))
    cols = int(np.ceil(n_samples / rows))
    
    plt.figure(figsize=figsize)
    for i, idx in enumerate(indices):
        plt.subplot(rows, cols, i + 1)
        
        # Handle different image shapes
        img = X_data[idx]
        if img.shape[-1] == 1:
            img = img.squeeze()
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        
        true_label = labels[y_true[idx]]
        pred_label = labels[y_pred[idx]]
        
        color = 'green' if y_true[idx] == y_pred[idx] else 'red'
        plt.title(f"True: {true_label} | Pred: {pred_label}", 
                 color=color, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_misclassified_examples(X_test, y_true, y_pred, labels, 
                                mis_idx=None, n_samples=12, 
                                figsize=(12, 6), save_path=None):
    """
    Plot misclassified examples.
    
    Args:
        X_test: test images
        y_true: true labels (integer encoded)
        y_pred: predicted labels (integer encoded)
        labels: list of class names
        mis_idx: indices of misclassified examples (if None, will compute)
        n_samples: number of samples to show
        figsize: tuple, figure size
        save_path: optional path to save the figure
    """
    if mis_idx is None:
        mis_idx = np.where(y_pred != y_true)[0]
    
    print(f'Total misclassified: {len(mis_idx)} out of {len(y_true)} ({100*len(mis_idx)/len(y_true):.2f}%)')
    
    if len(mis_idx) == 0:
        print("No misclassified examples found!")
        return
    
    np.random.shuffle(mis_idx)
    n_show = min(n_samples, len(mis_idx))
    
    rows = 3
    cols = int(np.ceil(n_show / rows))
    
    plt.figure(figsize=figsize)
    for i in range(n_show):
        idx = mis_idx[i]
        ax = plt.subplot(rows, cols, i + 1)
        
        # Handle different image shapes
        img = X_test[idx]
        if img.shape[-1] == 1:
            img = img.squeeze()
        
        plt.imshow(img, cmap='gray')
        
        true_char = labels[y_true[idx]]
        pred_char = labels[y_pred[idx]]
        
        plt.title(f'True: {true_char} | Pred: {pred_char}', 
                 fontsize=10, color='red')
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_class_distribution(data, label_column='label', labels=None, 
                           figsize=(14, 5), save_path=None):
    """
    Plot class distribution bar chart.
    
    Args:
        data: pandas DataFrame with label column
        label_column: name of the label column
        labels: list of class names (optional)
        figsize: tuple, figure size
        save_path: optional path to save the figure
    """
    class_counts = data[label_column].value_counts().sort_index()
    
    if labels is not None:
        label_names = [labels[i] for i in class_counts.index]
    else:
        label_names = class_counts.index
    
    plt.figure(figsize=figsize)
    plt.bar(label_names, class_counts.values)
    plt.xlabel("Class (digits + letters)", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.title("Class Distribution in Training Data", fontsize=16)
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def show_sample_images(X, y=None, labels=None, n=8, figsize=(12, 3)):
    """
    Display sample images from dataset.
    
    Args:
        X: input images
        y: labels (optional, can be integer or one-hot encoded)
        labels: list of class names (optional)
        n: number of images to show
        figsize: tuple, figure size
    """
    plt.figure(figsize=figsize)
    limit = min(n, len(X))
    
    for i in range(limit):
        ax = plt.subplot(1, limit, i + 1)
        
        # Handle different image shapes
        img = X[i]
        if img.shape[-1] == 1:
            img = img.squeeze()
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        
        # Handle labels
        if y is not None:
            if hasattr(y, "ndim") and getattr(y, "ndim", 1) > 1:
                lbl = int(np.argmax(y[i]))
            else:
                lbl = int(y[i])
            
            if labels is not None:
                plt.title(labels[lbl])
            else:
                plt.title(f"Class {lbl}")
        
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
