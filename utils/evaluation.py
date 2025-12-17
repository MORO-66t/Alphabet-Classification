"""
Evaluation utilities for model performance assessment.
"""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def evaluate_model(model, X_test, y_test, verbose=1):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: trained Keras/TensorFlow model
        X_test: test features
        y_test: test labels (one-hot encoded)
        verbose: verbosity level (0=silent, 1=progress bar, 2=one line)
    
    Returns:
        tuple: (test_loss, test_accuracy, y_pred, y_pred_proba, y_true)
    """
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=verbose)
    
    # Generate predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    if verbose >= 1:
        print(f"\n{'='*50}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"{'='*50}\n")
    
    return test_loss, test_acc, y_pred, y_pred_proba, y_true


def generate_classification_report(y_true, y_pred, labels, print_report=True):
    """
    Generate detailed classification report.
    
    Args:
        y_true: true labels (integer encoded)
        y_pred: predicted labels (integer encoded)
        labels: list of class names
        print_report: whether to print the report
    
    Returns:
        str: classification report text
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        digits=4
    )
    
    if print_report:
        print("Classification Report:")
        print("="*70)
        print(report)
    
    return report


def calculate_per_class_accuracy(y_true, y_pred, labels):
    """
    Calculate per-class accuracy.
    
    Args:
        y_true: true labels (integer encoded)
        y_pred: predicted labels (integer encoded)
        labels: list of class names
    
    Returns:
        dict: {class_name: accuracy} for each class
    """
    per_class_acc = {}
    
    for i, label in enumerate(labels):
        mask = (y_true == i)
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred[mask] == y_true[mask])
            per_class_acc[label] = class_acc
        else:
            per_class_acc[label] = 0.0
    
    return per_class_acc


def analyze_overfitting(history):
    """
    Analyze overfitting gap from training history.
    
    Args:
        history: Keras history object from model.fit()
    
    Returns:
        dict: overfitting metrics
    """
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    overfitting_gap = final_train_acc - final_val_acc
    
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"\n{'='*50}")
    print(f"Overfitting Analysis:")
    print(f"  Final Training Accuracy: {final_train_acc:.4f}")
    print(f"  Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"  Overfitting Gap: {overfitting_gap:.4f}")
    print(f"  Final Training Loss: {final_train_loss:.4f}")
    print(f"  Final Validation Loss: {final_val_loss:.4f}")
    
    if overfitting_gap < 0.05:
        print(f"  Status: ✅ Excellent generalization (gap < 5%)")
    elif overfitting_gap < 0.10:
        print(f"  Status: ✅ Good generalization (gap < 10%)")
    elif overfitting_gap < 0.15:
        print(f"  Status: ⚠️ Moderate overfitting (gap 10-15%)")
    else:
        print(f"  Status: ❌ High overfitting (gap > 15%)")
    
    print(f"{'='*50}\n")
    
    return {
        'train_acc': final_train_acc,
        'val_acc': final_val_acc,
        'overfitting_gap': overfitting_gap,
        'train_loss': final_train_loss,
        'val_loss': final_val_loss
    }


def find_misclassified_examples(y_true, y_pred, X_test=None, top_n=None):
    """
    Find indices of misclassified examples.
    
    Args:
        y_true: true labels (integer encoded)
        y_pred: predicted labels (integer encoded)
        X_test: optional test data (for returning actual examples)
        top_n: optional, return only top N misclassified examples
    
    Returns:
        numpy array: indices of misclassified examples
    """
    mis_idx = np.where(y_pred != y_true)[0]
    
    print(f"Total misclassified: {len(mis_idx)} out of {len(y_true)} ({100*len(mis_idx)/len(y_true):.2f}%)")
    
    if top_n is not None:
        np.random.shuffle(mis_idx)
        mis_idx = mis_idx[:top_n]
    
    return mis_idx
