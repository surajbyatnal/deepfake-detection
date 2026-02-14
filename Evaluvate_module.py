# Evaluvate/Evaluvate_module.py
"""
Evaluation metrics for binary classification (fake/real detection).
Computes AUC, accuracy, precision, recall, and F1 score.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support


def evaluate_logits(logits, labels):
    """
    Evaluate binary classification performance from raw logits.
    
    Args:
        logits: 1D numpy array of raw logits (shape N,)
        labels: 1D numpy array of binary labels (0/1, shape N,)
        
    Returns:
        dict: Dictionary containing evaluation metrics:
            - auc: Area Under ROC Curve
            - acc: Accuracy
            - precision: Precision score
            - recall: Recall score
            - f1: F1 score
    """
    # Convert logits to probabilities using sigmoid
    probs = 1.0 / (1.0 + np.exp(-logits))
    
    # Calculate AUC with error handling
    try:
        auc = roc_auc_score(labels, probs)
    except (ValueError, Exception):
        # Return NaN if AUC cannot be computed (e.g., only one class present)
        auc = float('nan')
    
    # Get predictions using 0.5 threshold
    preds = (probs >= 0.5).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    
    return {
        'auc': float(auc),
        'acc': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }