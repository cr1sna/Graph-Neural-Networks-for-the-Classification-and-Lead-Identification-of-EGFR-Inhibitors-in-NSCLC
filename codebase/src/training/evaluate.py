import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    confusion_matrix,
    matthews_corrcoef
)

def compute_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Computes classification metrics for binary predictions.
    
    Args:
        y_true: True labels (numpy array or tensor)
        y_pred_probs: Predicted probabilities [0, 1] (numpy array or tensor)
        threshold: Threshold for classifying as positive
        
    Returns:
        dict: Dictionary of metrics
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred_probs, torch.Tensor):
        y_pred_probs = y_pred_probs.cpu().numpy()
        
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    # Handle edge case where there's only one class in y_true (e.g., small batch)
    if len(np.unique(y_true)) == 1:
        # Cannot compute AUROC or AUPRC with 1 class
        metrics = {
            'auroc': float('nan'),
            'auprc': float('nan'),
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'mcc': float('nan')
        }
        return metrics
        
    # Calculate metrics
    auroc = roc_auc_score(y_true, y_pred_probs)
    auprc = average_precision_score(y_true, y_pred_probs)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'specificity': specificity,
        'mcc': mcc,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }

def print_metrics(metrics, prefix=""):
    """Pretty prints the metrics dictionary."""
    p = f"[{prefix}] " if prefix else ""
    print(f"{p}AUROC:     {metrics['auroc']:.4f}")
    print(f"{p}AUPRC:     {metrics['auprc']:.4f}")
    print(f"{p}Accuracy:  {metrics['accuracy']:.4f}")
    print(f"{p}F1 Score:  {metrics['f1']:.4f}")
    print(f"{p}MCC:       {metrics['mcc']:.4f}")
    print(f"{p}Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | Specificity: {metrics['specificity']:.4f}")
