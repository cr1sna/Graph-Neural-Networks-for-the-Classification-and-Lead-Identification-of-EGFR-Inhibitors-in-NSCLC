"""metrics.py — Standardised classification metrics for all 6 models."""
from __future__ import annotations
import logging
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, matthews_corrcoef, accuracy_score,
    precision_score, recall_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute classification metrics for binary predictions.

    Args:
        y_true: Ground-truth binary labels, shape (N,).
        y_prob: Predicted probabilities for the positive class, shape (N,).
        threshold: Decision threshold for label binarisation (default 0.5).

    Returns:
        Dict with keys: AUROC, AUPRC, F1, MCC, Accuracy, Precision, Recall.

    Raises:
        ValueError: If y_prob contains NaN values.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if np.any(np.isnan(y_prob)):
        raise ValueError(
            "y_prob contains NaN values. Check model output for numerical issues."
        )

    # Edge case: all labels the same
    if len(np.unique(y_true)) < 2:
        logger.warning(
            "Only one class present in y_true. AUROC set to 0.5 (undefined)."
        )
        auroc = 0.5
    else:
        auroc = float(roc_auc_score(y_true, y_prob))

    auprc = float(average_precision_score(y_true, y_prob))
    y_pred = (y_prob >= threshold).astype(int)
    f1  = float(f1_score(y_true, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))

    return {
        "AUROC": auroc, "AUPRC": auprc, "F1": f1,
        "MCC": mcc, "Accuracy": acc, "Precision": prec, "Recall": rec,
    }


def print_metrics_table(results_dict: dict[str, dict]) -> None:
    """Pretty-print a comparison table of metrics across all models.

    Args:
        results_dict: Mapping of model_name -> metrics dict (from compute_metrics).

    Example::

        print_metrics_table({
            "RandomForest": {"AUROC": 0.89, ...},
            "AttentiveFP":  {"AUROC": 0.84, ...},
        })
    """
    cols = ["AUROC", "AUPRC", "F1", "MCC", "Accuracy"]
    header = f"{'Model':<22}" + "".join(f"{c:>10}" for c in cols)
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for model_name, m in results_dict.items():
        row = f"{model_name:<22}" + "".join(f"{m.get(c, float('nan')):>10.4f}" for c in cols)
        print(row)
    print(sep)
