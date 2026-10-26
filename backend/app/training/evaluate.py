"""
Dermora AI - Comprehensive Evaluation Suite
Evaluates trained PyTorch or ONNX models on an independent test partition.
Computes:
- Per-class Precision, Sensitivity (Recall), Specificity, and F1-Score
- Balanced Multi-class Accuracy (BMA) and Overall Accuracy
- Multi-class One-vs-Rest ROC-AUC
- Confusion Matrix (7x7 ISIC lesions)
"""

from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    balanced_accuracy_score,
    accuracy_score,
    precision_recall_fscore_support
)
from backend.app.training.dataset import (
    HAM10000_LABEL_MAP,
    INDEX_TO_LABEL,
    LABEL_DISPLAY_NAMES
)

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    split_name: str = "Independent Test Set"
) -> Dict[str, Any]:
    """
    Computes rigorous clinical metrics from true label indices and predicted probability vectors.
    """
    y_pred = np.argmax(y_pred_probs, axis=1)
    num_classes = len(HAM10000_LABEL_MAP)
    labels = list(range(num_classes))

    # Basic accuracies
    acc = float(accuracy_score(y_true, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Per-class metrics
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    # Per-class specificity: TN / (TN + FP)
    specificity_scores = []
    for i in range(num_classes):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        specificity_scores.append(spec)

    # Multi-class One-vs-Rest ROC-AUC
    try:
        ovr_auc = float(roc_auc_score(y_true, y_pred_probs, multi_class="ovr", labels=labels))
    except Exception:
        ovr_auc = None

    per_class_summary = []
    for i in range(num_classes):
        code = INDEX_TO_LABEL[i].upper()
        display_name = LABEL_DISPLAY_NAMES[INDEX_TO_LABEL[i]]
        per_class_summary.append({
            "code": code,
            "name": display_name,
            "support": int(support[i]),
            "precision": round(float(p[i]), 4),
            "sensitivity_recall": round(float(r[i]), 4),
            "specificity": round(float(specificity_scores[i]), 4),
            "f1_score": round(float(f1[i]), 4)
        })

    return {
        "split_name": split_name,
        "total_evaluated_samples": len(y_true),
        "overall_accuracy": round(acc, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "macro_f1_score": round(float(np.mean(f1)), 4),
        "macro_sensitivity": round(float(np.mean(r)), 4),
        "macro_specificity": round(float(np.mean(specificity_scores)), 4),
        "roc_auc_ovr": round(ovr_auc, 4) if ovr_auc is not None else "N/A",
        "per_class_metrics": per_class_summary,
        "confusion_matrix": cm.tolist(),
        "class_labels": [INDEX_TO_LABEL[i].upper() for i in range(num_classes)],
        "clinical_note": (
            "Independent evaluation on unseen patient lesions. Balanced Multi-class "
            "Accuracy (BMA) is the primary clinical metric to avoid bias from the dominant Nevus class."
        )
    }
