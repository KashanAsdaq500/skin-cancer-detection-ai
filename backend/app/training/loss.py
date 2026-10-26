"""
Dermora AI - Loss Functions & Class Imbalance Handling
Addresses severe class distribution skew in HAM10000 (NV = ~67%, DF = ~1.1%).
"""

from typing import Optional, List
import numpy as np
import pandas as pd
from backend.app.training.dataset import HAM10000_LABEL_MAP

def compute_balanced_class_weights(
    dataframe: pd.DataFrame,
    label_col: str = "dx",
    smoothing: float = 0.15
) -> np.ndarray:
    """
    Computes normalized inverse-frequency class weights for Cross-Entropy loss.
    Smoothing prevents minority classes (DF, VASC) from dominating gradient updates.
    """
    counts = dataframe[label_col].str.lower().value_counts()
    num_classes = len(HAM10000_LABEL_MAP)
    raw_weights = np.zeros(num_classes, dtype=np.float32)

    total_samples = len(dataframe)
    for label_code, idx in HAM10000_LABEL_MAP.items():
        count = counts.get(label_code, 1)
        # Smoothed inverse frequency
        raw_weights[idx] = total_samples / (count ** (1.0 - smoothing))

    # Normalize weights so mean is 1.0
    normalized_weights = raw_weights / np.mean(raw_weights)
    return normalized_weights

def build_loss_criterion(
    loss_type: str = "weighted_ce",
    class_weights: Optional[np.ndarray] = None,
    gamma: float = 2.0
):
    """
    Constructs PyTorch loss function (Weighted CrossEntropy or Focal Loss).
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        raise ImportError("PyTorch required for loss function construction.")

    weight_tensor = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None

    if loss_type == "weighted_ce":
        return nn.CrossEntropyLoss(weight=weight_tensor)
    elif loss_type == "focal":
        class MultiClassFocalLoss(nn.Module):
            def __init__(self, weights=None, gamma=2.0):
                super().__init__()
                self.weights = weights
                self.gamma = gamma

            def forward(self, inputs, targets):
                ce_loss = F.cross_entropy(inputs, targets, weight=self.weights, reduction="none")
                pt = torch.exp(-ce_loss)
                focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
                return focal_loss.mean()

        return MultiClassFocalLoss(weights=weight_tensor, gamma=gamma)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
