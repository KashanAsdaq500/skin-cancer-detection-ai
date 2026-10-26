"""
Dermora AI - Model Architecture Definition
Supports Transfer Learning with EfficientNet-B0 or ResNet34.
Replaces ImageNet 1000-class head with custom 7-class dermatological lesion classifier.
"""

from typing import Tuple

def build_classifier_model(
    backbone_name: str = "efficientnet_b0",
    num_classes: int = 7,
    pretrained: bool = True,
    dropout_rate: float = 0.3
):
    """
    Constructs a PyTorch model with pre-trained vision backbone and custom classifier head.
    Dynamically imports torch and torchvision to remain safe when torch is not yet installed.
    """
    try:
        import torch
        import torch.nn as nn
        from torchvision import models
    except ImportError:
        raise ImportError(
            "PyTorch and torchvision are required to build the training model. "
            "Install them via 'pip install torch torchvision'."
        )

    backbone_name = backbone_name.lower().replace("-", "_")

    if backbone_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(256, num_classes)
        )
    elif backbone_name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )
    else:
        raise ValueError(f"Unsupported backbone '{backbone_name}'. Choose 'efficientnet_b0' or 'resnet34'.")

    return model
