"""
Model Factory for TrustBT-EfficientNet.
Instantiates backbones (ResNet50, EfficientNetB0, EfficientNetB3, EfficientNetV2-S).
"""

from typing import Dict, Any, Union
import torch.nn as nn
from src.models.resnet import BrainTumorResNet50
from src.models.efficientnet import BrainTumorEfficientNet


def create_model(
    model_name: str,
    num_classes: int = 3,
    pretrained: bool = True,
    dropout_rate: float = 0.3
) -> nn.Module:
    """
    Factory function for model instantiation.
    
    Args:
        model_name: 'resnet50', 'efficientnet_b0', 'efficientnet_b3', 'efficientnet_v2s'
        num_classes: Number of output classes (default: 3)
        pretrained: Whether to load ImageNet pretrained weights
        dropout_rate: Head dropout probability
        
    Returns:
        nn.Module model instance.
    """
    name_clean = model_name.lower().replace("-", "_")
    
    if name_clean == "resnet50":
        return BrainTumorResNet50(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    elif name_clean in ["efficientnet_b0", "effnet_b0", "b0"]:
        return BrainTumorEfficientNet(
            variant="b0",
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    elif name_clean in ["efficientnet_b3", "effnet_b3", "b3"]:
        return BrainTumorEfficientNet(
            variant="b3",
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    elif name_clean in ["efficientnet_v2s", "effnet_v2s", "v2s"]:
        return BrainTumorEfficientNet(
            variant="v2s",
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(
            f"Unknown model name: {model_name}. Supported: 'resnet50', 'efficientnet_b0', 'efficientnet_b3', 'efficientnet_v2s'"
        )
