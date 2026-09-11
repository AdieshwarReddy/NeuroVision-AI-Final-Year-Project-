"""
EfficientNet Architectures for Brain Tumor MRI Classification.
Supports EfficientNetB0, EfficientNetB3, and EfficientNetV2-S with transfer learning.
"""

from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights
)


class BrainTumorEfficientNet(nn.Module):
    """
    EfficientNet wrapper with configurable variant (b0, b3, v2s),
    Global Average Pooling, Dropout, and Linear classification head.
    """
    def __init__(
        self,
        variant: str = "b0",
        num_classes: int = 3,
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.variant = variant.lower()
        self.num_classes = num_classes
        
        if self.variant == "b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base_model = efficientnet_b0(weights=weights)
            in_features = base_model.classifier[1].in_features  # 1280
        elif self.variant == "b3":
            weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
            base_model = efficientnet_b3(weights=weights)
            in_features = base_model.classifier[1].in_features  # 1536
        elif self.variant == "v2s":
            weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            base_model = efficientnet_v2_s(weights=weights)
            in_features = base_model.classifier[1].in_features  # 1280
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {variant}. Choose 'b0', 'b3', or 'v2s'.")
            
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        self.feature_dim = in_features
        
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts spatial feature maps before pooling."""
        return self.features(x)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.forward_features(x)
        feat_pooled = self.avgpool(feat_map)
        feat_flat = torch.flatten(feat_pooled, 1)
        logits = self.classifier(feat_flat)
        return logits
        
    def freeze_backbone(self):
        """Freezes all feature extractor layers for Stage 1 head training."""
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
            
    def unfreeze_upper_blocks(self, num_blocks: int = 2):
        """
        Unfreezes the last `num_blocks` stages of the feature extractor and head for Stage 2.
        """
        for param in self.features.parameters():
            param.requires_grad = False
            
        # EfficientNet features is an nn.Sequential of MBConv blocks
        total_layers = len(self.features)
        for i in range(max(0, total_layers - num_blocks), total_layers):
            for param in self.features[i].parameters():
                param.requires_grad = True
                
        for param in self.classifier.parameters():
            param.requires_grad = True
            
    def unfreeze_all(self):
        """Unfreezes all parameters for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
