"""
ResNet Baseline Architecture for Brain Tumor MRI Classification.
Implements ResNet50 backbone with customizable classification head and layer freezing.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class BrainTumorResNet50(nn.Module):
    """
    ResNet-50 backbone with Global Average Pooling, Dropout, and Linear classification head.
    """
    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)
        
        in_features = self.backbone.fc.in_features  # 2048
        
        # Replace final fully connected layer with Dropout + Linear
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        
        self.feature_dim = in_features
        
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts spatial feature maps before global pooling."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.forward_features(x)
        feat_pooled = self.backbone.avgpool(feat_map)
        feat_flat = torch.flatten(feat_pooled, 1)
        logits = self.head(feat_flat)
        return logits
        
    def freeze_backbone(self):
        """Freezes all backbone parameters for Stage 1 head training."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True
            
    def unfreeze_upper_blocks(self):
        """Unfreezes layer4 and head for Stage 2 fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
        for param in self.head.parameters():
            param.requires_grad = True
            
    def unfreeze_all(self):
        """Unfreezes all parameters for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
