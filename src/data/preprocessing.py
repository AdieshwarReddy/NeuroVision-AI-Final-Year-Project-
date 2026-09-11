"""
TrustBT-EfficientNet Preprocessing Pipeline.
Implements intensity normalization, resizing, channel adaptation, and medically plausible augmentations.
"""

from typing import Tuple, Optional, Union
import numpy as np
import torch
import torchvision.transforms.v2 as T
from torchvision.transforms.v2 import functional as F
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def normalize_mri_intensity(image_np: np.ndarray) -> np.ndarray:
    """
    Min-max normalize raw MRI uint16/float array to [0.0, 1.0].
    
    Args:
        image_np: 2D numpy array of MRI slice.
        
    Returns:
        float32 2D numpy array in range [0.0, 1.0].
    """
    image_np = image_np.astype(np.float32)
    min_val = np.min(image_np)
    max_val = np.max(image_np)
    
    if max_val - min_val > 1e-7:
        norm_img = (image_np - min_val) / (max_val - min_val)
    else:
        norm_img = np.zeros_like(image_np, dtype=np.float32)
        
    return norm_img


def mri_to_3channel_tensor(image_norm: np.ndarray) -> torch.Tensor:
    """
    Converts 2D normalized float32 numpy array [H, W] to 3-channel PyTorch FloatTensor [3, H, W].
    """
    # Replicate across 3 channels for transfer learning backbones
    tensor_1ch = torch.from_numpy(image_norm).unsqueeze(0)  # [1, H, W]
    tensor_3ch = tensor_1ch.repeat(3, 1, 1)  # [3, H, W]
    return tensor_3ch.float()


def get_mri_transforms(
    target_size: Tuple[int, int] = (224, 224),
    is_training: bool = True
) -> T.Compose:
    """
    Constructs torchvision v2 transform pipeline.
    
    Args:
        target_size: (height, width) tuple (e.g. 224x224 or 300x300).
        is_training: If True, applies medically plausible subtle augmentations.
        
    Returns:
        torchvision.transforms.v2.Compose transform object.
    """
    transform_list = [
        T.Resize(target_size, antialias=True),
    ]
    
    if is_training:
        transform_list.extend([
            T.RandomRotation(degrees=(-10, 10)),
            T.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05)
            ),
            T.RandomHorizontalFlip(p=0.5),
        ])
        
    transform_list.append(
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    )
    
    return T.Compose(transform_list)


def preprocess_single_mri(
    image_np: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    apply_imagenet_norm: bool = True
) -> torch.Tensor:
    """
    Convenience function to preprocess an individual 2D MRI array for inference.
    
    Returns:
        torch.Tensor of shape [1, 3, target_h, target_w] ready for model input.
    """
    norm_img = normalize_mri_intensity(image_np)
    tensor_3ch = mri_to_3channel_tensor(norm_img)  # [3, H, W]
    
    # Resize
    tensor_resized = F.resize(tensor_3ch, target_size, antialias=True)
    
    if apply_imagenet_norm:
        tensor_normalized = F.normalize(tensor_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    else:
        tensor_normalized = tensor_resized
        
    return tensor_normalized.unsqueeze(0)  # [1, 3, H, W]
