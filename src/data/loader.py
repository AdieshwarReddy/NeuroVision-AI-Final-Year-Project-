"""
TrustBT-EfficientNet PyTorch Dataset and DataLoader Pipeline.
Loads MATLAB .mat Figshare brain tumor MRI slices, extracts metadata,
and constructs patient-grouped DataLoaders.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List, Union
import numpy as np
import pandas as pd
import h5py
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.v2 import functional as F

from src.data.preprocessing import (
    normalize_mri_intensity,
    mri_to_3channel_tensor,
    get_mri_transforms,
    IMAGENET_MEAN,
    IMAGENET_STD
)

# Label Mapping: Figshare uses 1: meningioma, 2: glioma, 3: pituitary
LABEL_MAP = {
    1: 0,  # meningioma -> 0
    2: 1,  # glioma -> 1
    3: 2   # pituitary -> 2
}

INV_LABEL_MAP = {
    0: "meningioma",
    1: "glioma",
    2: "pituitary"
}


def load_mat_slice(mat_filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Robustly loads a single Figshare .mat file (MATLAB v7.3 HDF5).
    
    Returns a dict with:
        - image: np.ndarray [512, 512] float32
        - label: int (1, 2, or 3)
        - pid: str (Patient ID)
        - tumor_mask: np.ndarray [512, 512] uint8 (0 or 1)
        - tumor_border: np.ndarray coordinates
    """
    mat_path = Path(mat_filepath)
    if not mat_path.exists():
        raise FileNotFoundError(f"File not found: {mat_path}")
        
    try:
        # Figshare Jun Cheng dataset is MATLAB v7.3 (HDF5)
        with h5py.File(str(mat_path), "r") as f:
            cjdata = f["cjdata"]
            label = int(np.array(cjdata["label"]).item())
            
            # Extract PID string from unicode array
            pid_data = np.array(cjdata["PID"])
            pid = "".join(chr(c[0]) for c in pid_data) if pid_data.ndim > 1 else str(pid_data)
            
            # In HDF5 MATLAB matrices are stored transposed [W, H]
            image = np.array(cjdata["image"]).T.astype(np.float32)
            tumor_mask = np.array(cjdata["tumorMask"]).T.astype(np.uint8)
            tumor_border = np.array(cjdata["tumorBorder"]).T
            
    except Exception:
        # Fallback for legacy v7/v6 format
        mat_dict = sio.loadmat(str(mat_path))
        cjdata = mat_dict["cjdata"]
        label = int(cjdata["label"][0, 0][0, 0])
        pid_raw = cjdata["PID"][0, 0]
        pid = str(pid_raw[0]) if len(pid_raw) > 0 else "UNKNOWN"
        image = cjdata["image"][0, 0].astype(np.float32)
        tumor_mask = cjdata["tumorMask"][0, 0].astype(np.uint8)
        tumor_border = cjdata["tumorBorder"][0, 0]
            
    return {
        "image": image,
        "label": label,
        "pid": str(pid).strip(),
        "tumor_mask": tumor_mask,
        "tumor_border": tumor_border,
        "filepath": str(mat_path)
    }


class BrainTumorMRIDataset(Dataset):
    """
    PyTorch Dataset for Brain Tumor MRI classification from .mat files or manifest CSV.
    Supports in-memory caching to accelerate multi-epoch training.
    """
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        target_size: Tuple[int, int] = (224, 224),
        is_training: bool = False,
        return_mask: bool = False,
        cache_in_memory: bool = True
    ):
        self.df = manifest_df.reset_index(drop=True)
        self.target_size = target_size
        self.is_training = is_training
        self.return_mask = return_mask
        self.cache_in_memory = cache_in_memory
        self.cache = {}
        self.transform = get_mri_transforms(target_size=target_size, is_training=is_training)
        
    def __len__(self) -> int:
        return len(self.df)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.cache_in_memory and idx in self.cache:
            data = self.cache[idx]
        else:
            row = self.df.iloc[idx]
            filepath = row["filepath"]
            data = load_mat_slice(filepath)
            if self.cache_in_memory:
                self.cache[idx] = data
                
        raw_image = data["image"]
        raw_label = data["label"]
        pid = data["pid"]
        tumor_mask = data["tumor_mask"]
        filepath = data["filepath"]
        
        # Zero-index class label
        mapped_label = LABEL_MAP.get(raw_label, 0)
        
        # Normalize and construct tensor
        norm_img = normalize_mri_intensity(raw_image)
        tensor_3ch = mri_to_3channel_tensor(norm_img)  # [3, H, W]
        
        # Apply transforms
        tensor_transformed = self.transform(tensor_3ch)
        
        sample = {
            "image": tensor_transformed,
            "label": torch.tensor(mapped_label, dtype=torch.long),
            "pid": pid,
            "filepath": filepath
        }
        
        if self.return_mask:
            # Resize mask to match target size using nearest neighbor
            mask_tensor = torch.from_numpy(tumor_mask).unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
            mask_resized = F.resize(mask_tensor, self.target_size, interpolation=F.InterpolationMode.NEAREST)
            sample["tumor_mask"] = (mask_resized.squeeze(0).squeeze(0) > 0.5).to(torch.uint8)
            
        return sample


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    target_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    num_workers: int = 0,
    cache_in_memory: bool = True
) -> Dict[str, DataLoader]:
    """
    Factory function creating PyTorch DataLoaders for train, val, and test splits.
    """
    train_ds = BrainTumorMRIDataset(train_df, target_size=target_size, is_training=True, cache_in_memory=cache_in_memory)
    val_ds = BrainTumorMRIDataset(val_df, target_size=target_size, is_training=False, return_mask=True, cache_in_memory=cache_in_memory)
    
    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    }
    
    if test_df is not None:
        test_ds = BrainTumorMRIDataset(test_df, target_size=target_size, is_training=False, return_mask=True, cache_in_memory=cache_in_memory)
        loaders["test"] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
    return loaders
