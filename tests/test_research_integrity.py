"""
Research Integrity Unit Tests for TrustBT-EfficientNet.
Validates zero patient overlap, tensor dimensions, probability normalization,
and model output shapes.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pytest

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import normalize_mri_intensity, mri_to_3channel_tensor, preprocess_single_mri
from src.data.splitting import verify_zero_patient_overlap
from src.models.factory import create_model


def test_zero_patient_overlap():
    """Mathematical verification of zero patient leakage across splits."""
    train_df = pd.DataFrame({"patient_id": ["P1", "P2", "P3"], "label": [1, 2, 3]})
    val_df = pd.DataFrame({"patient_id": ["P4", "P5"], "label": [1, 2]})
    test_df = pd.DataFrame({"patient_id": ["P6", "P7"], "label": [2, 3]})
    
    # Must pass cleanly
    res = verify_zero_patient_overlap(train_df, val_df, test_df)
    assert res["status"] == "PASSED_ZERO_LEAKAGE"
    assert res["total_unique_patients"] == 7
    
    # Must raise AssertionError on leakage
    leak_val_df = pd.DataFrame({"patient_id": ["P1", "P5"], "label": [1, 2]})
    with pytest.raises(AssertionError):
        verify_zero_patient_overlap(train_df, leak_val_df, test_df)


def test_mri_intensity_normalization():
    """Verifies that MRI normalization bounds outputs strictly to [0.0, 1.0]."""
    mock_mri = np.random.randint(0, 4095, size=(512, 512), dtype=np.uint16)
    norm = normalize_mri_intensity(mock_mri)
    
    assert norm.shape == (512, 512)
    assert norm.dtype == np.float32
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0


def test_tensor_3channel_shape():
    """Verifies single-channel to 3-channel replication for ImageNet backbones."""
    mock_norm = np.random.rand(224, 224).astype(np.float32)
    tensor = mri_to_3channel_tensor(mock_norm)
    
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    # Verify channels 0, 1, 2 are identical replicas
    assert torch.equal(tensor[0], tensor[1])
    assert torch.equal(tensor[1], tensor[2])


def test_model_forward_pass_dimensions():
    """Verifies forward pass output shapes and logits across all candidate architectures."""
    for model_name in ["resnet50", "efficientnet_b0", "efficientnet_b3"]:
        model = create_model(model_name=model_name, num_classes=3, pretrained=False)
        model.eval()
        
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(dummy_input)
            probs = torch.softmax(logits, dim=1)
            
        assert logits.shape == (2, 3), f"Failed shape for {model_name}"
        assert torch.allclose(probs.sum(dim=1), torch.ones(2)), f"Probs do not sum to 1 for {model_name}"


if __name__ == "__main__":
    print("[*] Running integrity unit tests...")
    test_zero_patient_overlap()
    print("[PASS] Passed: test_zero_patient_overlap")
    test_mri_intensity_normalization()
    print("[PASS] Passed: test_mri_intensity_normalization")
    test_tensor_3channel_shape()
    print("[PASS] Passed: test_tensor_3channel_shape")
    test_model_forward_pass_dimensions()
    print("[PASS] Passed: test_model_forward_pass_dimensions")
    print("\n[OK] ALL RESEARCH INTEGRITY UNIT TESTS PASSED!")
