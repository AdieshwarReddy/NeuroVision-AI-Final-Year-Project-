"""
TrustBT-EfficientNet Patient-Disjoint Splitting Module.
Guarantees strict patient-level isolation using StratifiedGroupKFold.
Mathematically enforces zero patient identity overlap across train, val, and test splits.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def verify_zero_patient_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pid_col: str = "patient_id"
) -> Dict[str, Any]:
    """
    Rigorously verifies zero patient overlap between train, val, and test sets.
    Raises AssertionError immediately if any intersection is detected.
    
    Returns:
        Summary dictionary with unique patient counts per split.
    """
    train_pids = set(train_df[pid_col].unique())
    val_pids = set(val_df[pid_col].unique())
    test_pids = set(test_df[pid_col].unique())
    
    # Assertions
    train_val_overlap = train_pids.intersection(val_pids)
    train_test_overlap = train_pids.intersection(test_pids)
    val_test_overlap = val_pids.intersection(test_pids)
    
    if len(train_val_overlap) > 0:
        raise AssertionError(
            f"CRITICAL RESEARCH INTEGRITY VIOLATION: Train-Val patient overlap detected! "
            f"Overlapping PIDs ({len(train_val_overlap)}): {train_val_overlap}"
        )
        
    if len(train_test_overlap) > 0:
        raise AssertionError(
            f"CRITICAL RESEARCH INTEGRITY VIOLATION: Train-Test patient overlap detected! "
            f"Overlapping PIDs ({len(train_test_overlap)}): {train_test_overlap}"
        )
        
    if len(val_test_overlap) > 0:
        raise AssertionError(
            f"CRITICAL RESEARCH INTEGRITY VIOLATION: Val-Test patient overlap detected! "
            f"Overlapping PIDs ({len(val_test_overlap)}): {val_test_overlap}"
        )
        
    return {
        "status": "PASSED_ZERO_LEAKAGE",
        "train_patients": len(train_pids),
        "val_patients": len(val_pids),
        "test_patients": len(test_pids),
        "total_unique_patients": len(train_pids | val_pids | test_pids)
    }


def generate_patient_disjoint_folds(
    manifest_df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    output_dir: Union[str, Path] = "artifacts/splits",
    save_csv: bool = True
) -> List[Dict[str, pd.DataFrame]]:
    """
    Generates n_splits patient-disjoint folds using StratifiedGroupKFold.
    Each fold partition assigns:
      - 3 folds -> Train (~60% patients)
      - 1 fold  -> Validation (~20% patients)
      - 1 fold  -> Test (~20% patients)
      
    Args:
        manifest_df: DataFrame with ['filepath', 'label', 'patient_id']
        n_splits: Number of cross-validation folds (default: 5)
        random_state: Random seed for reproducibility
        output_dir: Directory to save split manifests
        save_csv: Whether to write fold_k.csv to output_dir
        
    Returns:
        List of dicts: [{"fold": 1, "train": df_tr, "val": df_val, "test": df_test, "manifest": df_fold}, ...]
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    target_col = "label" if "label" in manifest_df.columns else "raw_label"
    X = manifest_df["filepath"].values
    y = manifest_df[target_col].values
    groups = manifest_df["patient_id"].values
    
    # Pre-compute fold assignment for each slice
    fold_assignments = np.zeros(len(manifest_df), dtype=int)
    for fold_idx, (_, test_indices) in enumerate(sgkf.split(X, y, groups=groups)):
        fold_assignments[test_indices] = fold_idx
        
    df_with_folds = manifest_df.copy()
    df_with_folds["outer_fold"] = fold_assignments
    if "label" not in df_with_folds.columns and "raw_label" in df_with_folds.columns:
        df_with_folds["label"] = df_with_folds["raw_label"]
        
    out_path = Path(output_dir)
    if save_csv:
        out_path.mkdir(parents=True, exist_ok=True)
        
    folds_data = []
    
    for k in range(n_splits):
        test_fold = k
        val_fold = (k + 1) % n_splits
        train_folds = [f for f in range(n_splits) if f != test_fold and f != val_fold]
        
        train_df = df_with_folds[df_with_folds["outer_fold"].isin(train_folds)].copy()
        val_df = df_with_folds[df_with_folds["outer_fold"] == val_fold].copy()
        test_df = df_with_folds[df_with_folds["outer_fold"] == test_fold].copy()
        
        # Verify zero patient overlap mathematically
        verify_zero_patient_overlap(train_df, val_df, test_df, pid_col="patient_id")
        
        # Build unified manifest for this fold
        train_df["split"] = "train"
        val_df["split"] = "val"
        test_df["split"] = "test"
        
        fold_manifest = pd.concat([train_df, val_df, test_df], ignore_index=True)
        cols_to_keep = [c for c in ["filepath", "patient_id", "label", "class_name", "split"] if c in fold_manifest.columns]
        fold_manifest = fold_manifest[cols_to_keep]
        
        if save_csv:
            csv_path = out_path / f"fold_{k+1}.csv"
            fold_manifest.to_csv(csv_path, index=False)
            
        folds_data.append({
            "fold": k + 1,
            "train": train_df,
            "val": val_df,
            "test": test_df,
            "manifest": fold_manifest
        })
        
    return folds_data
