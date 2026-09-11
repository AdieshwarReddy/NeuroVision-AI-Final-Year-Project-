"""
Patient-Disjoint Split Generator for TrustBT-EfficientNet.
Generates 5-fold StratifiedGroupKFold manifests and verifies zero patient overlap.
"""

import sys
import json
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.splitting import generate_patient_disjoint_folds, verify_zero_patient_overlap


def create_splits_from_manifest(
    manifest_csv_path: str = "artifacts/dataset_manifest_raw.csv",
    output_dir: str = "artifacts/splits",
    n_splits: int = 5,
    seed: int = 42
):
    csv_p = Path(manifest_csv_path)
    if not csv_p.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {csv_p}. Run audit_dataset.py first.")
        
    df = pd.read_csv(csv_p)
    print(f"[*] Loaded raw dataset manifest: {len(df)} slices from {df['patient_id'].nunique()} patients.")
    
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Generating {n_splits} patient-disjoint folds (seed={seed})...")
    folds = generate_patient_disjoint_folds(
        manifest_df=df,
        n_splits=n_splits,
        random_state=seed,
        output_dir=out_p,
        save_csv=True
    )
    
    summary = {
        "n_splits": n_splits,
        "random_seed": seed,
        "folds": []
    }
    
    for f_info in folds:
        k = f_info["fold"]
        tr_df = f_info["train"]
        val_df = f_info["val"]
        test_df = f_info["test"]
        
        # Verify mathematically
        overlap_res = verify_zero_patient_overlap(tr_df, val_df, test_df, pid_col="patient_id")
        
        fold_stat = {
            "fold": k,
            "verification": overlap_res,
            "train_slices": len(tr_df),
            "val_slices": len(val_df),
            "test_slices": len(test_df),
            "train_class_dist": tr_df["class_name"].value_counts().to_dict(),
            "val_class_dist": val_df["class_name"].value_counts().to_dict(),
            "test_class_dist": test_df["class_name"].value_counts().to_dict()
        }
        summary["folds"].append(fold_stat)
        print(f"[OK] Fold {k} generated: Train={len(tr_df)} ({overlap_res['train_patients']} pts), "
              f"Val={len(val_df)} ({overlap_res['val_patients']} pts), "
              f"Test={len(test_df)} ({overlap_res['test_patients']} pts) | ZERO LEAKAGE VERIFIED")
              
    summary_path = out_p / "splits_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"[OK] All {n_splits} patient-disjoint split manifests saved to {out_p}")
    return summary


if __name__ == "__main__":
    create_splits_from_manifest()
