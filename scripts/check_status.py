"""
Quick experiment status checker.
Run from project root to see all trained models and their results.

Usage: python scripts/check_status.py
"""

import sys
import json
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "artifacts" / "experiments"
SPLITS_DIR = ROOT / "artifacts" / "splits"


def print_header(text, char="="):
    print(f"\n{char * 65}")
    print(f"  {text}")
    print(f"{char * 65}")


def check_dataset():
    print_header("DATASET STATUS", "=")
    raw_mat_dir = ROOT / "data" / "raw_mat"
    if raw_mat_dir.exists():
        mat_files = list(raw_mat_dir.glob("*.mat"))
        print(f"  .mat files found: {len(mat_files)} / 3064")
        if len(mat_files) >= 3064:
            print(f"  [OK] Full dataset present ({len(mat_files)} files)")
        elif len(mat_files) > 0:
            print("  [PARTIAL] Dataset incomplete — run: python scripts/download_figshare.py")
        else:
            print("  [MISSING] Run: python scripts/download_figshare.py")
    else:
        print("  [MISSING] data/raw_mat/ directory not found")


def check_splits():
    print_header("SPLITS STATUS", "-")
    if SPLITS_DIR.exists():
        fold_csvs = list(SPLITS_DIR.glob("fold_*.csv"))
        print(f"  Fold manifests: {len(fold_csvs)} / 5")
        summary = SPLITS_DIR / "splits_summary.json"
        if summary.exists():
            with open(summary) as f:
                s = json.load(f)
            print("  Split summary:")
            for fold in s.get("folds", []):
                k = fold["fold"]
                v = fold["verification"]
                print(f"    Fold {k}: Train={fold['train_slices']} slices "
                      f"({v['train_patients']} pts), "
                      f"Val={fold['val_slices']} ({v['val_patients']} pts), "
                      f"Test={fold['test_slices']} ({v['test_patients']} pts) "
                      f"| {v['status']}")
        if len(fold_csvs) == 5:
            print("  [OK] All 5 folds generated with zero leakage")
    else:
        print("  [MISSING] Run: python scripts/create_splits.py")


def check_experiments():
    print_header("EXPERIMENT RESULTS", "-")
    if not EXPERIMENTS_DIR.exists() or not any(EXPERIMENTS_DIR.iterdir()):
        print("  [NONE] No experiments yet. Run: python scripts/train.py --config configs/...")
        return

    for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue

        checkpoint = exp_dir / "best_model.pt"
        metrics_path = exp_dir / "metrics.json"
        cal_path = exp_dir / "calibration" / "calibration_summary.json"
        gradcam_path = exp_dir / "gradcam"

        has_checkpoint = checkpoint.exists()
        has_metrics = metrics_path.exists()

        print(f"\n  [{exp_dir.name}]")
        print(f"    Model checkpoint: {'[OK]' if has_checkpoint else '[MISSING]'}")

        if has_metrics:
            with open(metrics_path) as f:
                m = json.load(f)
            tm = m.get("test_metrics", m.get("val_metrics", {}))
            print(f"    Best Epoch:     {m.get('best_epoch', '?')}")
            print(f"    Test Acc:       {tm.get('accuracy', 0):.4f}")
            print(f"    Test Macro F1:  {tm.get('macro_f1', 0):.4f}")
            print(f"    Test AUC-ROC:   {tm.get('roc_auc_macro_ovr', 0):.4f}")
            print(f"    Test ECE:       {tm.get('expected_calibration_error', 0):.4f}")
        else:
            print("    [TRAINING IN PROGRESS or FAILED]")

        if cal_path.exists():
            with open(cal_path) as f:
                cal = json.load(f)
            print(f"    Temperature Scaling: T={cal['optimal_temperature']:.4f} "
                  f"| ECE: {cal['ece_before']:.4f} -> {cal['ece_after']:.4f} "
                  f"(-{cal['ece_relative_improvement_pct']:.1f}%)")
        else:
            print("    Temperature Scaling: [NOT RUN] - run evaluate_xai.py")

        if gradcam_path.exists():
            panels = list(gradcam_path.glob("*.png"))
            print(f"    Grad-CAM++ panels: {len(panels)}")
        else:
            print("    Grad-CAM++ panels: [NOT GENERATED]")


def check_streamlit():
    print_header("STREAMLIT DASHBOARD", "-")
    try:
        import importlib.util
        spec = importlib.util.find_spec("streamlit")
        if spec:
            print("  [OK] Streamlit installed")
            print("  Launch: .\\run_dashboard.ps1")
        else:
            print("  [MISSING] Run: .venv\\Scripts\\pip install streamlit")
    except Exception:
        print("  [UNKNOWN] Could not check streamlit")


def main():
    print("\n" + "=" * 65)
    print("  TrustBT-EfficientNet / NeuroVision AI — Project Status")
    print("=" * 65)

    check_dataset()
    check_splits()
    check_experiments()
    check_streamlit()

    print("\n" + "=" * 65)
    print("  NEXT STEPS (in order):")
    print("  1. [RUNNING] EfficientNet-B0 training (Stage 1/2)")
    print("  2. After B0 done: python scripts/evaluate_xai.py --exp_dir ...")
    print("  3. python scripts/train.py --config configs/efficientnet_b3.yaml")
    print("  4. After B3 done: python scripts/evaluate_xai.py --exp_dir ...")
    print("  5. python scripts/aggregate_results.py")
    print("  6. .\\run_dashboard.ps1 to launch NeuroVision AI")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
