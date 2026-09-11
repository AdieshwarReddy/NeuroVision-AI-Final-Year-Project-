"""
Post-Training XAI + Calibration Evaluation Script for TrustBT-EfficientNet.

Runs AFTER a training experiment completes. Performs:
1. Loads best_model.pt from experiment directory
2. Runs Temperature Scaling calibration on validation fold
3. Generates Grad-CAM++ heatmaps for test fold samples
4. Saves publication-quality figures, JSON reports
5. Exports reliability diagrams

Usage:
    python scripts/evaluate_xai.py --exp_dir artifacts/experiments/candidate_efficientnet_b0_fold1
    python scripts/evaluate_xai.py --exp_dir artifacts/experiments/candidate_efficientnet_b0_fold1 --num_gradcam 30
"""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import torch

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.factory import create_model
from src.data.loader import create_dataloaders
from src.evaluation.gradcam import build_gradcam, generate_gradcam_for_batch
from src.evaluation.calibration import run_temperature_scaling


def run_xai_calibration(
    exp_dir: str,
    num_gradcam: int = 20,
    gradcam_method: str = "gradcam++"
):
    exp_path = Path(exp_dir)
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_path}")

    # Load config
    config_path = exp_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {exp_path}")
    with open(config_path) as f:
        cfg = json.load(f)

    print(f"\n{'='*70}")
    print(f"XAI + Calibration Evaluation: {exp_path.name}")
    print(f"Model: {cfg.get('model_name')} | Fold: {cfg.get('fold')}")
    print(f"{'='*70}")

    # Load fold manifest
    fold = cfg.get("fold", 1)
    split_csv = Path("artifacts/splits") / f"fold_{fold}.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Split manifest not found: {split_csv}")

    fold_df = pd.read_csv(split_csv)
    train_df = fold_df[fold_df["split"] == "train"].copy()
    val_df = fold_df[fold_df["split"] == "val"].copy()
    test_df = fold_df[fold_df["split"] == "test"].copy()

    print(f"[*] Loaded split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Create DataLoaders (small batch, return masks for XAI overlay)
    target_size = tuple(cfg.get("target_size", [224, 224]))
    loaders = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_size=target_size,
        batch_size=8,          # Small batch for inference to save RAM
        num_workers=0,
        cache_in_memory=False  # Don't cache during eval — save RAM
    )

    # Instantiate model
    model = create_model(
        model_name=cfg.get("model_name", "efficientnet_b0"),
        num_classes=3,
        pretrained=False,
        dropout_rate=cfg.get("dropout_rate", 0.3)
    )

    # Load best checkpoint
    checkpoint_path = exp_path / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"best_model.pt not found in {exp_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    print(f"[OK] Loaded best checkpoint from {checkpoint_path}")

    device = torch.device("cpu")  # Intel Iris Xe — always CPU

    # -----------------------------------------------------------------------
    # 1. Temperature Scaling Calibration
    # -----------------------------------------------------------------------
    print("\n[+] Running Temperature Scaling Calibration on validation fold...")
    calibration_dir = exp_path / "calibration"
    cal_results = run_temperature_scaling(
        model=model,
        val_loader=loaders["val"],
        device=device,
        output_dir=calibration_dir,
        max_iter=100,
        verbose=True
    )
    print(f"[OK] Calibration: T={cal_results['temperature']:.4f} | "
          f"ECE: {cal_results['ece_before']:.4f} -> {cal_results['ece_after']:.4f}")

    # -----------------------------------------------------------------------
    # 2. Grad-CAM++ Generation on Test Fold
    # -----------------------------------------------------------------------
    print(f"\n[+] Generating {num_gradcam} Grad-CAM++ panels on held-out test fold...")
    gradcam_dir = exp_path / "gradcam"

    cam_engine = build_gradcam(
        model=model,
        model_name=cfg.get("model_name", "efficientnet_b0"),
        method=gradcam_method
    )

    cam_results = generate_gradcam_for_batch(
        model=model,
        dataloader=loaders["test"],
        target_layer=cam_engine.target_layer,
        output_dir=gradcam_dir,
        num_samples=num_gradcam,
        method=gradcam_method,
        device=device
    )

    # Save Grad-CAM summary
    gradcam_summary_path = exp_path / "gradcam_summary.json"
    with open(gradcam_summary_path, "w") as f:
        # Convert any non-serialisable fields
        clean = [{k: v for k, v in r.items() if isinstance(v, (str, int, float, bool))} for r in cam_results]
        json.dump(clean, f, indent=4)

    print(f"[OK] Grad-CAM panels saved to: {gradcam_dir}")

    # -----------------------------------------------------------------------
    # Final Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"[OK] XAI + Calibration complete for {exp_path.name}")
    print(f"  Temperature:     T = {cal_results['temperature']:.4f}")
    print(f"  ECE (before):    {cal_results['ece_before']:.4f}")
    print(f"  ECE (after TS):  {cal_results['ece_after']:.4f}")
    print(f"  Grad-CAM panels: {len(cam_results)}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-training XAI + Calibration Evaluator")
    parser.add_argument("--exp_dir", type=str, required=True, help="Path to experiment directory")
    parser.add_argument("--num_gradcam", type=int, default=20, help="Number of Grad-CAM panels to generate")
    parser.add_argument("--method", type=str, default="gradcam++", choices=["gradcam", "gradcam++"],
                        help="Grad-CAM method to use")
    args = parser.parse_args()

    run_xai_calibration(
        exp_dir=args.exp_dir,
        num_gradcam=args.num_gradcam,
        gradcam_method=args.method
    )
