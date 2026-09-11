"""
Multi-Fold Training & Evaluation Orchestrator for TrustBT-EfficientNet.

Trains a specified configuration across multiple folds sequentially,
runs post-training calibration (Temperature Scaling) and Grad-CAM++,
and aggregates results after each fold.

Usage:
    python scripts/train_multifold.py --config configs/efficientnet_b0.yaml --folds 2 3 4 5
    python scripts/train_multifold.py --config configs/efficientnet_b3.yaml --folds 2 3 4 5
"""

import sys
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_cmd(cmd_list):
    print(f"\n{'='*70}")
    print(f"RUNNING: {' '.join(str(x) for x in cmd_list)}")
    print(f"{'='*70}\n")
    res = subprocess.run(cmd_list, cwd=str(ROOT))
    if res.returncode != 0:
        print(f"[!] Warning: Command failed with code {res.returncode}")
    return res.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Multi-Fold Training Orchestrator")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--folds", type=int, nargs="+", default=[2, 3, 4, 5], help="List of folds to train")
    parser.add_argument("--num_gradcam", type=int, default=12, help="Number of Grad-CAM panels per fold")
    args = parser.parse_args()

    python_exe = sys.executable

    import yaml, re
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    base_name = cfg.get("experiment_name", "exp")
    clean_base = re.sub(r"_fold\d+$", "", base_name)

    for fold in args.folds:
        print(f"\n{'#'*70}")
        print(f"  STARTING FOLD {fold} / 5 : {clean_base}")
        print(f"{'#'*70}\n")

        # 1. Train fold
        train_cmd = [
            python_exe, "scripts/train.py",
            "--config", args.config,
            "--fold", str(fold)
        ]
        success = run_cmd(train_cmd)
        if not success:
            print(f"[!] Training failed for fold {fold}. Continuing to next fold.")
            continue

        # 2. Run XAI + Calibration on the newly trained fold
        exp_dir = ROOT / "artifacts" / "experiments" / f"{clean_base}_fold{fold}"
        if exp_dir.exists() and (exp_dir / "best_model.pt").exists():
            xai_cmd = [
                python_exe, "scripts/evaluate_xai.py",
                "--exp_dir", str(exp_dir),
                "--num_gradcam", str(args.num_gradcam),
                "--method", "gradcam++"
            ]
            run_cmd(xai_cmd)

        # 3. Update cross-fold aggregated results
        agg_cmd = [python_exe, "scripts/aggregate_results.py"]
        run_cmd(agg_cmd)

    print(f"\n{'='*70}")
    print("Multi-fold training run completed!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
