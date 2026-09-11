"""
Master Training Script for TrustBT-EfficientNet.
Runs reproducible training experiments from YAML configs and validates zero patient leakage.
"""

import sys
import re
import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import torch

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.factory import create_model
from src.data.loader import create_dataloaders
from src.data.splitting import verify_zero_patient_overlap
from src.training.trainer import BrainTumorTrainer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_experiment(config_path: str, fold_override: int = None):
    # Load configuration
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
        
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    fold = fold_override if fold_override is not None else cfg.get("fold", 1)
    base_name = cfg.get('experiment_name', 'exp')
    clean_base = re.sub(r'_fold\d+$', '', base_name)
    exp_name = f"{clean_base}_fold{fold}"
    seed = cfg.get("random_seed", 42)
    set_seed(seed)
    
    print("=" * 70)
    print(f"EXPERIMENT: {exp_name}")
    print(f"Model: {cfg.get('model_name')} | Fold: {fold} | Seed: {seed}")
    print("=" * 70)
    
    # Load fold manifest
    split_csv = Path("artifacts/splits") / f"fold_{fold}.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Fold manifest not found: {split_csv}. Please run scripts/create_splits.py first.")
        
    fold_df = pd.read_csv(split_csv)
    train_df = fold_df[fold_df["split"] == "train"].copy()
    val_df = fold_df[fold_df["split"] == "val"].copy()
    test_df = fold_df[fold_df["split"] == "test"].copy()
    
    # CRITICAL RESEARCH UNIT TEST: Assert zero patient overlap
    overlap_report = verify_zero_patient_overlap(train_df, val_df, test_df, pid_col="patient_id")
    print(f"[OK] Zero Patient Leakage Verified: {overlap_report['train_patients']} train pts, "
          f"{overlap_report['val_patients']} val pts, {overlap_report['test_patients']} test pts.")
          
    # Target size
    target_size = tuple(cfg.get("target_size", [224, 224]))
    batch_size = cfg.get("batch_size", 32)
    num_workers = cfg.get("num_workers", 0)
    
    loaders = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_size=target_size,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Instantiate Model
    model = create_model(
        model_name=cfg.get("model_name", "resnet50"),
        num_classes=3,
        pretrained=cfg.get("pretrained", True),
        dropout_rate=cfg.get("dropout_rate", 0.3)
    )
    
    # Setup Trainer
    exp_dir = Path("artifacts/experiments") / exp_name
    trainer = BrainTumorTrainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        test_loader=loaders["test"],
        config=cfg,
        experiment_dir=exp_dir
    )
    
    results = trainer.train()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TrustBT-EfficientNet models")
    parser.add_argument("--config", type=str, default="configs/baseline_resnet50.yaml", help="Path to config YAML")
    parser.add_argument("--fold", type=int, default=None, help="Override fold number")
    args = parser.parse_args()
    
    run_experiment(config_path=args.config, fold_override=args.fold)
