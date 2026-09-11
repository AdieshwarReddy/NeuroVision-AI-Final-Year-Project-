"""
Trainer Module for TrustBT-EfficientNet.
Implements two-stage transfer learning, cosine learning-rate scheduling,
Validation Macro F1 checkpointing, and comprehensive experiment logging.
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score
from src.evaluation.metrics import (
    calculate_classification_metrics,
    plot_confusion_matrix,
    plot_multiclass_roc_curves,
    plot_training_history,
    CLASS_NAMES
)


class BrainTumorTrainer:
    """
    Two-stage transfer learning trainer with Macro F1 checkpointing.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        experiment_dir: Union[str, Path] = "artifacts/experiments/exp_default"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config or {}
        
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model.to(self.device)
        
        self.exp_dir = Path(experiment_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function with optional class weighting
        use_class_weights = self.config.get("use_class_weights", False)
        if use_class_weights and "class_weights" in self.config:
            weights_tensor = torch.tensor(self.config["class_weights"], dtype=torch.float, device=self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.history = []
        self.best_val_macro_f1 = -1.0
        self.best_epoch = -1
        
    def _train_one_epoch(self, optimizer: optim.Optimizer) -> Tuple[float, float, float]:
        """Trains for one epoch and returns (train_loss, train_acc, train_macro_f1)."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        for b_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            targets = batch["label"].to(self.device)
            
            optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.detach().cpu().numpy())
            
        avg_loss = total_loss / len(self.train_loader.dataset)
        tr_acc = float(accuracy_score(all_targets, all_preds))
        tr_macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
        return avg_loss, tr_acc, tr_macro_f1
        
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluates model on given dataloader.
        Returns (avg_loss, metrics_dict, y_true, y_pred, y_prob).
        """
        self.model.eval()
        total_loss = 0.0
        all_targets = []
        all_logits = []
        
        for batch in dataloader:
            images = batch["image"].to(self.device)
            targets = batch["label"].to(self.device)
            
            logits = self.model(images)
            loss = self.criterion(logits, targets)
            
            total_loss += loss.item() * images.size(0)
            all_targets.extend(targets.detach().cpu().numpy())
            all_logits.append(logits.detach().cpu())
            
        avg_loss = total_loss / len(dataloader.dataset)
        logits_tensor = torch.cat(all_logits, dim=0)
        probs_tensor = torch.softmax(logits_tensor, dim=1)
        
        y_true = np.array(all_targets)
        y_prob = probs_tensor.numpy()
        y_pred = np.argmax(y_prob, axis=1)
        
        metrics = calculate_classification_metrics(y_true, y_pred, y_prob)
        return avg_loss, metrics, y_true, y_pred, y_prob
        
    def train(self) -> Dict[str, Any]:
        """
        Executes two-stage transfer learning:
        Stage 1: Frozen backbone (train head only)
        Stage 2: Fine-tune backbone with cosine learning rate decay.
        """
        stage1_epochs = self.config.get("stage1_epochs", 5)
        stage2_epochs = self.config.get("stage2_epochs", 15)
        lr_stage1 = self.config.get("lr_stage1", 1e-3)
        lr_stage2 = self.config.get("lr_stage2", 1e-4)
        weight_decay = self.config.get("weight_decay", 1e-4)
        patience = self.config.get("patience", 7)
        
        print(f"\n[*] Starting Training Experiment -> {self.exp_dir.name}")
        print(f"[*] Device: {self.device} | Stage 1: {stage1_epochs} epochs | Stage 2: {stage2_epochs} epochs")
        
        # STAGE 1: Head Only
        if stage1_epochs > 0:
            print("\n[+] --- STAGE 1: Training Classification Head (Backbone Frozen) ---")
            if hasattr(self.model, "freeze_backbone"):
                self.model.freeze_backbone()
                
            opt_stage1 = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr_stage1,
                weight_decay=weight_decay
            )
            
            for epoch in range(1, stage1_epochs + 1):
                t0 = time.time()
                tr_loss, tr_acc, tr_f1 = self._train_one_epoch(opt_stage1)
                val_loss, val_metrics, _, _, _ = self.evaluate(self.val_loader)
                val_acc = val_metrics["accuracy"]
                val_f1 = val_metrics["macro_f1"]
                dur = time.time() - t0
                
                self.history.append({
                    "stage": 1,
                    "epoch": epoch,
                    "train_loss": tr_loss,
                    "train_acc": tr_acc,
                    "train_macro_f1": tr_f1,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_macro_f1": val_f1,
                    "lr": lr_stage1,
                    "time_sec": dur
                })
                
                print(f"Epoch {epoch:02d}/{stage1_epochs:02d} [{dur:.1f}s] - "
                      f"Tr Loss: {tr_loss:.4f}, Acc: {tr_acc:.3f}, F1: {tr_f1:.3f} | "
                      f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}, F1: {val_f1:.3f}", flush=True)
                      
                if val_f1 > self.best_val_macro_f1:
                    self.best_val_macro_f1 = val_f1
                    self.best_epoch = epoch
                    torch.save(self.model.state_dict(), self.exp_dir / "best_model.pt")
                    
        # STAGE 2: Fine-tuning
        if stage2_epochs > 0:
            print("\n[+] --- STAGE 2: End-to-End Fine-Tuning ---")
            if hasattr(self.model, "unfreeze_upper_blocks"):
                self.model.unfreeze_upper_blocks()
            elif hasattr(self.model, "unfreeze_all"):
                self.model.unfreeze_all()
                
            opt_stage2 = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr_stage2,
                weight_decay=weight_decay
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_stage2, T_max=stage2_epochs, eta_min=1e-6)
            
            patience_counter = 0
            
            for epoch_offset in range(1, stage2_epochs + 1):
                curr_epoch = stage1_epochs + epoch_offset
                curr_lr = scheduler.get_last_lr()[0]
                t0 = time.time()
                
                tr_loss, tr_acc, tr_f1 = self._train_one_epoch(opt_stage2)
                val_loss, val_metrics, _, _, _ = self.evaluate(self.val_loader)
                val_acc = val_metrics["accuracy"]
                val_f1 = val_metrics["macro_f1"]
                scheduler.step()
                dur = time.time() - t0
                
                self.history.append({
                    "stage": 2,
                    "epoch": curr_epoch,
                    "train_loss": tr_loss,
                    "train_acc": tr_acc,
                    "train_macro_f1": tr_f1,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_macro_f1": val_f1,
                    "lr": curr_lr,
                    "time_sec": dur
                })
                
                print(f"Epoch {curr_epoch:02d}/{(stage1_epochs + stage2_epochs):02d} (lr={curr_lr:.1e}) [{dur:.1f}s] - "
                      f"Tr Loss: {tr_loss:.4f}, Acc: {tr_acc:.3f}, F1: {tr_f1:.3f} | "
                      f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}, F1: {val_f1:.3f}", flush=True)
                      
                if val_f1 > self.best_val_macro_f1:
                    self.best_val_macro_f1 = val_f1
                    self.best_epoch = curr_epoch
                    patience_counter = 0
                    torch.save(self.model.state_dict(), self.exp_dir / "best_model.pt")
                    print(f"  --> [*] New Best Validation Macro F1: {val_f1:.4f} (Saved best_model.pt)", flush=True)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  --> [!] Early stopping triggered at epoch {curr_epoch} (patience={patience})", flush=True)
                        break
                        
        # Save training history
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(self.exp_dir / "history.csv", index=False)
        plot_training_history(history_df, save_path=self.exp_dir / "training_curves.png")
        
        # Load best weights for final evaluation
        if (self.exp_dir / "best_model.pt").exists():
            self.model.load_state_dict(torch.load(self.exp_dir / "best_model.pt", map_location=self.device))
            
        # Final Evaluation on Val
        _, final_val_metrics, y_v_true, y_v_pred, y_v_prob = self.evaluate(self.val_loader)
        plot_confusion_matrix(np.array(final_val_metrics["confusion_matrix"]), save_path=self.exp_dir / "val_confusion_matrix.png", title="Validation Confusion Matrix")
        plot_multiclass_roc_curves(y_v_true, y_v_prob, save_path=self.exp_dir / "val_roc_curve.png", title="Validation ROC Curves")
        
        results = {
            "best_epoch": self.best_epoch,
            "best_val_macro_f1": self.best_val_macro_f1,
            "val_metrics": final_val_metrics
        }
        
        # Final Evaluation on Held-Out Test if provided
        if self.test_loader is not None:
            _, test_metrics, y_t_true, y_t_pred, y_t_prob = self.evaluate(self.test_loader)
            plot_confusion_matrix(np.array(test_metrics["confusion_matrix"]), save_path=self.exp_dir / "test_confusion_matrix.png", title="Held-Out Test Confusion Matrix")
            plot_multiclass_roc_curves(y_t_true, y_t_prob, save_path=self.exp_dir / "test_roc_curve.png", title="Held-Out Test ROC Curves")
            
            # Save test predictions table
            test_preds_df = pd.DataFrame({
                "true_class": [CLASS_NAMES[t] for t in y_t_true],
                "pred_class": [CLASS_NAMES[p] for p in y_t_pred],
                "prob_meningioma": y_t_prob[:, 0],
                "prob_glioma": y_t_prob[:, 1],
                "prob_pituitary": y_t_prob[:, 2],
                "max_confidence": np.max(y_t_prob, axis=1),
                "is_correct": (y_t_true == y_t_pred)
            })
            test_preds_df.to_csv(self.exp_dir / "test_predictions.csv", index=False)
            results["test_metrics"] = test_metrics
            
        # Save metrics JSON & config JSON
        with open(self.exp_dir / "metrics.json", "w") as f:
            json.dump(results, f, indent=4)
        with open(self.exp_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=4)
            
        # Save environment info
        with open(self.exp_dir / "environment.txt", "w") as f:
            f.write(f"PyTorch Version: {torch.__version__}\n")
            f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
            if torch.cuda.is_available():
                f.write(f"GPU Device: {torch.cuda.get_device_name(0)}\n")
            f.write(f"Device Used: {self.device}\n")
            
        print(f"\n[OK] Experiment {self.exp_dir.name} completed successfully!")
        print(f"[OK] Artifacts saved in: {self.exp_dir}")
        return results
