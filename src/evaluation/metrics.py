"""
Comprehensive Evaluation Metrics Module for TrustBT-EfficientNet.
Calculates Macro F1 (primary metric), Accuracy, Per-class Precision/Recall/Specificity/F1,
Multi-Class One-vs-Rest ROC-AUC, Expected Calibration Error (ECE), Brier Score,
and generates publication-quality plots.
"""

from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
    brier_score_loss
)


CLASS_NAMES = ["Meningioma", "Glioma", "Pituitary"]


def compute_expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Expected Calibration Error (ECE) for multi-class predictions.
    
    Args:
        probabilities: [N, C] predicted softmax probabilities.
        labels: [N] ground-truth class indices (0 to C-1).
        n_bins: Number of confidence bins (default: 10).
        
    Returns:
        ece: Scalar expected calibration error.
        bin_accuracies: Accuracy within each bin.
        bin_confidences: Average confidence within each bin.
        bin_counts: Number of samples in each bin.
    """
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    ece = 0.0
    
    total_samples = len(labels)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        count = np.sum(in_bin)
        bin_counts[i] = count
        
        if count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            bin_accuracies[i] = bin_acc
            bin_confidences[i] = bin_conf
            ece += (count / total_samples) * np.abs(bin_acc - bin_conf)
            
    return float(ece), bin_accuracies, bin_confidences, bin_counts


def compute_brier_score(probabilities: np.ndarray, labels: np.ndarray, num_classes: int = 3) -> float:
    """Computes multi-class Brier Score."""
    one_hot_labels = np.eye(num_classes)[labels]
    brier = np.mean(np.sum((probabilities - one_hot_labels) ** 2, axis=1))
    return float(brier)


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str] = CLASS_NAMES
) -> Dict[str, Any]:
    """
    Calculates comprehensive classification and calibration metrics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    num_classes = len(class_names)
    
    # Overall metrics
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro"))
    
    # Per-class metrics
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(range(num_classes)), zero_division=0)
    
    # Calculate specificity per class from confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificities = []
    for c in range(num_classes):
        tn = np.sum(np.delete(np.delete(cm, c, axis=0), c, axis=1))
        fp = np.sum(np.delete(cm, c, axis=0)[:, c])
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
        
    per_class_stats = {}
    for c, name in enumerate(class_names):
        per_class_stats[name] = {
            "precision": float(p[c]),
            "recall_sensitivity": float(r[c]),
            "specificity": float(specificities[c]),
            "f1_score": float(f[c]),
            "support": int(s[c])
        }
        
    # Multi-class One-vs-Rest ROC-AUC
    try:
        if num_classes == 2:
            roc_auc = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        roc_auc = 0.0
        
    # Calibration
    ece, _, _, _ = compute_expected_calibration_error(y_prob, y_true, n_bins=10)
    brier = compute_brier_score(y_prob, y_true, num_classes=num_classes)
    
    return {
        "macro_f1": macro_f1,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "weighted_f1": weighted_f1,
        "micro_f1": micro_f1,
        "roc_auc_macro_ovr": roc_auc,
        "expected_calibration_error": ece,
        "brier_score": brier,
        "per_class": per_class_stats,
        "confusion_matrix": cm.tolist()
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    save_path: Optional[Path] = None,
    title: str = "Confusion Matrix"
):
    """Plots normalized and annotated confusion matrix."""
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Ground-Truth Class", fontsize=11)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_multiclass_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    save_path: Optional[Path] = None,
    title: str = "Multi-Class One-vs-Rest ROC Curves"
):
    """Plots ROC curves for all classes plus Macro-average."""
    plt.figure(figsize=(8, 6))
    num_classes = len(class_names)
    one_hot = np.eye(num_classes)[y_true]
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(one_hot[:, i], y_prob[:, i])
        class_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                 label=f"{class_names[i]} (AUC = {class_auc:.3f})")
                 
    # Chance line
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance (AUC = 0.500)")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_training_history(
    history_df: pd.DataFrame,
    save_path: Optional[Path] = None,
    title: str = "Training and Validation Convergence"
):
    """Plots loss, accuracy, and Macro F1 trajectories across epochs."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", color="#d62728", lw=2)
    axes[0].set_title("Loss Trajectory", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history_df["epoch"], history_df["train_acc"], label="Train Acc", color="#1f77b4", lw=2)
    axes[1].plot(history_df["epoch"], history_df["val_acc"], label="Val Acc", color="#2ca02c", lw=2)
    axes[1].set_title("Accuracy Trajectory", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Macro F1
    axes[2].plot(history_df["epoch"], history_df["train_macro_f1"], label="Train Macro F1", color="#1f77b4", lw=2)
    axes[2].plot(history_df["epoch"], history_df["val_macro_f1"], label="Val Macro F1", color="#9467bd", lw=2)
    axes[2].set_title("Macro F1 Trajectory (Selection Metric)", fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro F1")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()
