"""
Temperature Scaling Calibration Module for TrustBT-EfficientNet.

Implements post-hoc temperature scaling [Guo et al., 2017] to improve
probability calibration without affecting model accuracy.

Also includes:
- Reliability diagram plotting
- Pre/post calibration ECE comparison
- Calibration report generation

Reference:
    Guo, C. et al. (2017). On calibration of modern neural networks.
    ICML 2017. arXiv:1706.04599
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Temperature Scaling Module
# ---------------------------------------------------------------------------

class TemperatureScaler(nn.Module):
    """
    Learns a single scalar temperature T on the validation set via
    NLL minimisation over pre-computed logits.

    Usage:
        scaler = TemperatureScaler()
        scaler.fit(val_logits_tensor, val_labels_tensor)
        calibrated_probs = scaler.calibrate(test_logits_tensor)
    """

    def __init__(self, init_temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([init_temperature], dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Returns temperature-scaled logits."""
        return logits / self.temperature

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 100,
        lr: float = 0.01,
        verbose: bool = True
    ) -> float:
        """
        Optimises temperature on the validation set using L-BFGS.

        Args:
            logits: [N, C] raw logits (pre-softmax).
            labels: [N] integer class labels.
            max_iter: L-BFGS max iterations.
            lr: Initial learning rate.
            verbose: Print optimisation progress.

        Returns:
            Optimal temperature value.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")

        logits = logits.detach()
        labels = labels.detach()

        init_nll = criterion(logits, labels).item()
        init_temp = float(self.temperature.item())

        if verbose:
            print(f"[TS] Initial temperature: {init_temp:.4f} | Initial NLL: {init_nll:.4f}")

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        final_temp = float(self.temperature.item())
        final_nll = criterion(self.forward(logits), labels).item()

        if verbose:
            print(f"[TS] Optimal temperature: {final_temp:.4f} | Final NLL: {final_nll:.4f}")

        return final_temp

    @torch.no_grad()
    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Returns calibrated softmax probabilities for given logits."""
        scaled_logits = self.forward(logits)
        return torch.softmax(scaled_logits, dim=1)

    def get_temperature(self) -> float:
        return float(self.temperature.item())

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"temperature": self.temperature.item()}, path)
        print(f"[TS] Temperature saved to {path}")

    def load(self, path: Union[str, Path]):
        path = Path(path)
        state = torch.load(path, map_location="cpu")
        self.temperature = nn.Parameter(torch.tensor([state["temperature"]], dtype=torch.float32))
        print(f"[TS] Temperature loaded from {path}: T={self.temperature.item():.4f}")


# ---------------------------------------------------------------------------
# ECE and Reliability Diagrams
# ---------------------------------------------------------------------------

def compute_ece_from_probs(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes ECE from predicted probabilities.

    Args:
        probs: [N, C] predicted softmax probabilities.
        labels: [N] ground-truth class indices.
        n_bins: Number of confidence bins.

    Returns:
        (ece, bin_accuracies, bin_confidences, bin_counts)
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == 0:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences > lo) & (confidences <= hi)
        n = np.sum(in_bin)
        bin_counts[i] = n
        if n > 0:
            bin_accuracies[i] = np.mean(correct[in_bin])
            bin_confidences[i] = np.mean(confidences[in_bin])

    ece = float(np.sum(bin_counts / len(labels) * np.abs(bin_accuracies - bin_confidences)))
    return ece, bin_accuracies, bin_confidences, bin_counts


def plot_reliability_diagram(
    probs_before: np.ndarray,
    labels: np.ndarray,
    probs_after: Optional[np.ndarray] = None,
    save_path: Optional[Union[str, Path]] = None,
    n_bins: int = 10,
    title: str = "Reliability Diagram (Calibration)"
):
    """
    Plots reliability diagram comparing pre/post temperature scaling calibration.

    Args:
        probs_before: [N, C] predicted probabilities before calibration.
        labels: [N] ground-truth labels.
        probs_after: Optional [N, C] predicted probabilities after calibration.
        save_path: Output path for PNG.
        n_bins: Number of confidence bins.
        title: Figure title.
    """
    ece_before, bin_acc_b, bin_conf_b, bin_cnt_b = compute_ece_from_probs(probs_before, labels, n_bins)

    n_panels = 2 if probs_after is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    bin_centers = np.linspace(0, 1, n_bins + 1)[:-1] + 0.5 / n_bins
    bar_w = 0.9 / n_bins

    def _draw_panel(ax, bin_acc, bin_conf, bin_cnt, ece_val, panel_title):
        ax.bar(bin_centers, bin_acc, width=bar_w, alpha=0.7, color="#4c84e0", label="Accuracy", edgecolor="black", linewidth=0.5)
        ax.plot([0, 1], [0, 1], "r--", lw=2, label="Perfect Calibration")
        ax.plot(bin_conf[bin_cnt > 0], bin_acc[bin_cnt > 0], "o-", color="#e84393", lw=2, ms=5, label="Model Calibration")
        ax.set_xlabel("Confidence (Max Softmax Probability)", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"{panel_title}\nECE = {ece_val:.4f}", fontsize=12, fontweight="bold")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    _draw_panel(axes[0], bin_acc_b, bin_conf_b, bin_cnt_b, ece_before, "Before Calibration")

    if probs_after is not None:
        ece_after, bin_acc_a, bin_conf_a, bin_cnt_a = compute_ece_from_probs(probs_after, labels, n_bins)
        _draw_panel(axes[1], bin_acc_a, bin_conf_a, bin_cnt_a, ece_after, "After Temperature Scaling")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Full Calibration Workflow
# ---------------------------------------------------------------------------

def run_temperature_scaling(
    model: nn.Module,
    val_loader: DataLoader,
    device: Optional[torch.device] = None,
    output_dir: Optional[Union[str, Path]] = None,
    max_iter: int = 100,
    verbose: bool = True
) -> Dict[str, object]:
    """
    Full temperature scaling workflow:
    1. Collect validation logits from trained model.
    2. Fit temperature scaler.
    3. Compute ECE before and after.
    4. Save temperature, reliability diagram, and summary.

    Args:
        model: Trained classification model.
        val_loader: DataLoader for calibration set (validation fold).
        device: Inference device.
        output_dir: Directory to save temperature file and plots.
        max_iter: L-BFGS iterations.
        verbose: Print progress.

    Returns:
        Dict with keys: temperature, ece_before, ece_after, probs_before, probs_after, labels.
    """
    if device is None:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    probs_before = torch.softmax(logits_tensor, dim=1).numpy()
    labels_np = labels_tensor.numpy()

    ece_before, _, _, _ = compute_ece_from_probs(probs_before, labels_np)
    if verbose:
        print(f"[TS] ECE before calibration: {ece_before:.4f}")

    # Fit temperature
    scaler = TemperatureScaler()
    optimal_temp = scaler.fit(logits_tensor, labels_tensor, max_iter=max_iter, verbose=verbose)

    # Calibrated probabilities
    probs_after = scaler.calibrate(logits_tensor).numpy()
    ece_after, _, _, _ = compute_ece_from_probs(probs_after, labels_np)

    if verbose:
        print(f"[TS] ECE after calibration: {ece_after:.4f}")
        print(f"[TS] ECE improvement: {ece_before:.4f} -> {ece_after:.4f} (delta = {ece_before - ece_after:.4f})")

    result = {
        "temperature": optimal_temp,
        "ece_before": float(ece_before),
        "ece_after": float(ece_after),
        "ece_improvement": float(ece_before - ece_after),
        "probs_before": probs_before,
        "probs_after": probs_after,
        "labels": labels_np,
        "logits": logits_tensor.numpy()
    }

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save temperature
        scaler.save(out_dir / "temperature_scaler.pt")

        # Reliability diagram
        plot_reliability_diagram(
            probs_before=probs_before,
            labels=labels_np,
            probs_after=probs_after,
            save_path=out_dir / "reliability_diagram.png",
            title="Calibration: Before vs After Temperature Scaling"
        )

        # Summary
        import json
        summary = {
            "optimal_temperature": optimal_temp,
            "ece_before": float(ece_before),
            "ece_after": float(ece_after),
            "ece_improvement": float(ece_before - ece_after),
            "ece_relative_improvement_pct": float((ece_before - ece_after) / max(ece_before, 1e-8) * 100)
        }
        with open(out_dir / "calibration_summary.json", "w") as f:
            json.dump(summary, f, indent=4)

        if verbose:
            print(f"[TS] Calibration artifacts saved to: {out_dir}")

    return result
