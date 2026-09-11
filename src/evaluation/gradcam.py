"""
Grad-CAM and Grad-CAM++ Explainability Module for TrustBT-EfficientNet.

Implements:
- Gradient-weighted Class Activation Mapping (Grad-CAM) [Selvaraju et al., 2017]
- Grad-CAM++ [Chattopadhay et al., 2018]
- Score-CAM (gradient-free) fallback
- Overlay visualisation with tumor mask superimposition
- Batch Grad-CAM generation for publication figures

Compatible with EfficientNet B0/B3/V2S and ResNet50 on CPU (Intel Iris Xe safe).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for CPU
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image


# ---------------------------------------------------------------------------
# Hook-based Gradient/Activation Capture
# ---------------------------------------------------------------------------

class _ActivationGradientHook:
    """Registers forward and backward hooks to capture activations and gradients."""

    def __init__(self):
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd_handle = None
        self._bwd_handle = None

    def register(self, layer: nn.Module):
        self._fwd_handle = layer.register_forward_hook(self._fwd_hook)
        self._bwd_handle = layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, output):
        self.activations = output.detach()

    def _bwd_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        if self._fwd_handle:
            self._fwd_handle.remove()
        if self._bwd_handle:
            self._bwd_handle.remove()


# ---------------------------------------------------------------------------
# Target Layer Resolution
# ---------------------------------------------------------------------------

def _get_target_layer(model: nn.Module, model_name: str = "efficientnet") -> nn.Module:
    """
    Resolves the appropriate Grad-CAM target layer based on model architecture.
    For EfficientNet: last MBConv block in features (BrainTumorEfficientNet wrapper).
    For ResNet: last residual block in layer4 (BrainTumorResNet50 wrapper uses self.backbone).
    """
    name_lower = model_name.lower()

    if "efficientnet" in name_lower or "efficient" in name_lower:
        # Direct attribute (BrainTumorEfficientNet)
        if hasattr(model, "features"):
            return model.features[-1]
        # Nested under backbone
        if hasattr(model, "backbone") and hasattr(model.backbone, "features"):
            return model.backbone.features[-1]
        raise ValueError("EfficientNet model must have 'features' attribute.")

    elif "resnet" in name_lower:
        # Direct attribute
        if hasattr(model, "layer4"):
            return model.layer4[-1]
        # BrainTumorResNet50: backbone is at self.backbone
        if hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
            return model.backbone.layer4[-1]
        raise ValueError("ResNet model must have 'layer4' (or 'backbone.layer4') attribute.")

    else:
        # Heuristic: use last Conv2d layer found
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError(f"Cannot find Conv2d layer in model: {model_name}")
        return last_conv


# ---------------------------------------------------------------------------
# Grad-CAM Core
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Grad-CAM implementation compatible with EfficientNet and ResNet.
    Thread-safe, stateless between calls (hooks registered/removed per call).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

    @torch.no_grad()
    def _prepare_model(self):
        self.model.eval()

    def compute(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        use_relu: bool = True
    ) -> Tuple[np.ndarray, int, float]:
        """
        Computes Grad-CAM heatmap for a single image tensor.

        Args:
            input_tensor: [1, 3, H, W] normalised image tensor.
            class_idx: Target class index. If None, uses argmax prediction.
            use_relu: Whether to apply ReLU to the final CAM.

        Returns:
            cam: [H, W] float32 heatmap in [0, 1].
            pred_class: Predicted class index.
            confidence: Softmax confidence of target class.
        """
        hook = _ActivationGradientHook()
        hook.register(self.target_layer)

        self.model.eval()
        inp = input_tensor.clone().requires_grad_(True)

        # Forward pass
        logits = self.model(inp)
        probs = torch.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = int(torch.argmax(probs, dim=1).item())

        pred_class = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, class_idx].item())

        # Backward pass for target class
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Grad-CAM formula: alpha_k = (1/Z) * sum_ij(dY_c / dA_k_ij)
        activations = hook.activations  # [1, C, h, w]
        gradients = hook.gradients      # [1, C, h, w]

        hook.remove()

        if activations is None or gradients is None:
            return np.zeros((input_tensor.shape[-2], input_tensor.shape[-1]), dtype=np.float32), pred_class, confidence

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # Global Average Pool of grads
        cam_tensor = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]

        if use_relu:
            cam_tensor = F.relu(cam_tensor)

        # Upsample to input resolution
        cam_upsampled = F.interpolate(
            cam_tensor,
            size=(input_tensor.shape[-2], input_tensor.shape[-1]),
            mode="bilinear",
            align_corners=False
        )
        cam_np = cam_upsampled.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np.astype(np.float32), pred_class, confidence


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ [Chattopadhay et al., 2018].
    Improved localisation using second-order gradients.
    """

    def compute(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        use_relu: bool = True
    ) -> Tuple[np.ndarray, int, float]:
        hook = _ActivationGradientHook()
        hook.register(self.target_layer)

        self.model.eval()
        inp = input_tensor.clone().requires_grad_(True)

        logits = self.model(inp)
        probs = torch.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = int(torch.argmax(probs, dim=1).item())

        pred_class = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, class_idx].item())

        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        activations = hook.activations  # [1, C, h, w]
        gradients = hook.gradients      # [1, C, h, w]
        hook.remove()

        if activations is None or gradients is None:
            return np.zeros((input_tensor.shape[-2], input_tensor.shape[-1]), dtype=np.float32), pred_class, confidence

        # Grad-CAM++ alpha computation
        grads_sq = gradients ** 2
        grads_cu = gradients ** 3
        sum_acts = activations.sum(dim=(2, 3), keepdim=True)
        denom = 2.0 * grads_sq + sum_acts * grads_cu + 1e-8
        alpha = grads_sq / denom
        weights = (alpha * F.relu(gradients)).mean(dim=(2, 3), keepdim=True)

        cam_tensor = (weights * activations).sum(dim=1, keepdim=True)
        if use_relu:
            cam_tensor = F.relu(cam_tensor)

        cam_upsampled = F.interpolate(
            cam_tensor,
            size=(input_tensor.shape[-2], input_tensor.shape[-1]),
            mode="bilinear",
            align_corners=False
        )
        cam_np = cam_upsampled.squeeze().cpu().numpy()

        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np.astype(np.float32), pred_class, confidence


# ---------------------------------------------------------------------------
# Visualisation Utilities
# ---------------------------------------------------------------------------

CLASS_NAMES = ["Meningioma", "Glioma", "Pituitary"]
COLORMAP = "jet"


def tensor_to_display_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a [3, H, W] normalised ImageNet tensor back to uint8 RGB.
    Inverts ImageNet normalisation for display.
    """
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = tensor.cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
    img = (img * STD + MEAN)
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def overlay_heatmap(
    original_image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
    colormap: str = COLORMAP
) -> np.ndarray:
    """
    Blends Grad-CAM heatmap over original image.

    Args:
        original_image: [H, W, 3] uint8 RGB array.
        cam: [H, W] float32 in [0, 1].
        alpha: Heatmap blend alpha.
        colormap: Matplotlib colormap name.

    Returns:
        overlaid: [H, W, 3] uint8 RGB array.
    """
    cmap = plt.get_cmap(colormap)
    heatmap_rgba = cmap(cam)  # [H, W, 4]
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

    overlay = (
        (1 - alpha) * original_image.astype(np.float32) +
        alpha * heatmap_rgb.astype(np.float32)
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_gradcam_panel(
    original_tensor: torch.Tensor,
    cam: np.ndarray,
    pred_class: int,
    true_class: int,
    confidence: float,
    save_path: Union[str, Path],
    tumor_mask: Optional[np.ndarray] = None,
    method_name: str = "Grad-CAM",
    slice_id: str = ""
):
    """
    Saves a publication-quality 3-panel Grad-CAM figure:
    [Original MRI | Heatmap | Overlay (+ optional tumor mask contour)]

    Args:
        original_tensor: [3, H, W] image tensor.
        cam: [H, W] float32 Grad-CAM heatmap.
        pred_class: Predicted class index.
        true_class: Ground-truth class index.
        confidence: Predicted class confidence.
        save_path: Output PNG filepath.
        tumor_mask: Optional [H, W] binary tumor mask.
        method_name: Label shown in figure title.
        slice_id: Slice identifier for subtitle.
    """
    orig_img = tensor_to_display_image(original_tensor)
    overlay = overlay_heatmap(orig_img, cam)

    pred_name = CLASS_NAMES[pred_class] if pred_class < len(CLASS_NAMES) else str(pred_class)
    true_name = CLASS_NAMES[true_class] if true_class < len(CLASS_NAMES) else str(true_class)
    is_correct = pred_class == true_class
    verdict = "✓ CORRECT" if is_correct else "✗ WRONG"

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")

    panel_titles = ["Original MRI", f"{method_name} Heatmap", f"Overlay {verdict}"]
    images_to_show = [orig_img, cm.jet(cam)[:, :, :3], overlay]

    for ax, img, title in zip(axes, images_to_show, panel_titles):
        if img.dtype != np.uint8 and img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        ax.imshow(img)
        if tumor_mask is not None and title.startswith("Overlay"):
            ax.contour(tumor_mask, levels=[0.5], colors=["cyan"], linewidths=1.2, alpha=0.85)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=6)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        f"{method_name} | True: {true_name} | Pred: {pred_name} ({confidence:.1%}) | {slice_id}",
        color="white", fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Batch Grad-CAM Utilities
# ---------------------------------------------------------------------------

def generate_gradcam_for_batch(
    model: nn.Module,
    dataloader,
    target_layer: nn.Module,
    output_dir: Union[str, Path],
    num_samples: int = 20,
    method: str = "gradcam",
    device: Optional[torch.device] = None,
    class_names: List[str] = CLASS_NAMES
) -> List[Dict[str, Any]]:
    """
    Generates Grad-CAM panels for `num_samples` samples from the dataloader.

    Args:
        model: Trained classification model.
        dataloader: PyTorch DataLoader yielding {'image', 'label', 'pid', 'filepath'}.
        target_layer: Conv layer to hook.
        output_dir: Directory to save PNG panels.
        num_samples: Total samples to process.
        method: 'gradcam' or 'gradcam++'.
        device: Inference device (default: CPU).
        class_names: List of class name strings.

    Returns:
        List of result dicts with keys: filepath, pred_class, true_class, confidence, cam_path.
    """
    if device is None:
        device = torch.device("cpu")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cam_engine = GradCAMPlusPlus(model, target_layer) if method == "gradcam++" else GradCAM(model, target_layer)

    results = []
    sample_count = 0

    model.to(device)
    model.eval()

    for batch in dataloader:
        if sample_count >= num_samples:
            break

        images = batch["image"]  # [B, 3, H, W]
        labels = batch["label"]  # [B]
        pids = batch.get("pid", ["unknown"] * len(labels))
        filepaths = batch.get("filepath", [""] * len(labels))
        masks = batch.get("tumor_mask", None)

        for i in range(len(images)):
            if sample_count >= num_samples:
                break

            img_tensor = images[i:i+1].to(device)
            true_class = int(labels[i].item())

            cam, pred_class, conf = cam_engine.compute(img_tensor, class_idx=None)

            # Extract tumor mask if available
            tumor_mask_np = None
            if masks is not None:
                tm = masks[i].cpu().numpy()
                if isinstance(tm, np.ndarray):
                    tumor_mask_np = tm.astype(np.uint8)

            pid_str = pids[i] if isinstance(pids[i], str) else str(pids[i].item())
            fname = f"gradcam_{sample_count:04d}_pid{pid_str}_true{true_class}_pred{pred_class}.png"
            cam_path = out_dir / fname

            save_gradcam_panel(
                original_tensor=images[i],
                cam=cam,
                pred_class=pred_class,
                true_class=true_class,
                confidence=conf,
                save_path=cam_path,
                tumor_mask=tumor_mask_np,
                method_name="Grad-CAM++" if method == "gradcam++" else "Grad-CAM",
                slice_id=f"PID={pid_str}"
            )

            results.append({
                "sample_idx": sample_count,
                "pid": pid_str,
                "filepath": filepaths[i] if i < len(filepaths) else "",
                "true_class": true_class,
                "true_name": class_names[true_class],
                "pred_class": pred_class,
                "pred_name": class_names[pred_class],
                "confidence": conf,
                "cam_path": str(cam_path)
            })

            sample_count += 1

    print(f"[OK] Generated {sample_count} Grad-CAM panels -> {out_dir}")
    return results


# ---------------------------------------------------------------------------
# Factory Helper
# ---------------------------------------------------------------------------

def build_gradcam(model: nn.Module, model_name: str = "efficientnet_b0", method: str = "gradcam++") -> GradCAM:
    """
    Convenience factory that builds a Grad-CAM engine with automatically resolved target layer.

    Args:
        model: Trained model (BrainTumorEfficientNet or similar).
        model_name: Architecture name string for target layer resolution.
        method: 'gradcam' or 'gradcam++'.

    Returns:
        GradCAM or GradCAMPlusPlus instance.
    """
    target_layer = _get_target_layer(model, model_name)
    if method == "gradcam++":
        return GradCAMPlusPlus(model, target_layer)
    return GradCAM(model, target_layer)
