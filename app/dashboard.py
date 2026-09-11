"""
TrustBT-EfficientNet — NeuroVision AI Clinical Dashboard
Streamlit Application for Brain Tumor MRI Classification

Features:
- Upload MRI (.mat or .jpg/.png) for real-time inference
- EfficientNet / ResNet50 model selection
- Interactive Grad-CAM++ heatmap visualisation with tumor contour overlay
- Per-class probability gauge charts
- Temperature-scaled calibration probability display
- Experiment performance comparison tables
- Research summary panel

Run with:
    .venv/Scripts/streamlit run app/dashboard.py
"""

import sys
import json
import time
import textwrap
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

# Streamlit import
import streamlit as st
st.set_page_config(
    page_title="NeuroVision AI — Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
from PIL import Image
import io
import scipy.io as sio
import h5py

from src.models.factory import create_model
from src.evaluation.gradcam import build_gradcam, overlay_heatmap, tensor_to_display_image
from src.evaluation.calibration import TemperatureScaler
from src.data.preprocessing import normalize_mri_intensity, mri_to_3channel_tensor, get_mri_transforms
from src.evaluation.metrics import CLASS_NAMES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "artifacts" / "experiments"
SPLITS_DIR = PROJECT_ROOT / "artifacts" / "splits"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

CLASS_COLORS = {
    "Meningioma": "#4c84e0",
    "Glioma": "#e84393",
    "Pituitary": "#2eca7f"
}

# ---------------------------------------------------------------------------
# CSS Styling — Dark, Premium, Medical-Grade
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-card: #1c2128;
    --accent-blue: #4c84e0;
    --accent-pink: #e84393;
    --accent-green: #2eca7f;
    --accent-gold: #f5a623;
    --text-primary: #f0f6fc;
    --text-secondary: #8b949e;
    --border: #30363d;
    --radius: 12px;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}

.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%) !important;
    background-attachment: fixed !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

/* Cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    margin: 8px 0;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    transition: transform 0.2s, box-shadow 0.2s;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(76, 132, 224, 0.2);
}

/* Hero header */
.hero-header {
    background: linear-gradient(135deg, #161b22, #1c2128);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(76, 132, 224, 0.08) 0%, transparent 60%);
    animation: pulse 4s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 0.5; }
    50% { opacity: 1; }
}

.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #4c84e0, #e84393);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    position: relative;
}

.hero-subtitle {
    color: #8b949e;
    font-size: 1rem;
    margin-top: 8px;
    position: relative;
}

/* Probability bars */
.prob-bar-container {
    margin: 6px 0;
}

.prob-bar-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 2px;
}

.prob-bar {
    height: 12px;
    border-radius: 6px;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Status badges */
.badge-correct { background: rgba(46, 202, 127, 0.2); color: #2eca7f; border: 1px solid #2eca7f; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
.badge-wrong { background: rgba(232, 67, 147, 0.2); color: #e84393; border: 1px solid #e84393; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
.badge-info { background: rgba(76, 132, 224, 0.2); color: #4c84e0; border: 1px solid #4c84e0; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }

/* Dividers */
.section-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 20px 0;
}

/* Streamlit elements override */
.stSelectbox label, .stFileUploader label, .stSlider label {
    color: var(--text-secondary) !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #4c84e0, #3a6bc9) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.3s !important;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #5a92ee, #4878d7) !important;
    box-shadow: 0 4px 15px rgba(76, 132, 224, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary) !important;
    border-radius: 8px !important;
    padding: 4px !important;
}

.stTabs [data-baseweb="tab"] {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
}

.stTabs [aria-selected="true"] {
    background: var(--accent-blue) !important;
    color: white !important;
    border-radius: 6px !important;
}
</style>
"""

# ---------------------------------------------------------------------------
# Caching: Model Loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_trained_model(exp_dir_str: str) -> Optional[torch.nn.Module]:
    """Loads best_model.pt from experiment directory."""
    exp_dir = Path(exp_dir_str)
    config_path = exp_dir / "config.json"
    checkpoint_path = exp_dir / "best_model.pt"

    if not checkpoint_path.exists():
        return None

    try:
        with open(config_path) as f:
            cfg = json.load(f)

        model = create_model(
            model_name=cfg.get("model_name", "efficientnet_b0"),
            num_classes=3,
            pretrained=False,
            dropout_rate=cfg.get("dropout_rate", 0.3)
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_temperature_scaler(exp_dir_str: str) -> Optional[TemperatureScaler]:
    """Loads temperature scaler if available."""
    ts_path = Path(exp_dir_str) / "calibration" / "temperature_scaler.pt"
    if ts_path.exists():
        scaler = TemperatureScaler()
        scaler.load(ts_path)
        return scaler
    return None


def get_available_experiments() -> Dict[str, str]:
    """Returns dict of {display_name: exp_dir_path} for trained experiments."""
    experiments = {}
    if EXPERIMENTS_DIR.exists():
        for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
            if exp_dir.is_dir() and (exp_dir / "best_model.pt").exists():
                # Load config for display name
                config_path = exp_dir / "config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        cfg = json.load(f)
                    model_name = cfg.get("model_name", exp_dir.name)
                    fold = cfg.get("fold", "?")
                    display = f"{model_name} | Fold {fold}"
                else:
                    display = exp_dir.name
                experiments[display] = str(exp_dir)
    return experiments


# ---------------------------------------------------------------------------
# Image Loading from .mat or PIL-compatible formats
# ---------------------------------------------------------------------------

def load_image_from_upload(uploaded_file) -> Optional[Dict[str, Any]]:
    """Loads an uploaded file (.mat, .jpg, .png) and returns image dict."""
    filename = uploaded_file.name.lower()

    try:
        if filename.endswith(".mat"):
            # Save to temp bytes and load
            mat_bytes = uploaded_file.read()
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
                tmp.write(mat_bytes)
                tmp_path = tmp.name

            try:
                with h5py.File(tmp_path, "r") as f:
                    cjdata = f["cjdata"]
                    label = int(np.array(cjdata["label"]).item())
                    pid_data = np.array(cjdata["PID"])
                    pid = "".join(chr(c[0]) for c in pid_data) if pid_data.ndim > 1 else str(pid_data)
                    image = np.array(cjdata["image"]).T.astype(np.float32)
                    tumor_mask = np.array(cjdata["tumorMask"]).T.astype(np.uint8)
            except Exception:
                mat_dict = sio.loadmat(tmp_path)
                cjdata = mat_dict["cjdata"]
                label = int(cjdata["label"][0, 0][0, 0])
                pid = str(cjdata["PID"][0, 0][0])
                image = cjdata["image"][0, 0].astype(np.float32)
                tumor_mask = cjdata["tumorMask"][0, 0].astype(np.uint8)
            finally:
                os.unlink(tmp_path)

            return {"image": image, "label": label, "pid": str(pid).strip(),
                    "tumor_mask": tumor_mask, "has_ground_truth": True}

        else:
            # PNG/JPG — check for color and convert to grayscale float
            pil_img = Image.open(uploaded_file)
            is_color = False
            if pil_img.mode in ("RGB", "RGBA"):
                rgb_arr = np.array(pil_img.convert("RGB"), dtype=np.float32)
                channel_std = float(np.std(rgb_arr, axis=-1).mean())
                if channel_std > 8.0:
                    is_color = True

            pil_gray = pil_img.convert("L")
            image = np.array(pil_gray, dtype=np.float32)
            return {"image": image, "label": None, "pid": "uploaded",
                    "tumor_mask": None, "has_ground_truth": False, "is_color": is_color}

    except Exception as e:
        st.error(f"Failed to load image: {e}")
        return None


def load_image_input(source) -> Optional[Dict[str, Any]]:
    """Loads image from an uploaded file object or a local file Path."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        filename = path.name.lower()
        if filename.endswith(".mat"):
            try:
                with h5py.File(path, "r") as f:
                    cjdata = f["cjdata"]
                    label = int(np.array(cjdata["label"]).item())
                    pid_data = np.array(cjdata["PID"])
                    pid = "".join(chr(c[0]) for c in pid_data) if pid_data.ndim > 1 else str(pid_data)
                    image = np.array(cjdata["image"]).T.astype(np.float32)
                    tumor_mask = np.array(cjdata["tumorMask"]).T.astype(np.uint8)
            except Exception:
                mat_dict = sio.loadmat(str(path))
                cjdata = mat_dict["cjdata"]
                label = int(cjdata["label"][0, 0][0, 0])
                pid = str(cjdata["PID"][0, 0][0])
                image = cjdata["image"][0, 0].astype(np.float32)
                tumor_mask = cjdata["tumorMask"][0, 0].astype(np.uint8)
            return {"image": image, "label": label, "pid": str(pid).strip(),
                    "tumor_mask": tumor_mask, "has_ground_truth": True}
        else:
            pil_img = Image.open(path)
            is_color = False
            if pil_img.mode in ("RGB", "RGBA"):
                rgb_arr = np.array(pil_img.convert("RGB"), dtype=np.float32)
                channel_std = float(np.std(rgb_arr, axis=-1).mean())
                if channel_std > 8.0:
                    is_color = True
            pil_gray = pil_img.convert("L")
            image = np.array(pil_gray, dtype=np.float32)
            return {"image": image, "label": None, "pid": "sample",
                    "tumor_mask": None, "has_ground_truth": False, "is_color": is_color}
    else:
        return load_image_from_upload(source)


def is_valid_mri(img_data: Dict[str, Any]) -> tuple[bool, str]:
    """
    Multi-stage Out-of-Distribution (OOD) validity check for Brain MRI.
    Verifies:
    1. Grayscale acquisition (rejects natural color photos like clocks, scenes, etc.)
    2. Dark background proportion (air surrounding skull in scanner)
    3. Ambient dark corners (air margin around scan)
    4. Contrast and dimension validity
    Returns (is_valid, reason).
    """
    if img_data.get("is_color", False):
        return False, (
            "Input image contains natural chromatic colors. Brain MRI scans are strictly "
            "single-channel grayscale medical acquisitions. Natural objects (clocks, photos, etc.) "
            "cannot be processed."
        )

    image = img_data.get("image")
    if image is None or image.size == 0 or image.ndim < 2:
        return False, "Invalid image dimensions."

    img_max = float(image.max())
    if img_max <= 0:
        return False, "Image is completely blank / black."

    # Scale to 0-255 for standard threshold evaluation
    scaled = (image / img_max) * 255.0
    h, w = scaled.shape[:2]

    # Check 1: Dark background ratio (air surrounding skull)
    dark_pixels = np.sum(scaled < 35)
    dark_ratio = dark_pixels / scaled.size
    if dark_ratio < 0.15:
        return False, (
            f"Image lacks the characteristic dark ambient background of a brain MRI scanner field "
            f"(dark ratio: {dark_ratio:.1%}, expected > 15%)."
        )

    # Check 2: All 4 corners should be dark background air
    ch, cw = max(2, int(h * 0.08)), max(2, int(w * 0.08))
    c_tl = scaled[:ch, :cw].mean()
    c_tr = scaled[:ch, -cw:].mean()
    c_bl = scaled[-ch:, :cw].mean()
    c_br = scaled[-ch:, -cw:].mean()
    avg_corner = (c_tl + c_tr + c_bl + c_br) / 4.0
    if avg_corner > 65.0:
        return False, (
            f"Image corners contain high-intensity non-air content (avg corner intensity: {avg_corner:.1f}/255, expected < 65). "
            f"Valid brain MRI slices have dark ambient margins at all corners."
        )

    return True, ""


def preprocess_image(image: np.ndarray, target_size=(224, 224)) -> torch.Tensor:
    """Prepares MRI numpy array for model inference."""
    from torchvision.transforms import v2
    norm = normalize_mri_intensity(image)
    tensor_3ch = mri_to_3channel_tensor(norm)  # [3, H, W]
    transform = get_mri_transforms(target_size=target_size, is_training=False)
    return transform(tensor_3ch)


# ---------------------------------------------------------------------------
# Inference + Grad-CAM
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, input_tensor: torch.Tensor) -> Dict[str, Any]:
    """Runs model inference and returns logits + probabilities."""
    model.eval()
    inp = input_tensor.unsqueeze(0)
    logits = model(inp)
    probs = F.softmax(logits, dim=1).squeeze(0)
    pred_class = int(torch.argmax(probs).item())
    return {
        "logits": logits.squeeze(0),
        "probs": probs.cpu().numpy(),
        "pred_class": pred_class,
        "pred_name": CLASS_NAMES[pred_class],
        "confidence": float(probs[pred_class].item())
    }


def run_gradcam(model, input_tensor: torch.Tensor, model_name: str, class_idx: Optional[int] = None):
    """Runs Grad-CAM++ and returns CAM heatmap."""
    try:
        cam_engine = build_gradcam(model, model_name, method="gradcam++")
        cam, pred_class, conf = cam_engine.compute(
            input_tensor.unsqueeze(0),
            class_idx=class_idx
        )
        return cam, pred_class, conf
    except Exception as e:
        st.warning(f"Grad-CAM generation failed: {e}")
        return None, None, None


# ---------------------------------------------------------------------------
# Plot Helpers
# ---------------------------------------------------------------------------

def plot_probability_bars(probs: np.ndarray) -> str:
    """Renders probability bars as HTML."""
    html = ""
    for i, (cls, prob) in enumerate(zip(CLASS_NAMES, probs)):
        color = list(CLASS_COLORS.values())[i]
        pct = prob * 100
        html += f"""
        <div class="prob-bar-container">
            <div class="prob-bar-label">{cls}: {pct:.1f}%</div>
            <div style="background: #30363d; border-radius: 6px; height: 14px; overflow: hidden;">
                <div class="prob-bar" style="width: {pct:.1f}%; background: linear-gradient(90deg, {color}, {color}88); height: 100%;"></div>
            </div>
        </div>
        """
    return html


def create_gradcam_figure(
    orig_tensor: torch.Tensor,
    cam: np.ndarray,
    tumor_mask: Optional[np.ndarray] = None
) -> plt.Figure:
    """Creates a 3-panel Grad-CAM figure for Streamlit display."""
    orig_img = tensor_to_display_image(orig_tensor)
    overlay = overlay_heatmap(orig_img, cam, alpha=0.45)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.patch.set_facecolor("#1c2128")

    panels = [
        ("Original MRI", orig_img),
        ("Grad-CAM++ Heatmap", mplcm.jet(cam)[:, :, :3]),
        ("Overlay" + (" + Tumor Mask" if tumor_mask is not None else ""), overlay)
    ]

    for ax, (title, img) in zip(axes, panels):
        if img.dtype != np.uint8 and img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        ax.imshow(img)
        if tumor_mask is not None and "Overlay" in title:
            ax.contour(tumor_mask, levels=[0.5], colors=["cyan"], linewidths=1.5, alpha=0.9)
        ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=6)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor("#1c2128")

    plt.tight_layout(pad=0.5)
    return fig


def load_experiment_metrics(exp_dir: Path) -> Optional[Dict]:
    metrics_path = exp_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Hero Header
    st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">🧠 NeuroVision AI</h1>
        <p class="hero-subtitle">
            Leakage-Aware Explainable Brain Tumor MRI Classification &nbsp;·&nbsp;
            EfficientNet + Grad-CAM++ + Temperature Scaling &nbsp;·&nbsp;
            Final Year B.Tech AI/ML Research Project
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Model Configuration")

        available_exps = get_available_experiments()

        if not available_exps:
            st.error("No trained models found in artifacts/experiments/")
            st.info("Run: `python scripts/train.py --config configs/efficientnet_b0.yaml`")
            selected_exp_name = None
            selected_exp_dir = None
        else:
            selected_exp_name = st.selectbox(
                "Select Trained Model",
                options=list(available_exps.keys()),
                index=0
            )
            selected_exp_dir = available_exps[selected_exp_name]

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("### 🔬 Grad-CAM Settings")
        gradcam_class = st.selectbox(
            "Explain for class",
            options=["Predicted (auto)"] + CLASS_NAMES,
            index=0
        )
        show_tumor_mask = st.checkbox("Show tumor mask overlay", value=True)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("### 📂 Upload MRI Slice")
        uploaded_file = st.file_uploader(
            "Drop .mat, .jpg or .png",
            type=["mat", "jpg", "jpeg", "png"]
        )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("""
        <div style="color: #8b949e; font-size: 0.78rem;">
        <b>Dataset:</b> Jun Cheng Figshare (3064 slices, 233 patients)<br>
        <b>Split:</b> 5-fold patient-disjoint (StratifiedGroupKFold)<br>
        <b>XAI:</b> Grad-CAM++ [Chattopadhay, 2018]<br>
        <b>Calib:</b> Temperature Scaling [Guo, 2017]<br>
        <b>Hardware:</b> Intel Iris Xe (CPU-only)
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Main Tabs
    # -----------------------------------------------------------------------

    tab_inference, tab_performance, tab_research = st.tabs([
        "🔍 Live Inference", "📊 Experiment Results", "📄 Research Summary"
    ])

    # =======================================================================
    # TAB 1: Live Inference
    # =======================================================================
    with tab_inference:

        if selected_exp_dir is None:
            st.warning("No trained model available. Please train a model first.")
        elif uploaded_file is None:
            st.info("Upload a brain tumor MRI slice (.mat from the Figshare dataset, or any .jpg/.png) to begin inference.")

            # Demo instruction panel
            col1, col2, col3 = st.columns(3)
            for col, (cls, color, desc) in zip(
                [col1, col2, col3],
                [
                    ("Meningioma", "#4c84e0", "Arises from meninges. Typically benign, slow-growing."),
                    ("Glioma", "#e84393", "Originates in glial cells. Most common malignant brain tumor."),
                    ("Pituitary", "#2eca7f", "Pituitary gland tumor. Often affects hormone regulation.")
                ]
            ):
                with col:
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 3px solid {color};">
                        <h4 style="color: {color}; margin: 0;">{cls}</h4>
                        <p style="color: #8b949e; font-size: 0.85rem; margin-top: 8px;">{desc}</p>
                    </div>
                    """, unsafe_allow_html=True)

        else:
            # Load model
            with st.spinner("Loading model..."):
                model = load_trained_model(selected_exp_dir)
                temperature_scaler = load_temperature_scaler(selected_exp_dir)

            if model is None:
                st.error("Failed to load model. Check if best_model.pt exists.")
            else:
                # Load image
                img_data = load_image_from_upload(uploaded_file)
                if img_data is None:
                    st.stop()

                # OOD Detection Check
                is_valid, ood_reason = is_valid_mri(img_data)
                if not is_valid:
                    st.error("🚫 Invalid Input / Out-of-Distribution Detected")
                    st.warning(ood_reason)
                    st.info(
                        "ℹ️ **Clinical Safety Notice:** The system only processes valid T1-weighted Brain MRI scans. "
                        "Non-medical inputs (e.g., clocks, photographs, non-cranial images) are rejected to prevent hallucinated predictions."
                    )
                    st.stop()

                # Get config
                config_path = Path(selected_exp_dir) / "config.json"
                with open(config_path) as f:
                    cfg = json.load(f)
                model_name = cfg.get("model_name", "efficientnet_b0")
                target_size = tuple(cfg.get("target_size", [224, 224]))

                # Preprocess
                input_tensor = preprocess_image(img_data["image"], target_size=target_size)

                # Inference
                with st.spinner("Running inference..."):
                    t0 = time.time()
                    result = run_inference(model, input_tensor)
                    inference_ms = (time.time() - t0) * 1000

                probs = result["probs"]
                pred_class = result["pred_class"]
                pred_name = result["pred_name"]
                confidence = result["confidence"]

                # Apply temperature scaling if available
                if temperature_scaler is not None:
                    logits_tensor = result["logits"].unsqueeze(0)
                    cal_probs = temperature_scaler.calibrate(logits_tensor).squeeze(0).numpy()
                    temp = temperature_scaler.get_temperature()
                else:
                    cal_probs = probs
                    temp = 1.0

                # Grad-CAM class selection
                cam_class_idx = None
                if gradcam_class != "Predicted (auto)":
                    cam_class_idx = CLASS_NAMES.index(gradcam_class)

                with st.spinner("Generating Grad-CAM++..."):
                    cam, cam_pred, cam_conf = run_gradcam(model, input_tensor, model_name, cam_class_idx)

                # ---------------------------------------------------------------
                # Results Layout
                # ---------------------------------------------------------------
                col_info, col_probs, col_cal = st.columns([1.2, 1, 1])

                with col_info:
                    pred_color = list(CLASS_COLORS.values())[pred_class]
                    gt_text = ""
                    if img_data["has_ground_truth"] and img_data["label"] is not None:
                        from src.data.loader import LABEL_MAP
                        true_idx = LABEL_MAP.get(img_data["label"], 0)
                        true_name = CLASS_NAMES[true_idx]
                        is_correct = pred_class == true_idx
                        badge = "badge-correct" if is_correct else "badge-wrong"
                        verdict = "✓ CORRECT" if is_correct else "✗ WRONG"
                        gt_text = f'<p style="color: #8b949e;">Ground Truth: <b style="color: {list(CLASS_COLORS.values())[true_idx]};">{true_name}</b> &nbsp; <span class="{badge}">{verdict}</span></p>'
                        if img_data["pid"]:
                            gt_text += f'<p style="color: #8b949e; font-size: 0.85rem;">Patient ID: {img_data["pid"]}</p>'

                    conf_warning = ""
                    if confidence < 0.60:
                        conf_warning = (
                            '<div style="margin-top: 8px; padding: 6px 10px; background: rgba(245, 166, 35, 0.15); '
                            'border: 1px solid #f5a623; border-radius: 6px; color: #f5a623; font-size: 0.8rem;">'
                            '⚠️ <b>Low Confidence / Ambiguous Scan:</b> Prediction confidence is below 60%. '
                            'Clinical verification required.</div>'
                        )

                    pred_html = (
                        f'<div class="metric-card" style="border-left: 4px solid {pred_color};">'
                        f'<p style="color: #8b949e; margin: 0; font-size: 0.85rem;">PREDICTION</p>'
                        f'<h2 style="color: {pred_color}; margin: 4px 0;">{pred_name}</h2>'
                        f'<p style="margin: 0; font-size: 1.1rem; font-weight: 700;">{confidence:.1%} confidence</p>'
                        + gt_text
                        + conf_warning
                        + f'<p style="color: #8b949e; font-size: 0.8rem; margin-top: 8px;">⚡ Inference: {inference_ms:.1f} ms</p>'
                        + '</div>'
                    )
                    st.markdown(pred_html, unsafe_allow_html=True)

                with col_probs:
                    raw_card_header = textwrap.dedent("""
                    <div class="metric-card">
                        <p style="color: #8b949e; margin: 0 0 12px; font-size: 0.85rem;">RAW PROBABILITIES</p>
                    """).strip()
                    st.markdown(raw_card_header, unsafe_allow_html=True)
                    st.markdown(plot_probability_bars(probs), unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with col_cal:
                    t_badge = f'<span class="badge-info">T = {temp:.3f}</span>' if temperature_scaler is not None else ""
                    cal_card_header = textwrap.dedent(f"""
                    <div class="metric-card">
                        <p style="color: #8b949e; margin: 0 0 12px; font-size: 0.85rem;">CALIBRATED PROBABILITIES {t_badge}</p>
                    """).strip()
                    st.markdown(cal_card_header, unsafe_allow_html=True)
                    st.markdown(plot_probability_bars(cal_probs), unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                # Grad-CAM panel
                if cam is not None:
                    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
                    st.markdown("#### 🔥 Grad-CAM++ Explainability")

                    tumor_mask_np = None
                    if show_tumor_mask and img_data.get("tumor_mask") is not None:
                        from PIL import Image as PILImage
                        import torchvision.transforms.functional as TF
                        mask = img_data["tumor_mask"]
                        mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
                        mask_resized = np.array(mask_pil.resize((target_size[1], target_size[0]),
                                                                PILImage.NEAREST)) > 127
                        tumor_mask_np = mask_resized.astype(np.uint8)

                    fig = create_gradcam_figure(input_tensor, cam, tumor_mask=tumor_mask_np)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    # Hotspot interpretation
                    cam_max_loc = np.unravel_index(np.argmax(cam), cam.shape)
                    h, w = cam.shape
                    quadrant = ("superior" if cam_max_loc[0] < h//2 else "inferior") + \
                               ("-left" if cam_max_loc[1] < w//2 else "-right")
                    coverage_pct = float((cam > 0.5).mean() * 100)

                    analysis_html = textwrap.dedent(f"""
                    <div class="metric-card" style="border-left: 3px solid #f5a623;">
                        <p style="color: #8b949e; margin: 0; font-size: 0.85rem;">GRAD-CAM++ ANALYSIS</p>
                        <p style="margin: 8px 0; font-size: 0.95rem;">
                            Peak activation region: <b style="color: #f5a623;">{quadrant}</b> quadrant &nbsp;·&nbsp;
                            High-attention coverage: <b style="color: #f5a623;">{coverage_pct:.1f}%</b> of image &nbsp;·&nbsp;
                            Method: <b>Grad-CAM++</b> [Chattopadhay et al., 2018]
                        </p>
                        <p style="color: #8b949e; font-size: 0.8rem; margin: 0;">
                            ⚠️ <i>Grad-CAM heatmaps are for research/educational purposes only. Not a clinical diagnostic tool.</i>
                        </p>
                    </div>
                    """).strip()
                    st.markdown(analysis_html, unsafe_allow_html=True)

    # =======================================================================
    # TAB 2: Experiment Results
    # =======================================================================
    with tab_performance:
        st.markdown("### 📊 Training Experiment Results")

        if not EXPERIMENTS_DIR.exists() or not any(EXPERIMENTS_DIR.iterdir()):
            st.warning("No experiments found. Run training scripts first.")
        else:
            rows = []
            for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
                if not exp_dir.is_dir():
                    continue
                metrics = load_experiment_metrics(exp_dir)
                if metrics is None:
                    continue

                config_path = exp_dir / "config.json"
                cfg = {}
                if config_path.exists():
                    with open(config_path) as f:
                        cfg = json.load(f)

                m = metrics.get("test_metrics", metrics.get("val_metrics", {}))
                if not m:
                    continue

                rows.append({
                    "Experiment": exp_dir.name,
                    "Model": cfg.get("model_name", "unknown"),
                    "Fold": cfg.get("fold", "?"),
                    "Best Epoch": metrics.get("best_epoch", -1),
                    "Test Accuracy": f"{m.get('accuracy', 0):.4f}",
                    "Test Macro F1": f"{m.get('macro_f1', 0):.4f}",
                    "AUC-ROC": f"{m.get('roc_auc_macro_ovr', 0):.4f}",
                    "ECE": f"{m.get('expected_calibration_error', 0):.4f}",
                    "Brier": f"{m.get('brier_score', 0):.4f}",
                })

            if rows:
                import pandas as pd
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, height=300)

                # Per-class detail for selected experiment
                st.markdown("#### Per-Class Breakdown")
                exp_names = [r["Experiment"] for r in rows]
                selected_exp = st.selectbox("Select experiment for per-class detail:", exp_names)
                selected_dir = EXPERIMENTS_DIR / selected_exp
                exp_metrics = load_experiment_metrics(selected_dir)

                if exp_metrics:
                    m = exp_metrics.get("test_metrics", exp_metrics.get("val_metrics", {}))
                    per_class = m.get("per_class", {})
                    pc_rows = []
                    for cls_name, cls_vals in per_class.items():
                        pc_rows.append({
                            "Class": cls_name,
                            "Precision": f"{cls_vals.get('precision', 0):.4f}",
                            "Recall (Sens.)": f"{cls_vals.get('recall_sensitivity', 0):.4f}",
                            "Specificity": f"{cls_vals.get('specificity', 0):.4f}",
                            "F1 Score": f"{cls_vals.get('f1_score', 0):.4f}",
                            "Support": cls_vals.get("support", 0)
                        })
                    pc_df = pd.DataFrame(pc_rows)
                    st.dataframe(pc_df, use_container_width=True)

                    # Training curves
                    history_path = selected_dir / "history.csv"
                    if history_path.exists():
                        st.markdown("#### Training Curves")
                        hist_df = pd.read_csv(history_path)
                        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
                        fig2.patch.set_facecolor("#1c2128")
                        for ax in axes2:
                            ax.set_facecolor("#1c2128")
                            ax.tick_params(colors="white")
                            ax.xaxis.label.set_color("white")
                            ax.yaxis.label.set_color("white")
                            ax.title.set_color("white")
                            for spine in ax.spines.values():
                                spine.set_edgecolor("#30363d")

                        axes2[0].plot(hist_df["epoch"], hist_df["train_loss"], color="#4c84e0", lw=2, label="Train")
                        axes2[0].plot(hist_df["epoch"], hist_df["val_loss"], color="#e84393", lw=2, label="Val")
                        axes2[0].set_title("Loss", color="white", fontweight="bold")
                        axes2[0].legend(facecolor="#1c2128", labelcolor="white")
                        axes2[0].grid(True, alpha=0.2)

                        axes2[1].plot(hist_df["epoch"], hist_df["train_acc"], color="#4c84e0", lw=2, label="Train")
                        axes2[1].plot(hist_df["epoch"], hist_df["val_acc"], color="#2eca7f", lw=2, label="Val")
                        axes2[1].set_title("Accuracy", color="white", fontweight="bold")
                        axes2[1].legend(facecolor="#1c2128", labelcolor="white")
                        axes2[1].grid(True, alpha=0.2)

                        axes2[2].plot(hist_df["epoch"], hist_df["train_macro_f1"], color="#4c84e0", lw=2, label="Train")
                        axes2[2].plot(hist_df["epoch"], hist_df["val_macro_f1"], color="#9467bd", lw=2, label="Val")
                        axes2[2].set_title("Macro F1 (Primary Metric)", color="white", fontweight="bold")
                        axes2[2].legend(facecolor="#1c2128", labelcolor="white")
                        axes2[2].grid(True, alpha=0.2)

                        plt.tight_layout()
                        st.pyplot(fig2, use_container_width=True)
                        plt.close(fig2)

                    # Display confusion matrix image if exists
                    cm_path = selected_dir / "test_confusion_matrix.png"
                    roc_path = selected_dir / "test_roc_curve.png"
                    if cm_path.exists() or roc_path.exists():
                        c1, c2 = st.columns(2)
                        with c1:
                            if cm_path.exists():
                                st.image(str(cm_path), caption="Test Confusion Matrix", use_container_width=True)
                        with c2:
                            if roc_path.exists():
                                st.image(str(roc_path), caption="Test ROC Curves", use_container_width=True)

    # =======================================================================
    # TAB 3: Research Summary
    # =======================================================================
    with tab_research:
        st.markdown("### 📄 TrustBT-EfficientNet — Research Overview")

        col_l, col_r = st.columns([1, 1])

        with col_l:
            st.markdown("""
            <div class="metric-card">
                <h4 style="color: #4c84e0;">🎯 Research Question</h4>
                <p><i>"Can an EfficientNet-based model trained with strict patient-disjoint splits,
                post-hoc calibration, and Grad-CAM explainability achieve clinically trustworthy
                performance on the Jun Cheng Brain Tumor Figshare dataset while remaining
                reproducible on consumer hardware?"</i></p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="metric-card">
                <h4 style="color: #2eca7f;">📊 Dataset</h4>
                <ul style="color: #8b949e;">
                    <li><b>Source:</b> Jun Cheng, Figshare DOI: 10.6084/m9.figshare.1512427.v5</li>
                    <li><b>Size:</b> 3,064 T1-weighted MRI slices, 233 patients</li>
                    <li><b>Classes:</b> Meningioma (708), Glioma (1,426), Pituitary (930)</li>
                    <li><b>Splitting:</b> 5-fold StratifiedGroupKFold (patient-disjoint)</li>
                    <li><b>Leakage:</b> Mathematically verified zero patient overlap</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col_r:
            st.markdown("""
            <div class="metric-card">
                <h4 style="color: #e84393;">🔬 Novel Contributions</h4>
                <ol style="color: #8b949e;">
                    <li><b>Leakage-free benchmark:</b> StratifiedGroupKFold enforcing patient isolation</li>
                    <li><b>Calibration-aware evaluation:</b> ECE + Brier Score + Temperature Scaling</li>
                    <li><b>Explainable AI:</b> Grad-CAM++ with ground-truth tumor mask alignment</li>
                    <li><b>Hardware-inclusive:</b> Full pipeline on Intel Iris Xe (no GPU required)</li>
                    <li><b>Reproducibility:</b> Fixed seeds, deterministic ops, manifest-based splits</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="metric-card">
                <h4 style="color: #f5a623;">⚙️ Technical Stack</h4>
                <ul style="color: #8b949e;">
                    <li><b>Models:</b> EfficientNet-B0, EfficientNet-B3, ResNet-50 (baseline)</li>
                    <li><b>Training:</b> Two-stage transfer learning (head → full fine-tune)</li>
                    <li><b>Scheduler:</b> Cosine Annealing LR with Early Stopping (Val Macro F1)</li>
                    <li><b>XAI:</b> Grad-CAM++ [Chattopadhay et al., 2018]</li>
                    <li><b>Calibration:</b> Temperature Scaling [Guo et al., 2017]</li>
                    <li><b>Framework:</b> PyTorch 2.2+ | Python 3.12 | Streamlit</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Baseline results
        st.markdown("---")
        st.markdown("#### 📈 Current Results Summary")

        baseline_dir = EXPERIMENTS_DIR / "baseline_resnet50_fold1"
        if baseline_dir.exists():
            bm = load_experiment_metrics(baseline_dir)
            if bm:
                tm = bm.get("test_metrics", {})
                cols = st.columns(5)
                metrics_to_show = [
                    ("Accuracy", tm.get("accuracy", 0), "#4c84e0"),
                    ("Macro F1", tm.get("macro_f1", 0), "#e84393"),
                    ("AUC-ROC", tm.get("roc_auc_macro_ovr", 0), "#2eca7f"),
                    ("ECE", tm.get("expected_calibration_error", 0), "#f5a623"),
                    ("Brier", tm.get("brier_score", 0), "#9467bd"),
                ]
                for col, (name, val, color) in zip(cols, metrics_to_show):
                    with col:
                        st.markdown(f"""
                        <div class="metric-card" style="text-align: center; border-top: 3px solid {color};">
                            <p style="color: #8b949e; margin: 0; font-size: 0.8rem;">{name}</p>
                            <h3 style="color: {color}; margin: 4px 0;">{val:.4f}</h3>
                            <p style="color: #8b949e; margin: 0; font-size: 0.75rem;">ResNet-50 Fold 1</p>
                        </div>
                        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top: 24px; padding: 16px; background: #1c2128; border-radius: 8px; border: 1px solid #30363d;">
            <p style="color: #8b949e; font-size: 0.82rem; margin: 0;">
                ⚠️ <b>Disclaimer:</b> This system is developed exclusively for academic research purposes.
                It is NOT validated for clinical use and should NOT be used for medical decision-making.
                All predictions are probabilistic estimates for research evaluation only.
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
