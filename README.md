# TrustBT-EfficientNet
Deployed link : https://neurovision-ai-adhis-brain-tumer.streamlit.app/
**Leakage-Aware Explainable Brain Tumor MRI Classification via EfficientNet with Temperature Scaling**

> Final-Year B.Tech AI/ML Research Project | NeuroVision AI Dashboard

---

## 🧠 Overview

TrustBT-EfficientNet is a rigorous, reproducible research project investigating the application of EfficientNet deep learning architectures to brain tumor MRI classification on the **Jun Cheng Figshare dataset** (3,064 slices, 233 patients, 3 classes).

**Research Question:**  
*Can an EfficientNet-based model trained with strict patient-disjoint splits, post-hoc temperature scaling calibration, and Grad-CAM++ explainability achieve clinically trustworthy performance—while remaining fully reproducible on consumer-grade CPU hardware (Intel Iris Xe)?*

---

## 🎯 Key Contributions

| Contribution | Description |
|---|---|
| **Zero-Leakage Splitting** | StratifiedGroupKFold at the patient level; mathematically verified no patient appears in >1 split |
| **Calibration-Aware Evaluation** | Expected Calibration Error (ECE), Brier Score, and Temperature Scaling post-hoc correction |
| **Explainable AI** | Grad-CAM++ with ground-truth tumor mask alignment; 24-panel figure generation |
| **Hardware-Inclusive** | Full pipeline runs on Intel Iris Xe integrated GPU (no NVIDIA required) |
| **Reproducibility** | Fixed seeds, deterministic ops, JSON-manifest-based data management |

---

## 📊 Dataset

**Jun Cheng Brain Tumor MRI Dataset**  
- Source: Figshare DOI: [10.6084/m9.figshare.1512427.v5](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
- 3,064 T1-weighted MRI slices
- 233 unique patients
- 3 tumor classes: **Meningioma** (708), **Glioma** (1,426), **Pituitary** (930)

**Patient-Disjoint 5-Fold Split:**
- ~60% train / ~20% validation / ~20% test (by patient count)
- Zero patient overlap verified mathematically on all 5 folds

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12
- Windows / Linux (tested on Windows 11 + Intel Iris Xe)
- ~4 GB free RAM for inference, ~6 GB for training

### 1. Setup Environment
```bash
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # Linux/Mac
pip install -r requirements.txt
```

### 2. Dataset
The `.mat` files should be placed in `data/raw_mat/`. Download from Figshare:
```bash
python scripts/download_figshare.py
```

### 3. Verify Dataset & Generate Splits
```bash
python scripts/audit_dataset.py
python scripts/create_splits.py
```

### 4. Train Models
```bash
# Baseline (ResNet-50)
python scripts/train.py --config configs/baseline_resnet50.yaml --fold 1

# EfficientNet-B0 (Candidate)
python scripts/train.py --config configs/efficientnet_b0.yaml --fold 1

# EfficientNet-B3 (Main Model)
python scripts/train.py --config configs/efficientnet_b3.yaml --fold 1
```

### 5. XAI + Calibration Evaluation
```bash
python scripts/evaluate_xai.py --exp_dir artifacts/experiments/candidate_efficientnet_b0_fold1
```

### 6. Aggregate Cross-Fold Results
```bash
python scripts/aggregate_results.py
```

### 7. Launch Dashboard
```bash
.venv\Scripts\streamlit run app\dashboard.py
```

---

## 📁 Project Structure

```
TrustBT-EfficientNet Final Year Proj/
├── app/
│   └── dashboard.py           # Streamlit NeuroVision AI dashboard
├── artifacts/
│   ├── splits/                # 5-fold patient-disjoint split manifests
│   ├── experiments/           # Training artifacts (weights, metrics, figures)
│   └── figures/               # Cross-model comparison figures
├── configs/
│   ├── baseline_resnet50.yaml
│   ├── efficientnet_b0.yaml
│   └── efficientnet_b3.yaml
├── data/
│   └── raw_mat/               # 3064 .mat files from Figshare
├── docs/
│   ├── LITERATURE_REVIEW.md
│   ├── RESEARCH_GAP.md
│   ├── EXPERIMENTAL_PROTOCOL.md
│   └── AI_USAGE_LOG.md
├── paper/                     # IEEE-style manuscript drafts
├── scripts/
│   ├── audit_dataset.py       # Dataset integrity audit
│   ├── create_splits.py       # Patient-disjoint fold generation
│   ├── train.py               # Master training script
│   ├── evaluate_xai.py        # Grad-CAM++ + Temperature Scaling evaluation
│   └── aggregate_results.py   # Cross-fold metrics aggregation + LaTeX table
├── src/
│   ├── data/
│   │   ├── loader.py          # .mat Dataset + DataLoader pipeline
│   │   ├── preprocessing.py   # MRI normalization + transforms
│   │   └── splitting.py       # StratifiedGroupKFold + leakage verification
│   ├── evaluation/
│   │   ├── metrics.py         # Macro F1, ECE, Brier, ROC-AUC computation
│   │   ├── gradcam.py         # Grad-CAM and Grad-CAM++ with visualisation
│   │   └── calibration.py     # Temperature Scaling + reliability diagrams
│   ├── models/
│   │   ├── efficientnet.py    # EfficientNet B0/B3/V2S wrapper
│   │   ├── resnet.py          # ResNet-50 baseline wrapper
│   │   └── factory.py         # Model factory
│   └── training/
│       └── trainer.py         # Two-stage transfer learning trainer
└── requirements.txt
```

---

## 📈 Results (Fold 1)

| Model | Test Accuracy | Test Macro F1 | AUC-ROC | ECE |
|---|---|---|---|---|
| ResNet-50 (Baseline) | **0.9054** | **0.8981** | **0.9857** | 0.0363 |
| EfficientNet-B0 | *(training)* | — | — | — |
| EfficientNet-B3 | *(pending)* | — | — | — |

*All results on held-out test fold with zero patient leakage. Primary metric: Macro F1.*

---

## 🔬 Methods

### Architecture
- **EfficientNet-B0**: 5.3M parameters, compound scaling, CPU-feasible
- **EfficientNet-B3**: 12M parameters, improved accuracy
- **ResNet-50** (baseline): Standard benchmark

### Training Protocol
1. **Stage 1** (Head-only): Frozen backbone, AdamW, 5 epochs
2. **Stage 2** (Full fine-tune): Unfrozen upper blocks, Cosine Annealing LR, early stopping on Val Macro F1

### Explainability
- **Grad-CAM++** [Chattopadhay et al., 2018]: Second-order gradients for improved spatial localisation
- Tumor mask overlay comparison for clinical validity assessment

### Calibration
- **Temperature Scaling** [Guo et al., 2017]: Post-hoc single-parameter ECE minimisation via L-BFGS
- Reliability diagram before/after comparison

---

## 📄 Citation

If you use this work, please cite:
```bibtex
@article{trustbt_efficientnet_2026,
  title={TrustBT-EfficientNet: Leakage-Aware Explainable Brain Tumor MRI Classification},
  author={[Author Names]},
  journal={[IEEE Venue]},
  year={2026}
}
```

---

## ⚠️ Disclaimer

This system is developed exclusively for academic research purposes. It is **NOT** validated for clinical use and should **NOT** be used for medical decision-making without appropriate clinical validation.

---

## 📜 License

MIT License — see LICENSE file.

*Developed as a Final-Year B.Tech AI/ML Project. All experiments run on Intel Iris Xe integrated graphics (CPU-mode PyTorch).*
