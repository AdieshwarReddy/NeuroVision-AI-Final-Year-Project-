# Experimental and Evaluation Protocol: TrustBT-EfficientNet

## 1. Experimental Overview
This document specifies the exact, reproducible experimental protocol for data preparation, model training, evaluation, calibration, explainability, robustness, and statistical testing for the TrustBT-EfficientNet research framework.

---

## 2. Dataset & Partitioning Protocol

### 2.1 Primary Dataset
- **Source**: Jun Cheng Figshare Brain Tumor Dataset (DOI: [10.6084/m9.figshare.1512427](https://doi.org/10.6084/m9.figshare.1512427))
- **Total Slices**: 3,064 T1-weighted contrast-enhanced (T1-CE) MRI 2D axial/coronal/sagittal slices.
- **Patients**: 233 unique patients (`PID`).
- **Classes**:
  - Class 0 (Meningioma): 708 slices
  - Class 1 (Glioma): 1,426 slices
  - Class 2 (Pituitary Tumor): 930 slices

### 2.2 Patient-Disjoint Cross-Validation Strategy
- **Partitioning Algorithm**: 5-Fold `StratifiedGroupKFold` grouped strictly by Patient ID (`PID`).
- In each iteration $k \in \{1, 2, 3, 4, 5\}$:
  - 3 folds $\rightarrow$ Training ($\approx 60\%$ patients)
  - 1 fold $\rightarrow$ Validation ($\approx 20\%$ patients, used for early stopping, temperature scaling fit, and XAI threshold tuning)
  - 1 fold $\rightarrow$ Test ($\approx 20\%$ patients, held-out untouched evaluation)
- **Mandatory Disjoint Assertions**:
  $$\text{Train}_{\text{PID}} \cap \text{Val}_{\text{PID}} = \emptyset$$
  $$\text{Train}_{\text{PID}} \cap \text{Test}_{\text{PID}} = \emptyset$$
  $$\text{Val}_{\text{PID}} \cap \text{Test}_{\text{PID}} = \emptyset$$

---

## 3. Data Preprocessing & Augmentation Protocol

### 3.1 Preprocessing Pipeline
1. **Raw MRI Load**: Extract 512x512 uint16 slice from `.mat` (`cjdata.image`).
2. **Min-Max Intensity Normalization**: Map intensity range $[0, \max(I)]$ to $[0.0, 1.0]$.
3. **Spatial Resizing**: Bilinear interpolation to target resolution:
   - ResNet50: $224 \times 224 \times 3$
   - EfficientNetB0: $224 \times 224 \times 3$
   - EfficientNetB3: $300 \times 300 \times 3$
4. **Channel Replication**: Single-channel grayscale converted to 3-channel RGB representation via channel duplication $[I, I, I]$ for ImageNet backbone compatibility.
5. **Standard ImageNet Z-score Normalization**:
   $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

### 3.2 Augmentation Policy (Training Only)
- **Random Rotation**: $\pm 10^\circ$ (small anatomical rotation)
- **Random Affine Translation**: $\pm 5\%$ horizontal/vertical translation
- **Random Scaling / Zoom**: $[0.95, 1.05]$
- **Horizontal Flip**: $p = 0.5$ (justified by intracranial bilateral anatomical symmetry)
- **Forbidden Augmentations**: Vertical flips, aggressive shearing, extreme color distortion, synthetic cut-and-paste artifacts.

---

## 4. Model Architectures & Transfer Learning Strategy

### 4.1 Candidate Architectures
1. **Baseline**: ResNet50 (25.6M parameters)
2. **Candidate 1**: EfficientNetB0 (5.3M parameters)
3. **Candidate 2**: EfficientNetB3 (12.2M parameters)

### 4.2 Classification Head Design
$$\text{Backbone} \longrightarrow \text{GlobalAveragePooling2D} \longrightarrow \text{Dropout}(p=0.3) \longrightarrow \text{Linear}(d_{\text{feat}}, 3)$$

### 4.3 Two-Stage Training Protocol
- **Stage 1 (Feature Extraction)**:
  - Backbone frozen with ImageNet weights.
  - Classification head trained for 5 epochs using AdamW optimizer ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$).
- **Stage 2 (Fine-Tuning)**:
  - Top convolutional blocks unfrozen.
  - End-to-end fine-tuning with reduced learning rate ($\text{lr} = 10^{-4}$).
  - Cosine annealing learning rate schedule with minimum $\text{lr} = 10^{-6}$.
  - Early stopping patience = 7 epochs monitored on **Validation Macro F1**.
  - Checkpoint rule: Save model weights achieving highest Validation Macro F1.

---

## 5. Loss Formulation & Class Imbalance Handling

- Standard **Categorical Cross-Entropy Loss**:
  $$\mathcal{L}_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{i,c} \log(\hat{p}_{i,c})$$
- Evaluated against **Balanced Class-Weighted Cross-Entropy Loss**:
  $$w_c = \frac{N}{C \cdot N_c}$$

---

## 6. Evaluation Metrics

Model selection and evaluation report all of the following:
1. **Macro F1 Score** (Primary Selection Metric):
   $$\text{Macro F1} = \frac{1}{C} \sum_{c=1}^C \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
2. **Overall Accuracy** and **Balanced Accuracy**
3. **Per-Class Metrics**: Precision, Recall / Sensitivity, Specificity, F1-Score
4. **Multi-Class One-vs-Rest ROC-AUC**: Macro-averaged Area Under the ROC Curve
5. **Expected Calibration Error (ECE)** & **Brier Score**
6. **Quantitative XAI Metrics**: Saliency-Mask IoU, Dice Score, Pointing Game Hit-Rate
7. **Computational Metrics**: Total parameters, FP32 Model Size (MB), Batch-1 Inference Latency (ms)

---

## 7. Calibration Protocol
- Validation set logits $z \in \mathbb{R}^C$ used to optimize scalar temperature $T > 0$:
  $$\min_{T} -\sum_{i} \log \sigma(z_i / T)_{y_i}$$
- Fitted temperature $T^*$ applied to test set predictions without retraining.

---

## 8. Reproducibility & Random Seed Protocol
- Fixed Global Random Seed: `42` across Python `random`, `numpy`, `torch` (`torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`), and deterministic CuDNN operations (`torch.backends.cudnn.deterministic = True`).
