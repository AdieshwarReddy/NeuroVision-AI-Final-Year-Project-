# Research Gap and Core Contributions: TrustBT-EfficientNet

## 1. Precise Articulation of the Research Gap

Recent publications (2024–2026) in automated brain tumor MRI classification predominantly emphasize achieving near-perfect classification accuracies (>98–99%) by deploying complex convolutional or hybrid vision transformer architectures. However, a rigorous, critical analysis of this literature reveals four fundamental research gaps that undermine real-world trustworthiness and clinical translational potential:

### Gap 1: Widespread Slice-Level Data Leakage (Patient Identity Memorization)
Most existing works randomly shuffle 2D MRI slices into training and testing partitions without tracking Patient Identifiers (`PID`). In 3D MRI acquisitions, adjacent 2D axial/coronal slices from the same subject share identical cranial geometry, ventricular landmarks, tissue intensities, and acquisition noise. When slices from the same subject appear in both training and test partitions, classifiers exploit patient anatomical signatures rather than tumor-specific pathological features, yielding inflated performance metrics that collapse when evaluated on unseen clinical cohorts (as documented by Yagis et al., 2021).

### Gap 2: Overconfident, Uncalibrated Softmax Probabilities
Standard deep neural networks trained with Cross-Entropy loss produce overconfident softmax probabilities that do not correspond to true empirical likelihoods of correctness. In high-stakes neuro-oncology workflows, an erroneous prediction with 99% reported confidence can mislead clinicians. Existing brain tumor classification papers rarely evaluate probability calibration (ECE, Brier Score) or implement validation-fitted post-hoc calibration (e.g., Temperature Scaling) and selective prediction / abstention mechanisms.

### Gap 3: Qualitative "Eyeball" XAI without Quantitative Saliency Grounding
Explainable AI (XAI) in existing literature is frequently reduced to presenting qualitative Grad-CAM heatmaps for a handful of successful cases without objective validation. Because the Jun Cheng Figshare dataset contains ground-truth expert radiologist tumor masks (`tumorMask`), it is methodologically imperative to quantitatively measure saliency alignment using objective spatial overlap metrics (Intersection over Union, Dice coefficient, Pointing Game Hit-Rate), rather than subjective visual inspection.

### Gap 4: Lack of Systematic Robustness and Edge Efficiency Profiling
Real-world clinical MRI scans exhibit variability in signal-to-noise ratio (SNR), patient micro-motion blur, and radiofrequency (RF) coil inhomogeneity. Existing models are rarely benchmarked under controlled sensory perturbations (Gaussian noise, Gaussian blur, contrast variation). Furthermore, studies proposing massive architectures (such as customized EfficientNet-B9 or deep Vision Transformers) neglect practical deployment constraints (edge latency, parameter efficiency, memory footprint).

---

## 2. Formal Research Question

$$\begin{aligned}
\textbf{Research Question:} \quad &\text{"Can an EfficientNet-based brain tumor MRI classifier achieve reliable} \\
&\text{patient-level generalization while remaining explainable, confidence-aware,} \\
&\text{robust, and computationally practical?"}
\end{aligned}$$

---

## 3. Methodological Contributions of TrustBT-EfficientNet

To address these defined gaps, the TrustBT-EfficientNet framework establishes the following verified contributions:

1. **Leakage-Aware Patient-Disjoint Partitioning Protocol**:
   - Enforcement of strict `StratifiedGroupKFold` partitioning grouped exclusively by Patient ID (`PID`).
   - Automated unit test assertions mathematically guaranteeing that $\text{Train}_{\text{PID}} \cap \text{Val}_{\text{PID}} = \emptyset$, $\text{Train}_{\text{PID}} \cap \text{Test}_{\text{PID}} = \emptyset$, and $\text{Val}_{\text{PID}} \cap \text{Test}_{\text{PID}} = \emptyset$.
   - A controlled empirical ablation quantifying the exact performance divergence between slice-level random splitting versus patient-disjoint splitting.

2. **Rigorous Backbone Benchmark under Identical Conditions**:
   - Controlled multi-stage transfer learning comparison between ResNet50, EfficientNetB0, and EfficientNetB3 under identical patient folds, preprocessing, loss formulations, and early stopping criteria.
   - Primary model selection guided by Macro F1 score rather than overall accuracy, accounting for class imbalance (1426 Gliomas vs. 708 Meningiomas vs. 930 Pituitary tumors).

3. **Validation-Fitted Probability Calibration & Selective Prediction**:
   - Implementation and evaluation of Expected Calibration Error (ECE), Brier Score, and Reliability Diagrams.
   - Post-hoc Temperature Scaling fitted strictly on validation partitions to prevent test leakage.
   - Uncertainty-aware selective prediction mechanism that triggers "Low Confidence / Expert Review Required" status when maximum calibrated confidence falls below safety thresholds.

4. **Quantitative XAI Localization Benchmark**:
   - Quantitative evaluation of Grad-CAM / Grad-CAM++ saliency maps against ground-truth expert radiologist tumor masks (`tumorMask`) using IoU, Dice overlap, and Pointing Game hit rate.
   - Rigorous delineation: clearly framing Grad-CAM as classification attribution rather than segmentation.

5. **Sensory Robustness Stress-Testing Protocol**:
   - Evaluation of frozen model checkpoints under controlled multi-severity Gaussian noise, Gaussian blur, and contrast perturbations to measure $\Delta \text{Macro F1}$ degradation without test-set retraining.

6. **Computational Efficiency & Clinical Prototype Deployment**:
   - Benchmarking parameter count, model size, and CPU/edge inference latency (batch size = 1).
   - Deployment of a reproducible, privacy-preserving full-stack prototype (Streamlit + Supabase) that operates without storing raw patient MRI pixels.
