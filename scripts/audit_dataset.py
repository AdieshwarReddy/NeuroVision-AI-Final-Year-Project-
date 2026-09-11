"""
Dataset Audit Script for Jun Cheng Figshare Brain Tumor MRI Dataset.
Verifies image integrity, label encodings, Patient IDs (PID), tumor masks,
generates statistical summary artifacts and diagnostic distribution figures.
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_mat_slice, INV_LABEL_MAP, LABEL_MAP


def run_dataset_audit(
    raw_mat_dir: str = "data/raw_mat",
    artifacts_dir: str = "artifacts"
):
    mat_dir = Path(raw_mat_dir)
    art_dir = Path(artifacts_dir)
    fig_dir = art_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    mat_files = sorted([f for f in mat_dir.glob("*.mat") if f.name != "cvind.mat"])
    
    if len(mat_files) == 0:
        raise FileNotFoundError(f"No .mat files found in {mat_dir}. Please run download script first.")
        
    print(f"[*] Auditing {len(mat_files)} MRI .mat slices in {mat_dir}...")
    
    records = []
    corrupt_files = []
    missing_fields = []
    
    for mat_path in tqdm(mat_files, desc="Auditing slices"):
        try:
            data = load_mat_slice(mat_path)
            img = data["image"]
            lbl = data["label"]
            pid = data["pid"]
            mask = data["tumor_mask"]
            border = data["tumor_border"]
            
            # Validation checks
            if img is None or img.size == 0:
                missing_fields.append({"file": mat_path.name, "issue": "Missing image array"})
            if lbl not in [1, 2, 3]:
                missing_fields.append({"file": mat_path.name, "issue": f"Invalid label: {lbl}"})
            if not pid or pid == "UNKNOWN":
                missing_fields.append({"file": mat_path.name, "issue": "Missing PID"})
            if mask is None or mask.size == 0:
                missing_fields.append({"file": mat_path.name, "issue": "Missing tumor mask"})
                
            records.append({
                "filename": mat_path.name,
                "filepath": str(mat_path.resolve()),
                "raw_label": lbl,
                "class_name": INV_LABEL_MAP[LABEL_MAP[lbl]],
                "patient_id": pid,
                "height": img.shape[0],
                "width": img.shape[1],
                "img_min": float(np.min(img)),
                "img_max": float(np.max(img)),
                "mask_sum_pixels": int(np.sum(mask > 0)),
                "has_mask": bool(np.sum(mask > 0) > 0),
                "has_border": bool(len(border) > 0)
            })
            
        except Exception as e:
            corrupt_files.append({"file": mat_path.name, "error": str(e)})
            
    df = pd.DataFrame(records)
    
    # Save raw audit manifest CSV
    manifest_csv = art_dir / "dataset_manifest_raw.csv"
    df.to_csv(manifest_csv, index=False)
    print(f"[OK] Saved dataset raw manifest: {manifest_csv}")
    
    # Compute Statistics
    total_images = len(df)
    unique_patients = df["patient_id"].nunique()
    
    class_counts = df["class_name"].value_counts().to_dict()
    patients_per_class = df.groupby("class_name")["patient_id"].nunique().to_dict()
    slices_per_patient = df.groupby("patient_id")["filename"].count()
    
    audit_summary = {
        "dataset_name": "Jun Cheng Figshare Brain Tumor Dataset",
        "dataset_doi": "10.6084/m9.figshare.1512427.v5",
        "total_images": total_images,
        "unique_patients": unique_patients,
        "corrupt_files_count": len(corrupt_files),
        "corrupt_files": corrupt_files,
        "missing_fields_count": len(missing_fields),
        "missing_fields": missing_fields,
        "image_dimensions": {
            "unique_shapes": [f"{h}x{w}" for h, w in df[["height", "width"]].drop_duplicates().values],
            "uniform_512x512": bool((df["height"] == 512).all() and (df["width"] == 512).all())
        },
        "slice_counts_by_class": class_counts,
        "patient_counts_by_class": patients_per_class,
        "slices_per_patient_stats": {
            "min": int(slices_per_patient.min()),
            "max": int(slices_per_patient.max()),
            "mean": float(slices_per_patient.mean()),
            "median": float(slices_per_patient.median()),
            "std": float(slices_per_patient.std())
        },
        "mask_integrity": {
            "all_slices_have_masks": bool(df["has_mask"].all()),
            "mean_tumor_area_pixels": float(df["mask_sum_pixels"].mean())
        }
    }
    
    # Save JSON summary
    audit_json_path = art_dir / "dataset_audit.json"
    with open(audit_json_path, "w") as f:
        json.dump(audit_summary, f, indent=4)
    print(f"[OK] Saved dataset audit summary: {audit_json_path}")
    
    # Generate Publication Figures
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # Figure 1: Class and Patient Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    classes = list(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    p_counts = [patients_per_class[c] for c in classes]
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    sns.barplot(x=classes, y=counts, ax=axes[0], palette=palette, hue=classes, legend=False)
    axes[0].set_title("Total MRI Slices per Class", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Number of Slices")
    axes[0].set_xlabel("Tumor Category")
    for i, v in enumerate(counts):
        axes[0].text(i, v + 25, str(v), ha="center", fontweight="bold")
        
    sns.barplot(x=classes, y=p_counts, ax=axes[1], palette=palette, hue=classes, legend=False)
    axes[1].set_title("Unique Patients per Class", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Number of Patients")
    axes[1].set_xlabel("Tumor Category")
    for i, v in enumerate(p_counts):
        axes[1].text(i, v + 2, str(v), ha="center", fontweight="bold")
        
    plt.tight_layout()
    fig1_path = fig_dir / "dataset_class_distribution.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"[OK] Saved figure: {fig1_path}")
    
    # Figure 2: Slices per patient distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(slices_per_patient, bins=25, kde=True, ax=ax, color="#1f77b4")
    ax.set_title("Distribution of Slices per Patient Cohort", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Slices per Patient")
    ax.set_ylabel("Patient Count")
    plt.tight_layout()
    fig2_path = fig_dir / "slices_per_patient_distribution.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"[OK] Saved figure: {fig2_path}")
    
    print("\n" + "="*60)
    print("DATASET AUDIT COMPLETED SUCCESSFULLY")
    print(f"Total Slices: {total_images} | Total Patients: {unique_patients}")
    print(f"Slices: {class_counts}")
    print(f"Patients: {patients_per_class}")
    print("="*60 + "\n")
    
    return audit_summary


if __name__ == "__main__":
    run_dataset_audit()
