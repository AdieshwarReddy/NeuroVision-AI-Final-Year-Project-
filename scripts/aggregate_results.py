"""
Cross-Fold Results Aggregation Script for TrustBT-EfficientNet.

Reads metrics.json from multiple experiment directories and produces:
1. Per-fold performance table (mean ± std)
2. Cross-fold aggregated metrics CSV
3. Publication-quality comparison bar chart
4. LaTeX table for IEEE paper

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --exp_pattern "candidate_efficientnet_b0_fold*"
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))


def load_experiment_metrics(exp_dir: Path) -> Dict:
    """Loads metrics.json from an experiment directory."""
    metrics_path = exp_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        return json.load(f)


def aggregate_folds(exp_pattern: str = "candidate_efficientnet_b0_fold*") -> pd.DataFrame:
    """
    Aggregates metrics from all fold directories matching the pattern.
    Returns DataFrame with one row per fold + summary row.
    """
    artifacts_dir = Path("artifacts/experiments")
    matching_dirs = sorted(artifacts_dir.glob(exp_pattern))

    if not matching_dirs:
        print(f"[!] No experiments found matching pattern: {exp_pattern}")
        return pd.DataFrame()

    records = []
    for exp_dir in matching_dirs:
        metrics = load_experiment_metrics(exp_dir)
        if metrics is None:
            print(f"[!] Skipping {exp_dir.name} — no metrics.json")
            continue

        # Try test metrics first, fallback to val
        m = metrics.get("test_metrics", metrics.get("val_metrics", {}))
        if not m:
            continue

        record = {
            "experiment": exp_dir.name,
            "best_epoch": metrics.get("best_epoch", -1),
            "accuracy": m.get("accuracy", 0.0),
            "macro_f1": m.get("macro_f1", 0.0),
            "balanced_accuracy": m.get("balanced_accuracy", 0.0),
            "roc_auc": m.get("roc_auc_macro_ovr", 0.0),
            "ece": m.get("expected_calibration_error", 0.0),
            "brier_score": m.get("brier_score", 0.0),
        }

        # Per-class F1
        per_class = m.get("per_class", {})
        for cls_name in ["Meningioma", "Glioma", "Pituitary"]:
            cls_data = per_class.get(cls_name, {})
            record[f"f1_{cls_name.lower()}"] = cls_data.get("f1_score", 0.0)
            record[f"sens_{cls_name.lower()}"] = cls_data.get("recall_sensitivity", 0.0)

        records.append(record)

    if not records:
        print("[!] No valid metric records found.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    return df


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Computes mean and std for numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary = df[numeric_cols].agg(["mean", "std"]).T
    summary.columns = ["Mean", "Std"]
    summary["Mean±Std"] = summary.apply(lambda r: f"{r['Mean']:.4f} ± {r['Std']:.4f}", axis=1)
    return summary


def plot_model_comparison(
    results: Dict[str, pd.DataFrame],
    output_path: Path,
    metrics: List[str] = ["accuracy", "macro_f1", "roc_auc", "ece"]
):
    """
    Creates a publication-quality grouped bar chart comparing models.

    Args:
        results: {model_name: aggregated_df} mapping.
        output_path: Output PNG path.
        metrics: Metrics to plot.
    """
    metric_labels = {
        "accuracy": "Accuracy",
        "macro_f1": "Macro F1",
        "roc_auc": "AUC-ROC",
        "ece": "ECE (↓)"
    }

    model_names = list(results.keys())
    n_models = len(model_names)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    colors = ["#4c84e0", "#e84393", "#2eca7f", "#f5a623"]

    for ax, metric in zip(axes, metrics):
        means = []
        stds = []
        for model_name in model_names:
            df = results[model_name]
            if metric in df.columns:
                means.append(df[metric].mean())
                stds.append(df[metric].std() if len(df) > 1 else 0.0)
            else:
                means.append(0.0)
                stds.append(0.0)

        x = np.arange(n_models)
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors[:n_models],
                      alpha=0.85, edgecolor="black", linewidth=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=10)
        ax.set_title(metric_labels.get(metric, metric), fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Value labels on bars
        for bar, mean_val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.005,
                f"{mean_val:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold"
            )

        # ECE: lower is better; invert y-axis direction indicator
        if metric == "ece":
            ax.invert_yaxis()

    fig.suptitle("TrustBT-EfficientNet: Model Comparison (Fold 1 Test Set)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Model comparison chart saved: {output_path}")


def generate_latex_table(df: pd.DataFrame, model_name: str) -> str:
    """Generates a LaTeX table row for IEEE paper."""
    acc = df["accuracy"].mean()
    f1 = df["macro_f1"].mean()
    auc = df["roc_auc"].mean()
    ece = df["ece"].mean()
    acc_std = df["accuracy"].std() if len(df) > 1 else 0.0
    f1_std = df["macro_f1"].std() if len(df) > 1 else 0.0

    return (f"{model_name} & "
            f"{acc:.4f} $\\pm$ {acc_std:.4f} & "
            f"{f1:.4f} $\\pm$ {f1_std:.4f} & "
            f"{auc:.4f} & "
            f"{ece:.4f} \\\\\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patterns", nargs="+",
                        default=["baseline_resnet50_fold*",
                                 "candidate_efficientnet_b0_fold*",
                                 "candidate_efficientnet_b3_fold*"],
                        help="Glob patterns for experiment directories")
    parser.add_argument("--output_dir", type=str, default="artifacts/figures",
                        help="Output directory for plots and tables")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_labels = {
        "baseline_resnet50_fold*": "ResNet-50 (Baseline)",
        "candidate_efficientnet_b0_fold*": "EfficientNet-B0",
        "candidate_efficientnet_b3_fold*": "EfficientNet-B3"
    }

    all_results = {}
    all_dfs = []

    latex_rows = []

    for pattern in args.patterns:
        model_label = model_labels.get(pattern, pattern)
        df = aggregate_folds(pattern)
        if df.empty:
            continue

        all_results[model_label] = df
        df["model"] = model_label
        all_dfs.append(df)

        print(f"\n{'='*60}")
        print(f"Model: {model_label}")
        print(f"{'='*60}")
        summary = compute_summary_stats(df)
        print(summary["Mean±Std"].to_string())

        latex_rows.append(generate_latex_table(df, model_label))

        # Save per-model CSV
        safe_name = model_label.replace(" ", "_").replace("-", "").replace("(", "").replace(")", "")
        df.to_csv(output_dir / f"results_{safe_name}.csv", index=False)

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv(output_dir / "all_results_combined.csv", index=False)
        print(f"\n[OK] Combined results saved to: {output_dir / 'all_results_combined.csv'}")

    if all_results:
        # Model comparison chart
        plot_model_comparison(
            results=all_results,
            output_path=output_dir / "model_comparison.png"
        )

    # LaTeX table
    if latex_rows:
        latex_table = (
            "\\begin{table}[!h]\n"
            "\\caption{Classification Performance Comparison on Jun Cheng Brain Tumor Dataset}\n"
            "\\label{tab:main_results}\n"
            "\\centering\n"
            "\\begin{tabular}{lcccc}\n"
            "\\hline\n"
            "\\textbf{Model} & \\textbf{Accuracy} & \\textbf{Macro F1} & \\textbf{AUC-ROC} & \\textbf{ECE} \\\\\n"
            "\\hline\n"
        )
        for row in latex_rows:
            latex_table += row
        latex_table += (
            "\\hline\n"
            "\\end{tabular}\n"
            "\\end{table}\n"
        )

        latex_path = output_dir / "table_main_results.tex"
        with open(latex_path, "w") as f:
            f.write(latex_table)
        print(f"[OK] LaTeX table saved: {latex_path}")
        print("\n" + latex_table)


if __name__ == "__main__":
    main()
