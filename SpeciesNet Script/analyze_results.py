"""
SpeciesNet Results Analysis Script.

Reads speciesnet_results.json and produces:
  - Confusion matrix (text + heatmap PNG)
  - Per-class precision, recall, F1
  - Top-N most confused species pairs
  - Overall accuracy and macro averages

The match strategy is: predicted label is checked against BOTH the
CommonName and ScientificName (case-insensitive), consistent with
run_speciesnet.py.  Labels are normalized to lowercase before all
sklearn metric calls to avoid case-mismatch inflating error counts.

Usage:
    python analyze_results.py --results speciesnet_results.json
    python analyze_results.py --results speciesnet_results.json --top_n 20 --output_dir ./analysis
"""

import argparse
import json
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_results(results_path: str):
    with open(results_path, "r") as f:
        data = json.load(f)
    results = data.get("results", [])
    # Only keep entries where ground truth exists
    filtered = [r for r in results if r.get("true_label") or r.get("true_scientific")]
    print(f"[INFO] Loaded {len(results)} total entries, {len(filtered)} with ground-truth labels.")
    return filtered


def get_labels_and_preds(results: list[dict]):
    """
    Return (y_true, y_pred) lists, both normalized to lowercase so that
    case differences (e.g. 'White-Tailed Deer' vs 'white-tailed deer')
    do not inflate error counts.

    Uses CommonName as the canonical true label (falls back to ScientificName
    if CommonName is empty).
    """
    y_true = []
    y_pred = []
    for r in results:
        true = (r.get("true_label") or r.get("true_scientific", "")).strip().lower()
        pred = (r.get("predicted_label") or "").strip().lower()
        y_true.append(true)
        y_pred.append(pred)
    return y_true, y_pred


# --------------------------------------------------------------------------- #
# 1. Overall Accuracy & Macro Averages
# --------------------------------------------------------------------------- #

def print_overall_metrics(y_true, y_pred):
    acc    = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    macro    = report.get("macro avg", {})
    weighted = report.get("weighted avg", {})

    print(f"\n{'='*60}")
    print(f"  Overall Metrics")
    print(f"{'='*60}")
    print(f"  Accuracy                  : {acc:.4f}  ({acc:.2%})")
    print(f"  Macro    Precision        : {macro.get('precision', 0):.4f}")
    print(f"  Macro    Recall           : {macro.get('recall', 0):.4f}")
    print(f"  Macro    F1-score         : {macro.get('f1-score', 0):.4f}")
    print(f"  Weighted Precision        : {weighted.get('precision', 0):.4f}")
    print(f"  Weighted Recall           : {weighted.get('recall', 0):.4f}")
    print(f"  Weighted F1-score         : {weighted.get('f1-score', 0):.4f}")
    print(f"  Total classes             : {len(set(y_true))}")
    print(f"{'='*60}\n")

    return acc, report


# --------------------------------------------------------------------------- #
# 2. Per-Class Precision / Recall / F1
# --------------------------------------------------------------------------- #

def save_per_class_metrics(report: dict, output_dir: str):
    skip = {"accuracy", "macro avg", "weighted avg"}
    rows = [
        (cls, v["precision"], v["recall"], v["f1-score"], int(v["support"]))
        for cls, v in report.items()
        if cls not in skip
    ]
    # Sort by F1 descending
    rows.sort(key=lambda x: x[3], reverse=True)

    out_path = os.path.join(output_dir, "per_class_metrics.txt")
    with open(out_path, "w") as f:
        header = f"{'Species':<45} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for cls, p, r, f1, sup in rows:
            f.write(f"{cls:<45} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {sup:>10}\n")

    print(f"[INFO] Per-class metrics saved to: {out_path}")

    print(f"\n--- Top 20 Species by F1 ---")
    for cls, p, r, f1, sup in rows[:20]:
        print(f"  {cls:<40}  F1={f1:.3f}  P={p:.3f}  R={r:.3f}  n={sup}")

    print(f"\n--- Bottom 20 Species by F1 (min 5 samples) ---")
    bottom = [row for row in rows if row[4] >= 5][-20:]
    for cls, p, r, f1, sup in reversed(bottom):
        print(f"  {cls:<40}  F1={f1:.3f}  P={p:.3f}  R={r:.3f}  n={sup}")


# --------------------------------------------------------------------------- #
# 3. Top-N Most Confused Species Pairs
# --------------------------------------------------------------------------- #

def print_top_confusions(y_true, y_pred, top_n: int, output_dir: str):
    confusion_pairs = Counter()
    for true, pred in zip(y_true, y_pred):
        if true != pred:        # already lowercased
            confusion_pairs[(true, pred)] += 1

    most_common = confusion_pairs.most_common(top_n)

    print(f"\n--- Top {top_n} Most Confused Species Pairs ---")
    print(f"  {'True Label':<35} {'Predicted As':<35} {'Count':>6}")
    print(f"  {'-'*35} {'-'*35} {'-'*6}")
    for (true, pred), count in most_common:
        print(f"  {true:<35} {pred:<35} {count:>6}")

    out_path = os.path.join(output_dir, "top_confusions.txt")
    with open(out_path, "w") as f:
        f.write(f"{'True Label':<35} {'Predicted As':<35} {'Count':>6}\n")
        f.write(f"{'-'*35} {'-'*35} {'-'*6}\n")
        for (true, pred), count in most_common:
            f.write(f"{true:<35} {pred:<35} {count:>6}\n")
    print(f"[INFO] Top confusions saved to: {out_path}")


# --------------------------------------------------------------------------- #
# 4. Confusion Matrix
# --------------------------------------------------------------------------- #

def save_confusion_matrix(y_true, y_pred, output_dir: str, max_classes: int = 40):
    labels = sorted(set(y_true))

    if len(labels) > max_classes:
        top_labels = [label for label, _ in Counter(y_true).most_common(max_classes)]
        top_labels = sorted(top_labels)
        mask       = [i for i, (t, _) in enumerate(zip(y_true, y_pred)) if t in top_labels]
        y_true_f   = [y_true[i] for i in mask]
        y_pred_f   = [y_pred[i] for i in mask]
        title_note = f"(Top {max_classes} most frequent species)"
        print(f"[INFO] Too many classes ({len(labels)}) for a readable plot. "
              f"Showing top {max_classes} by frequency.")
    else:
        y_true_f, y_pred_f = y_true, y_pred
        top_labels         = labels
        title_note         = ""

    cm = confusion_matrix(y_true_f, y_pred_f, labels=top_labels)

    # --- Save raw text matrix ---
    txt_path = os.path.join(output_dir, "confusion_matrix.txt")
    with open(txt_path, "w") as f:
        header = f"{'':>35}" + "".join(f"{l[:12]:>14}" for l in top_labels)
        f.write(header + "\n")
        for i, row_label in enumerate(top_labels):
            row_str = f"{row_label[:35]:>35}" + "".join(f"{v:>14}" for v in cm[i])
            f.write(row_str + "\n")
    print(f"[INFO] Confusion matrix (text) saved to: {txt_path}")

    # --- Plot heatmap ---
    n        = len(top_labels)
    fig_size = max(12, n * 0.45)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    # Normalize rows for color (show recall per class visually)
    cm_norm  = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm  = cm_norm / row_sums

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall (row-normalised)")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(top_labels, rotation=90, fontsize=max(5, 9 - n // 10))
    ax.set_yticklabels(top_labels, fontsize=max(5, 9 - n // 10))
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"Confusion Matrix {title_note}\n(colour = row-normalised recall)", fontsize=12)

    if n <= 25:
        for i in range(n):
            for j in range(n):
                val = cm[i, j]
                if val > 0:
                    color = "white" if cm_norm[i, j] > 0.6 else "black"
                    ax.text(j, i, str(val), ha="center", va="center",
                            fontsize=7, color=color)

    plt.tight_layout()
    png_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix (heatmap) saved to: {png_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Analyze SpeciesNet results: confusion matrix, per-class metrics, confusions."
    )
    parser.add_argument(
        "--results", default="speciesnet_results.json",
        help="Path to speciesnet_results.json (default: speciesnet_results.json)"
    )
    parser.add_argument(
        "--top_n", type=int, default=20,
        help="Number of top confused species pairs to show (default: 20)"
    )
    parser.add_argument(
        "--max_classes", type=int, default=40,
        help="Max classes to show in the confusion matrix heatmap (default: 40)"
    )
    parser.add_argument(
        "--output_dir", default="analysis",
        help="Directory to save output files (default: ./analysis)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"[ERROR] Results file not found: {args.results}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load
    results         = load_results(args.results)
    y_true, y_pred  = get_labels_and_preds(results)

    # 1. Overall metrics
    acc, report = print_overall_metrics(y_true, y_pred)

    # 2. Per-class metrics
    save_per_class_metrics(report, args.output_dir)

    # 3. Top-N confusions
    print_top_confusions(y_true, y_pred, args.top_n, args.output_dir)

    # 4. Confusion matrix
    save_confusion_matrix(y_true, y_pred, args.output_dir, args.max_classes)

    print(f"\n[INFO] All analysis outputs saved to: ./{args.output_dir}/")
    print(f"         - confusion_matrix.png")
    print(f"         - confusion_matrix.txt")
    print(f"         - per_class_metrics.txt")
    print(f"         - top_confusions.txt")


if __name__ == "__main__":
    main()