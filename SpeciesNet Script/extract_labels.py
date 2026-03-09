"""
Extract a clean CSV and JSON of all images with true and predicted labels.

Reads speciesnet_results.json and writes:
  - labeled_results.json  : [{image_path, true_label, true_scientific,
                               predicted_label, predicted_scientific,
                               match, prediction_score}, ...]
  - labeled_results.csv   : same data as a CSV

Usage:
    python extract_labels.py
    python extract_labels.py --results speciesnet_results.json --output labeled_results
"""

import argparse
import csv
import json
import os
import sys


FIELDNAMES = [
    "image_path",
    "true_label",
    "true_scientific",
    "predicted_label",
    "predicted_scientific",
    "match",
    "prediction_score",
    "raw_prediction",
]


def main():
    parser = argparse.ArgumentParser(
        description="Extract image_path, true labels, and predicted labels from SpeciesNet results."
    )
    parser.add_argument(
        "--results", default="speciesnet_results.json",
        help="Path to speciesnet_results.json (default: speciesnet_results.json)"
    )
    parser.add_argument(
        "--output", default="labeled_results",
        help="Output filename without extension (default: labeled_results)"
    )
    parser.add_argument(
        "--only_labeled", action="store_true",
        help="If set, skip images that have no true_label AND no true_scientific."
    )
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"[ERROR] File not found: {args.results}")
        sys.exit(1)

    with open(args.results, "r") as f:
        data = json.load(f)

    results = data.get("results", [])
    print(f"[INFO] Loaded {len(results)} entries from {args.results}")

    # Print match strategy from summary if present
    strategy = data.get("summary", {}).get("match_strategy", "")
    if strategy:
        print(f"[INFO] Match strategy: {strategy}")

    rows    = []
    skipped = 0

    for r in results:
        true_label      = r.get("true_label",           "").strip()
        true_scientific = r.get("true_scientific",       "").strip()
        has_truth       = bool(true_label or true_scientific)

        if args.only_labeled and not has_truth:
            skipped += 1
            continue

        rows.append({
            "image_path":           r.get("image_path",           "").strip(),
            "true_label":           true_label,
            "true_scientific":      true_scientific,
            "predicted_label":      r.get("predicted_label",      "").strip(),
            "predicted_scientific": r.get("predicted_scientific",  "").strip(),
            "match":                r.get("match",                 ""),
            "prediction_score":     r.get("prediction_score",      ""),
            "raw_prediction":       r.get("raw_prediction",        "").strip(),
        })

    if skipped:
        print(f"[INFO] Skipped {skipped} entries with no true_label or true_scientific.")

    # --- Save JSON ---
    json_path = args.output + ".json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[INFO] JSON saved to : {json_path}  ({len(rows)} entries)")

    # --- Save CSV ---
    csv_path = args.output + ".csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] CSV  saved to : {csv_path}  ({len(rows)} entries)")


if __name__ == "__main__":
    main()