"""
SpeciesNet inference script.

Reads labels.json, runs SpeciesNet on each image, and compares
the predicted species against both the ground-truth CommonName AND
ScientificName label (case-insensitive).

After saving results, automatically runs:
  - analyze_results.py   (metrics + confusion matrix)
  - extract_labels.py    (CSV / JSON export)

Usage:
    python run_speciesnet.py --labels labels.json --output predictions.json

Optional flags:
    --country USA          (3-letter ISO code, improves accuracy via geofencing)
    --admin1_region GA     (US state code, only used when --country USA)
    --predictions_json     (skip inference, re-run comparison only)
    --skip_analysis        (skip auto-running analyze/extract scripts)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile


# --------------------------------------------------------------------------- #
# Prediction normalization
#
# SpeciesNet sometimes returns genus/family-level fallbacks or alternate common
# names.  Map those to the canonical CommonName strings used in labels.json.
# Keys are lowercase; values are the exact CommonName strings from labels.json.
# Expand this table as you discover new mismatches in the confusion matrix.
# --------------------------------------------------------------------------- #

PREDICTION_NORMALIZATION: dict[str, str] = {
    # Raccoon variants
    "northern raccoon":          "Raccoon",
    "procyon lotor":             "Raccoon",

    # White-Tailed Deer variants
    "white-tailed deer":         "White-Tailed Deer",
    "odocoileus virginianus":    "White-Tailed Deer",
    "odocoileus species":        "White-Tailed Deer",
    "cervidae family":           "White-Tailed Deer",

    # Eastern Gray Squirrel variants
    "eastern gray squirrel":     "Eastern Gray Squirrel",
    "sciurus carolinensis":      "Eastern Gray Squirrel",
    "sciurus species":           "Eastern Gray Squirrel",

    # Coyote variants
    "coyote":                    "Coyote",
    "canis latrans":             "Coyote",

    # Canada Goose variants
    "canada goose":              "Canada Goose",
    "branta canadensis":         "Canada Goose",

    # Coarse / non-species predictions – keep as-is (will count as wrong)
    "animal":                    "animal",
    "mammal":                    "mammal",
    "bird":                      "bird",
    "blank":                     "blank",
    "no cv result":              "no cv result",
}


def normalize_prediction(label: str) -> str:
    """Map a raw SpeciesNet label to the canonical CommonName (or keep as-is)."""
    return PREDICTION_NORMALIZATION.get(label.strip().lower(), label.strip())


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_labels(labels_path: str) -> list[dict]:
    """Load labels.json; return list of {image_path, CommonName, ScientificName}."""
    with open(labels_path, "r") as f:
        data = json.load(f)

    # Support both a bare list and {"data": [...]} / {"labels": [...]} wrappers
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        for key in ("data", "labels", "images", "annotations"):
            if key in data:
                entries = data[key]
                break
        else:
            raise ValueError(
                "labels.json is a dict but no recognised list key found "
                "(expected 'data', 'labels', 'images', or 'annotations')."
            )
    else:
        raise ValueError(f"Unexpected top-level type in labels.json: {type(data)}")

    cleaned = []
    for i, entry in enumerate(entries):
        path = entry.get("image_path", "").strip()
        if not path:
            print(f"[WARN] Entry {i} missing 'image_path', skipping.")
            continue
        cleaned.append({
            "image_path":     path,
            "CommonName":     entry.get("CommonName",     "").strip(),
            "ScientificName": entry.get("ScientificName", "").strip(),
        })

    return cleaned


def build_instances_txt(entries: list[dict], out_path: str) -> None:
    """Write a plain-text filepaths file, one absolute path per line."""
    with open(out_path, "w") as f:
        for e in entries:
            f.write(e["image_path"] + "\n")


def run_speciesnet(instances_txt: str, predictions_json: str,
                   country: str | None, admin1: str | None) -> bool:
    """Call SpeciesNet's run_model script. Returns True on success."""
    cmd = [
        sys.executable, "-m", "speciesnet.scripts.run_model",
        "--filepaths_txt", instances_txt,
        "--predictions_json", predictions_json,
    ]
    if country:
        cmd += ["--country", country]
    if admin1:
        cmd += ["--admin1_region", admin1]

    print(f"\n[INFO] Running SpeciesNet:\n  {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    return result.returncode == 0


def extract_label_from_prediction(pred: dict) -> str:
    """
    Pull the human-readable species name from a SpeciesNet prediction dict.

    SpeciesNet's 'prediction' field looks like:
        "guid;;kingdom;phylum;class;order;family;genus;species;common_name"

    We also capture the scientific name (genus + species tokens) so the
    caller can do dual matching.

    Returns the normalized common name (last token), after applying the
    PREDICTION_NORMALIZATION table.
    """
    raw = pred.get("prediction", "")
    if not raw:
        return "no prediction"

    # Format: <guid>;;k;p;c;o;f;g;species;common_name  (common_name is last)
    parts = raw.split(";")
    common = parts[-1].strip() if parts else ""

    if common:
        return normalize_prediction(common)

    return normalize_prediction(raw)


def extract_scientific_from_prediction(pred: dict) -> str:
    """
    Extract the genus + species tokens from a raw SpeciesNet prediction string.

    Prediction format:
        <guid>;;kingdom;phylum;class;order;family;genus;species;common_name
    Indices (0-based after split on ';'):
        0  = guid
        1  = '' (empty)
        2  = kingdom
        3  = phylum
        4  = class
        5  = order
        6  = family
        7  = genus
        8  = species (epithet)
        9  = common_name
    Returns "Genus Species" title-cased, or "" if not available.
    """
    raw = pred.get("prediction", "")
    if not raw:
        return ""
    parts = raw.split(";")
    # Need at least genus (index 7) and species (index 8)
    if len(parts) >= 9:
        genus   = parts[7].strip()
        species = parts[8].strip()
        if genus and species:
            return f"{genus.capitalize()} {species.capitalize()}"
        if genus:
            return genus.capitalize()
    return ""


def is_match(true_common: str, true_scientific: str, predicted_label: str,
             pred_scientific: str) -> bool:
    """
    Return True if the prediction matches either:
      - the ground-truth CommonName  (case-insensitive), OR
      - the ground-truth ScientificName (case-insensitive).

    Also checks the raw predicted scientific name against both ground-truth names.
    """
    pl  = predicted_label.lower().strip()
    psl = pred_scientific.lower().strip()
    tcl = true_common.lower().strip()
    tsl = true_scientific.lower().strip()

    if not tcl and not tsl:
        return False  # no ground truth to compare

    # Match predicted common name against GT common or scientific
    if tcl and pl == tcl:
        return True
    if tsl and pl == tsl:
        return True

    # Match predicted scientific name against GT common or scientific
    if psl:
        if tcl and psl == tcl:
            return True
        if tsl and psl == tsl:
            return True

    return False


def compare_results(entries: list[dict], predictions_json: str) -> tuple[list[dict], float]:
    """
    Merge ground-truth labels with SpeciesNet predictions.

    Returns (results_list, accuracy).
    """
    with open(predictions_json, "r") as f:
        preds_data = json.load(f)

    predictions: list[dict] = preds_data.get("predictions", [])

    # Build lookup: filepath -> prediction dict
    pred_by_path: dict[str, dict] = {p["filepath"]: p for p in predictions}

    results      = []
    correct      = 0
    total_labeled = 0

    for entry in entries:
        img_path        = entry["image_path"]
        true_common     = entry["CommonName"]
        true_scientific = entry["ScientificName"]
        pred            = pred_by_path.get(img_path, {})

        predicted_label    = extract_label_from_prediction(pred)
        predicted_scientific = extract_scientific_from_prediction(pred)

        has_truth = bool(true_common or true_scientific)
        match = (
            is_match(true_common, true_scientific, predicted_label, predicted_scientific)
            if has_truth
            else None
        )

        if has_truth:
            total_labeled += 1
            if match:
                correct += 1

        results.append({
            "image_path":           img_path,
            "true_label":           true_common,
            "true_scientific":      true_scientific,
            "predicted_label":      predicted_label,
            "predicted_scientific": predicted_scientific,
            "match":                match,
            "raw_prediction":       pred.get("prediction", ""),
            "prediction_score":     pred.get("prediction_score", None),
        })

    accuracy = correct / total_labeled if total_labeled > 0 else 0.0
    print(f"\n{'='*60}")
    print(f"  Results summary")
    print(f"{'='*60}")
    print(f"  Total images processed : {len(results)}")
    print(f"  Images with labels     : {total_labeled}")
    print(f"  Correct predictions    : {correct}")
    print(f"  Accuracy               : {accuracy:.2%}")
    print(f"  (match = common name OR scientific name, case-insensitive)")
    print(f"{'='*60}\n")

    return results, accuracy


def save_results(results: list[dict], accuracy: float, out_path: str) -> None:
    """Save the merged results to a JSON file."""
    output = {
        "summary": {
            "total":    len(results),
            "correct":  sum(1 for r in results if r["match"] is True),
            "accuracy": round(accuracy, 4),
            "match_strategy": "common_name OR scientific_name (case-insensitive)",
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[INFO] Full results saved to: {out_path}")


def run_script(script_name: str, extra_args: list[str] = []) -> bool:
    """Run a companion Python script (same directory) as a subprocess."""
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, script_name)
    if not os.path.exists(script_path):
        print(f"[WARN] {script_name} not found at {script_path}, skipping.")
        return False
    cmd = [sys.executable, script_path] + extra_args
    print(f"\n[INFO] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[WARN] {script_name} exited with code {result.returncode}.")
    return result.returncode == 0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Run SpeciesNet inference and compare against CommonName/ScientificName labels."
    )
    parser.add_argument(
        "--labels", default="labels.json",
        help="Path to your labels.json file (default: labels.json)"
    )
    parser.add_argument(
        "--output", default="speciesnet_results.json",
        help="Where to save the final comparison results (default: speciesnet_results.json)"
    )
    parser.add_argument(
        "--predictions_json", default=None,
        help="If you already have a SpeciesNet predictions JSON, skip inference "
             "and just run the comparison. Useful for re-running the analysis."
    )
    parser.add_argument(
        "--country", default="USA",
        help="3-letter ISO country code for geofencing (default: USA). "
             "Pass empty string '' to disable."
    )
    parser.add_argument(
        "--admin1_region", default="GA",
        help="US state code for geofencing (default: GA for Georgia). "
             "Only used when --country USA."
    )
    parser.add_argument(
        "--skip_analysis", action="store_true",
        help="Skip automatically running analyze_results.py and extract_labels.py."
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # 1. Load labels
    # ------------------------------------------------------------------ #
    if not os.path.exists(args.labels):
        print(f"[ERROR] labels.json not found at: {args.labels}")
        sys.exit(1)

    print(f"[INFO] Loading labels from: {args.labels}")
    entries = load_labels(args.labels)
    print(f"[INFO] Found {len(entries)} images.")

    country = args.country.strip() or None
    admin1  = args.admin1_region.strip() if country == "USA" else None

    # ------------------------------------------------------------------ #
    # 2. Run SpeciesNet (unless caller already has predictions)
    # ------------------------------------------------------------------ #
    if args.predictions_json:
        predictions_json = args.predictions_json
        print(f"[INFO] Skipping inference; using existing predictions: {predictions_json}")
    else:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_instances.txt", delete=False
        ) as tmp:
            instances_path = tmp.name

        predictions_json = os.path.splitext(args.output)[0] + "_raw_predictions.json"

        print(f"[INFO] Writing SpeciesNet filepaths TXT to: {instances_path}")
        build_instances_txt(entries, instances_path)

        success = run_speciesnet(instances_path, predictions_json, country, admin1)
        os.unlink(instances_path)

        if not success:
            print("[ERROR] SpeciesNet exited with a non-zero return code.")
            print("        Check the output above for details.")
            sys.exit(1)

    # ------------------------------------------------------------------ #
    # 3. Compare predictions vs ground truth (common + scientific name)
    # ------------------------------------------------------------------ #
    results, accuracy = compare_results(entries, predictions_json)

    # ------------------------------------------------------------------ #
    # 4. Save results
    # ------------------------------------------------------------------ #
    save_results(results, accuracy, args.output)

    # Print a quick preview of mismatches
    mismatches = [r for r in results if r["match"] is False]
    if mismatches:
        print(f"[INFO] First 10 mismatches:")
        for r in mismatches[:10]:
            print(
                f"  TRUE: {r['true_label']!r:30s} ({r['true_scientific']!r})  "
                f"PRED: {r['predicted_label']!r} ({r['predicted_scientific']!r})"
            )

    # ------------------------------------------------------------------ #
    # 5. Auto-run analyze_results.py and extract_labels.py
    # ------------------------------------------------------------------ #
    if not args.skip_analysis:
        print("\n" + "="*60)
        print("  Running analyze_results.py ...")
        print("="*60)
        run_script("analyze_results.py", ["--results", args.output])

        print("\n" + "="*60)
        print("  Running extract_labels.py ...")
        print("="*60)
        run_script("extract_labels.py", ["--results", args.output])
    else:
        print("\n[INFO] --skip_analysis set; skipping analyze_results.py and extract_labels.py.")


if __name__ == "__main__":
    main()