#!/usr/bin/env python3
import csv
import json
import os
from typing import Set, List, Dict, Any

# ---------- CONFIGURE THESE PATHS ----------
INPUT_CSV = "/home/hice1/kpanchal30/code/HAAG_Stone_Mountain_Species_Detection_Project/diffusion_test/images_needing_resolution.csv"               # your original CSV (example name)
BW_CSV = "black_and_white_images.csv"      # output CSV with only B/W images

LABELS_INPUT = "/home/hice1/kpanchal30/code/HAAG_Stone_Mountain_Species_Detection_Project/scripts/All_Labels_2.jsonl"    # input JSONL/JSON file
LABELS_OUTPUT = "All_labels_2_with_bw.jsonl"  # output JSONL/JSON file with Black_white label
# ------------------------------------------


def load_black_and_white_filenames(csv_path: str) -> Set[str]:
    """
    Reads the master CSV and returns a set of filenames where is_black_and_white is True.
    Assumes a header row with at least columns: full_path, filename, is_black_and_white, ...
    """
    bw_rows: List[Dict[str, str]] = []
    bw_filenames: Set[str] = set()

    with open(csv_path, "r", newline="") as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            # Handle True/False string robustly
            flag = row.get("is_black_and_white", "").strip().lower()
            is_bw = flag in ("true", "1", "yes")
            if is_bw:
                bw_rows.append(row)
                filename = row.get("filename")
                if filename:
                    bw_filenames.add(filename)

    # Write filtered CSV
    if bw_rows:
        with open(BW_CSV, "w", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=bw_rows[0].keys())
            writer.writeheader()
            writer.writerows(bw_rows)
    else:
        # Create an empty file with just the header if nothing matched
        with open(BW_CSV, "w", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=["full_path", "filename", "is_black_and_white", "width", "height", "error"])
            writer.writeheader()

    print(f"Saved black-and-white-only CSV to: {BW_CSV}")
    print(f"Found {len(bw_filenames)} unique B/W filenames")
    return bw_filenames


def load_labels_file(path: str) -> List[Dict[str, Any]]:
    """
    Loads All_labels_2.jsonl which might be either:
      - JSON Lines (one JSON object per line), or
      - A single JSON array [ {...}, {...}, ... ]
    Returns a list of dicts.
    """
    with open(path, "r") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == "[":
            # JSON array
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Expected a JSON array at top-level.")
            return data
        else:
            # JSONL: one JSON object per line
            entries: List[Dict[str, Any]] = []
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
            return entries


def save_labels_file(path: str, entries: List[Dict[str, Any]], as_array: bool) -> None:
    """
    Saves the entries either as:
      - JSON array if as_array is True
      - JSONL otherwise
    """
    with open(path, "w") as f:
        if as_array:
            json.dump(entries, f, indent=2)
        else:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")


def add_black_white_label_to_labels(
    labels_input: str,
    labels_output: str,
    bw_filenames: Set[str]
) -> None:
    """
    Loads All_labels_2.jsonl, adds 'Black_white' key to each entry based on filename
    membership in bw_filenames, and writes out a new file.
    """
    # Determine if input is array or JSONL so we can preserve format
    with open(labels_input, "r") as f:
        first_char = f.read(1)
        is_array = (first_char == "[")
    entries = load_labels_file(labels_input)

    for entry in entries:
        image_path = entry.get("image_path", "")
        filename = os.path.basename(image_path)
        is_bw = filename in bw_filenames

        # If you prefer "True"/"False" strings instead of booleans, change to:
        # entry["Black_white"] = "True" if is_bw else "False"
        entry["Black_white"] = is_bw

    save_labels_file(labels_output, entries, as_array=is_array)
    print(f"Saved updated labels with 'Black_white' field to: {labels_output}")


def main():
    # 1) Build black_and_white_images.csv and get set of filenames
    bw_filenames = load_black_and_white_filenames(INPUT_CSV)

    # 2) Add 'Black_white' label to All_labels_2.jsonl based on filename
    add_black_white_label_to_labels(LABELS_INPUT, LABELS_OUTPUT, bw_filenames)


if __name__ == "__main__":
    main()