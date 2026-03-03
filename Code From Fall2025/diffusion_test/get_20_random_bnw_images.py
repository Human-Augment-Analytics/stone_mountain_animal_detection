#!/usr/bin/env python3
import json
import random

INPUT = "All_labels_2_with_bw.jsonl"
OUTPUT = "sampled_20_mixed.jsonl"

def load_jsonl_or_array(path):
    """Load either a JSON array file or a JSONL file into a list of dicts."""
    with open(path, "r") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)  # JSON array
        else:
            return [json.loads(line) for line in f if line.strip()]  # JSONL

def is_black_white(entry):
    """Robustly interpret the Black_white field as a boolean."""
    val = entry.get("Black_white")
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    return str(val).strip().lower() in ("true", "1", "yes")

def main():
    entries = load_jsonl_or_array(INPUT)

    # Shuffle so we get a random-ish sample instead of always the first ones
    random.shuffle(entries)

    selected_bw = []         # 10 black & white
    selected_non_bw = []     # 5 non–black & white
    selected_extra = []      # 5 extra with new CommonName
    used_common_names = set()

    # -------- 1) Pick 10 black & white with unique CommonName --------
    for e in entries:
        if not is_black_white(e):
            continue

        common_name = e.get("CommonName")
        if not common_name:
            continue

        if common_name in used_common_names:
            continue

        selected_bw.append(e)
        used_common_names.add(common_name)

        if len(selected_bw) == 10:
            break

    print(f"Selected {len(selected_bw)} black & white entries.")

    # -------- 2) Pick 5 non–black & white with unique CommonName --------
    for e in entries:
        if is_black_white(e):
            continue

        common_name = e.get("CommonName")
        if not common_name:
            continue

        if common_name in used_common_names:
            continue

        selected_non_bw.append(e)
        used_common_names.add(common_name)

        if len(selected_non_bw) == 5:
            break

    print(f"Selected {len(selected_non_bw)} non–black & white entries.")

    # -------- 3) Pick 5 extra with new CommonName (any Black_white) --------
    for e in entries:
        common_name = e.get("CommonName")
        if not common_name:
            continue

        if common_name in used_common_names:
            continue

        selected_extra.append(e)
        used_common_names.add(common_name)

        if len(selected_extra) == 5:
            break

    print(f"Selected {len(selected_extra)} extra entries with new CommonName.")

    # Combine all selected entries
    all_selected = selected_bw + selected_non_bw + selected_extra

    # Save as JSONL for easy downstream use
    with open(OUTPUT, "w") as f:
        for e in all_selected:
            f.write(json.dumps(e) + "\n")

    print(f"Total selected: {len(all_selected)} (saved to {OUTPUT})")

if __name__ == "__main__":
    main()