#!/usr/bin/env python3
import os
import csv
from PIL import Image

# -------- CONFIGURE THESE --------
ROOT_DIR = r"/home/hice1/kpanchal30/scratch/stone mt camera full/Camera Trap Photos/Processed_Images/SM_1/20250329"
OUTPUT_CSV = "sm1_20250329_image_stats.csv"
# ---------------------------------


def is_grayscale(img, tolerance=3, min_gray_fraction=0.95):
    """
    Decide if an image is effectively black & white (grayscale).

    - If mode is 'L', '1', or 'LA', treat as grayscale.
    - Otherwise convert to RGB and sample pixels.
    - If at least `min_gray_fraction` of sampled pixels have
      |R-G|, |G-B|, |R-B| <= tolerance, treat as grayscale.
    """
    if img.mode in ("1", "L", "LA"):
        return True

    rgb = img.convert("RGB")
    width, height = rgb.size

    step_x = max(1, width // 32)
    step_y = max(1, height // 32)

    total = 0
    gray_like = 0

    for y in range(0, height, step_y):
        for x in range(0, width, step_x):
            r, g, b = rgb.getpixel((x, y))
            total += 1
            if (abs(r - g) <= tolerance and
                abs(g - b) <= tolerance and
                abs(r - b) <= tolerance):
                gray_like += 1

    if total == 0:
        return False

    fraction = gray_like / total
    return fraction >= min_gray_fraction

def analyze_image(path):
    """Return (width, height, is_bw) for a given image path."""
    with Image.open(path) as img:
        width, height = img.size
        bw = is_grayscale(img)
    return width, height, bw


def main():
    rows = []

    for root, dirs, files in os.walk(ROOT_DIR):
        for fname in files:
            # Basic JPEG filter – extend if you also have .jpeg, .jpg, etc.
            if not fname.lower().endswith((".jpg", ".jpeg")):
                continue

            full_path = os.path.join(root, fname)

            try:
                width, height, bw = analyze_image(full_path)
                rows.append(
                    {
                        "full_path": full_path,
                        "filename": fname,
                        "is_black_and_white": bw,
                        "width": width,
                        "height": height,
                    }
                )
            except Exception as e:
                # If an image is corrupted or unreadable, log it in the CSV
                rows.append(
                    {
                        "full_path": full_path,
                        "filename": fname,
                        "is_black_and_white": None,
                        "width": None,
                        "height": None,
                        "error": str(e),
                    }
                )
                print(f"Error processing {full_path}: {e}")

    # Determine fieldnames dynamically so 'error' column is optional
    fieldnames = ["full_path", "filename", "is_black_and_white", "width", "height", "error"]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            # Ensure all keys exist
            for key in fieldnames:
                row.setdefault(key, None)
            writer.writerow(row)

    print(f"Done. Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
