"""
Batch SAM3 Animal Detection Processor
--------------------------------------
Reads all images recursively from SOURCE_DIR, runs SAM3 inference on each,
saves the best-detected crop (or original if no detection) to the mirrored
path in OUTPUT_DIR, and logs any failures to failed_images.txt.

Usage:
    python batch_sam3_process.py

Run from:   /home/hice1/ssinha348/sam3
            /home/hice1/ssinha348/scratch/miniconda3/envs/sam3/bin/python /home/hice1/ssinha348/scratch/codes/Sam3Tesst_3.py
"""

import torch
import time
import os
import shutil
import traceback
from pathlib import Path
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ─────────────────────────────────────────────
# CONFIG  – adjust if needed
# ─────────────────────────────────────────────
SOURCE_DIR = Path(
    "/storage/ice-shared/cs8903onl/Mcquire-animal-detection/"
    "stonemt_cameratrap/Camera Trap Photos/Processed_Images"
)
OUTPUT_DIR = Path(
    "/storage/ice-shared/cs8903onl/Mcquire-animal-detection/"
    "stonemt_cameratrap/Camera Trap Photos/Sam3"
)
FAILED_LOG = OUTPUT_DIR / "failed_images.txt"

# Image extensions to process (case-insensitive)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

TEXT_PROMPTS = [
    "Locate image of an animal",
    "Locate image of a bird",
    "Locate image of an insect",
    "Locate image of a rodent",
    "Locate image of a reptile",
    "Locate image of an amphibian"
]
# ─────────────────────────────────────────────


def find_all_images(root: Path) -> list[Path]:
    """Recursively find all image files under root."""
    images = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    return sorted(images)


def mirror_path(src_file: Path, src_root: Path, dst_root: Path) -> Path:
    """Return the output path that mirrors src_file's relative position."""
    relative = src_file.relative_to(src_root)
    return dst_root / relative


def process_image(image_path: Path, processor: Sam3Processor) -> tuple:
    """
    Run SAM3 inference on one image with multiple prompts.
    Tries all TEXT_PROMPTS and returns the best detection.
    Selection criteria:
    1. Find the highest score across all prompts
    2. Among detections within 0.1 of that max score, pick the largest box
    Returns: (best_crop_image, best_score, detected_flag, best_prompt_used, score_range, num_detections)
    """
    image = Image.open(image_path).convert("RGB")
    inference_state = processor.set_image(image)
    
    # Collect all detections from all prompts
    all_detections = []  # list of (score, box, prompt, box_area)
    
    for prompt in TEXT_PROMPTS:
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)
        boxes = output["boxes"]
        scores = output["scores"]
        
        for box, score in zip(boxes, scores):
            # Calculate box area
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            area = width * height
            
            all_detections.append((
                float(score),
                box,
                prompt,
                float(area)
            ))
    
    num_detections = len(all_detections)
    
    # If no detections found, return original image
    if num_detections == 0:
        return image, 0, False, "none", 0.0, 0
    
    # Calculate score range (max - min)
    all_scores = [det[0] for det in all_detections]
    max_score = max(all_scores)
    min_score = min(all_scores)
    score_range = max_score - min_score
    
    # Filter to detections within 0.25 of max score
    candidates = [
        det for det in all_detections 
        if det[0] >= max_score - 0.25
    ]
    
    # Among candidates, select the one with largest area
    best_detection = max(candidates, key=lambda det: det[3])
    best_score, best_box, best_prompt, best_area = best_detection
    
    # Crop to best detection
    x_min, y_min, x_max, y_max = map(int, best_box)
    best_crop = image.crop((x_min, y_min, x_max, y_max))
    
    return best_crop, best_score, True, best_prompt, score_range, num_detections


def main():
    total_start = time.perf_counter()

    # ── Validate source directory ──────────────────────────────────────────
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source directory not found:\n  {SOURCE_DIR}")

    # ── Create output root ─────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load model once ────────────────────────────────────────────────────
    print("Loading SAM3 model … ", end="", flush=True)
    model     = build_sam3_image_model()
    processor = Sam3Processor(model)
    print("done.")

    # ── Collect images ─────────────────────────────────────────────────────
    all_images = find_all_images(SOURCE_DIR)
    total = len(all_images)
    print(f"\nFound {total} image(s) under:\n  {SOURCE_DIR}\n")

    failed    = []   # list of (path, error_message)
    no_detect = []   # list of paths where no animal was found (saved as-is)
    results   = []   # list of (relative_path, status, score, prompt) for summary CSV
    success   = 0

    # ── Process each image ─────────────────────────────────────────────────
    for idx, src_path in enumerate(all_images, start=1):
        dst_path = mirror_path(src_path, SOURCE_DIR, OUTPUT_DIR)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        rel = src_path.relative_to(SOURCE_DIR)
        print(f"[{idx:>5}/{total}] {rel}", end="  … ", flush=True)

        img_start = time.perf_counter()
        try:
            result_img, score, detected, best_prompt, score_range, num_detections = process_image(src_path, processor)

            # Save with same format as original (PIL infers from extension)
            result_img.save(dst_path)

            elapsed = time.perf_counter() - img_start
            if detected:
                print(f"✓  score={score:.3f}  range={score_range:.3f}  dets={num_detections}  prompt='{best_prompt}'  ({elapsed:.1f}s)")
                success += 1
                results.append((str(rel), "DETECTED", f"{score:.4f}", best_prompt))
            else:
                print(f"–  no detection, saved original  ({elapsed:.1f}s)")
                no_detect.append(str(rel))
                success += 1          # not a failure, just no crop
                results.append((str(rel), "NO_DETECTION", "0.0000", "none"))

        except Exception as exc:
            elapsed = time.perf_counter() - img_start
            err_msg = traceback.format_exc()
            print(f"✗  ERROR  ({elapsed:.1f}s)")
            print(f"         {exc}")
            failed.append((str(rel), str(exc)))
            results.append((str(rel), "FAILED", "N/A", "error"))

    # ── Write failed-images log ────────────────────────────────────────────
    with open(FAILED_LOG, "w") as f:
        f.write("SAM3 Batch Processing – Failed Images\n")
        f.write("=" * 60 + "\n")
        f.write(f"Source : {SOURCE_DIR}\n")
        f.write(f"Output : {OUTPUT_DIR}\n")
        f.write(f"Total images  : {total}\n")
        f.write(f"Succeeded     : {success}\n")
        f.write(f"  └ No detection (original saved): {len(no_detect)}\n")
        f.write(f"Failed        : {len(failed)}\n")
        f.write("=" * 60 + "\n\n")

        if failed:
            f.write("FAILED FILES\n")
            f.write("-" * 60 + "\n")
            for path_str, err in failed:
                f.write(f"\n[FILE] {path_str}\n")
                f.write(f"[ERR]  {err}\n")
        else:
            f.write("No failures – all images processed successfully.\n")

        if no_detect:
            f.write("\n\nIMAGES WITH NO ANIMAL DETECTED (original saved as-is)\n")
            f.write("-" * 60 + "\n")
            for path_str in no_detect:
                f.write(f"  {path_str}\n")

    # ── Write per-image detection summary CSV ─────────────────────────────
    SUMMARY_LOG = OUTPUT_DIR / "detection_summary.csv"
    with open(SUMMARY_LOG, "w") as f:
        f.write("image_path,status,score,best_prompt\n")
        for path_str, status, score, prompt in results:
            f.write(f"{path_str},{status},{score},{prompt}\n")

    # ── Summary ────────────────────────────────────────────────────────────
    total_elapsed = (time.perf_counter() - total_start) / 60
    print("\n" + "=" * 60)
    print(f"  Total images   : {total}")
    print(f"  Succeeded      : {success}")
    print(f"    ↳ Cropped     : {success - len(no_detect)}")
    print(f"    ↳ No detect   : {len(no_detect)}  (original saved)")
    print(f"  Failed         : {len(failed)}")
    print(f"  Total time     : {total_elapsed:.1f} minutes")
    print(f"  Failed log     : {FAILED_LOG}")
    print(f"  Detection CSV  : {SUMMARY_LOG}")
    print("=" * 60)


if __name__ == "__main__":
    main()