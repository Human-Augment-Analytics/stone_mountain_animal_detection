"""
Batch SAM3 Animal Detection Processor (JSON-based) - Bounding Box Version
--------------------------------------------------------------------------
Reads image paths from a JSON file, runs SAM3 inference on each,
draws RED bounding boxes for all detections above threshold (0.1),
saves annotated image to mirrored path in OUTPUT_DIR, and logs
detailed processing information.

Usage:
    python batch_sam3_process_from_json_bbox.py

Run from:   /home/hice1/ssinha348/sam3
            /home/hice1/ssinha348/scratch/miniconda3/envs/sam3/bin/python /home/hice1/ssinha348/scratch/codes/Sam3Test_5.py
"""

import torch
import time
import os
import json
import shutil
import traceback
from pathlib import Path
from PIL import Image, ImageDraw

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ─────────────────────────────────────────────
# CONFIG  – adjust if needed
# ─────────────────────────────────────────────
JSON_PATH = Path("/storage/ice-shared/cs8903onl/Mcquire-animal-detection/stonemt_cameratrap/Camera Trap Photos/Sam3/split_train_80_dynamic_burst_with_sam3_v3.json")
# JSON_PATH = Path("/storage/ice-shared/cs8903onl/Mcquire-animal-detection/stonemt_cameratrap/Camera Trap Photos/Sam3/test.json")
SOURCE_DIR = Path(
    "/storage/ice-shared/cs8903onl/Mcquire-animal-detection/"
    "stonemt_cameratrap/Camera Trap Photos/Processed_Images"
)
OUTPUT_DIR = Path(
    "/storage/ice-shared/cs8903onl/Mcquire-animal-detection/"
    "stonemt_cameratrap/Camera Trap Photos/Sam3_BBox/"
)
FAILED_LOG = OUTPUT_DIR / "failed_images.txt"
PROCESSING_LOG = OUTPUT_DIR / "processing_log.csv"

DETECTION_THRESHOLD = 0.1  # Only draw boxes for detections above this score
BOX_COLOR = "red"
BOX_WIDTH = 8

TEXT_PROMPTS = [
    "Locate image of an animal",
    "Locate image of a bird",
    "Locate image of an insect",
    "Locate image of a rodent",
    "Locate image of a reptile",
    "Locate image of an amphibian",
    "Locate image of a mammal",
    "Locate image of a species",
    "Locate image of a creature",
    "Locate image of a squirrel",
    "Locate image of an opossum",
    "Locate image of a cricetidae",
    "Locate image of a deer"
]
# ─────────────────────────────────────────────


def load_image_groups_from_json(json_path: Path) -> list[dict]:
    """
    Load image groups from JSON file.
    Expected JSON structure: list of split groups, each containing:
    - split_group_id
    - correct_label
    - night_time
    - location
    - frames: list of frame objects with file_path
    
    Returns: list of dicts, each containing:
    - group_id: str
    - label: str
    - night_time: str
    - location: str
    - image_paths: list of Path objects
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON to be a list, got {type(data)}")
    
    image_groups = []
    total_images = 0
    missing_paths = []
    
    for idx, group in enumerate(data):
        if not isinstance(group, dict):
            print(f"Warning: Group {idx} is not a dictionary, skipping")
            continue
        
        if 'frames' not in group:
            print(f"Warning: Group {idx} missing 'frames' field, skipping")
            continue
        
        group_id = group.get('split_group_id', f'group_{idx}')
        label = group.get('correct_label', 'unknown')
        night_time = group.get('night_time', 'unknown')
        location = group.get('location', 'unknown')
        
        # Extract image paths from frames
        image_paths = []
        for frame in group['frames']:
            if not isinstance(frame, dict):
                continue
            
            if 'file_path' not in frame:
                continue
            
            file_path = Path(frame['file_path'])
            
            if file_path.exists() and file_path.is_file():
                image_paths.append(file_path)
                total_images += 1
            else:
                missing_paths.append(str(file_path))
        
        if image_paths:  # Only add group if it has valid images
            image_groups.append({
                'group_id': group_id,
                'label': label,
                'night_time': night_time,
                'location': location,
                'image_paths': image_paths
            })
    
    if missing_paths:
        print(f"Warning: {len(missing_paths)} paths in JSON do not exist or are not files")
        print(f"First few missing: {missing_paths[:3]}")
    
    print(f"Loaded {len(image_groups)} groups with {total_images} total images from JSON")
    return image_groups


def mirror_path(src_file: Path, src_root: Path, dst_root: Path) -> Path:
    """Return the output path that mirrors src_file's relative position."""
    relative = src_file.relative_to(src_root)
    return dst_root / relative


def process_image_group(image_paths: list[Path], processor: Sam3Processor, group_id: str) -> tuple:
    """
    Run SAM3 inference on all images in a group.
    Collects detections from ALL images, then applies all detections above threshold
    to ALL images in the group.
    
    Returns: (annotated_images, detection_info_dict)
    where detection_info_dict contains:
        - detected: bool (any detections above threshold across all images)
        - num_detections: int (total unique detections above threshold)
        - max_score: float
        - min_score: float
        - avg_score: float
        - boxes_info: list of dict (score, prompt, box coords, source_image)
        - per_image_stats: dict mapping image path to stats
    """
    # Step 1: Collect all detections from all images in the group
    all_detections = []  # list of (score, box, prompt, source_image_path)
    per_image_stats = {}
    
    print(f"      Analyzing {len(image_paths)} images in group...", end=" ", flush=True)
    
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        inference_state = processor.set_image(image)
        
        image_detections = []
        
        for prompt in TEXT_PROMPTS:
            output = processor.set_text_prompt(state=inference_state, prompt=prompt)
            boxes = output["boxes"]
            scores = output["scores"]
            
            for box, score in zip(boxes, scores):
                score_val = float(score)
                if score_val >= DETECTION_THRESHOLD:
                    all_detections.append((score_val, box, prompt, img_path))
                    image_detections.append((score_val, box, prompt))
        
        # Store per-image stats
        if image_detections:
            img_scores = [d[0] for d in image_detections]
            per_image_stats[str(img_path)] = {
                'num_detections': len(image_detections),
                'max_score': max(img_scores),
                'avg_score': sum(img_scores) / len(img_scores)
            }
        else:
            per_image_stats[str(img_path)] = {
                'num_detections': 0,
                'max_score': 0.0,
                'avg_score': 0.0
            }
    
    print(f"found {len(all_detections)} detection(s)")
    
    # Prepare detection info
    detection_info = {
        'detected': len(all_detections) > 0,
        'num_detections': len(all_detections),
        'max_score': 0.0,
        'min_score': 0.0,
        'avg_score': 0.0,
        'boxes_info': [],
        'per_image_stats': per_image_stats
    }
    
    if len(all_detections) > 0:
        scores = [det[0] for det in all_detections]
        detection_info['max_score'] = max(scores)
        detection_info['min_score'] = min(scores)
        detection_info['avg_score'] = sum(scores) / len(scores)
        
        # Store box info
        for score, box, prompt, source_img in all_detections:
            x_min, y_min, x_max, y_max = map(int, box)
            detection_info['boxes_info'].append({
                'score': score,
                'prompt': prompt,
                'box': [x_min, y_min, x_max, y_max],
                'source_image': str(source_img)
            })
    
    # Step 2: Apply ALL detections to ALL images in the group
    print(f"      Drawing bounding boxes on all images...", end=" ", flush=True)
    annotated_images = []
    
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        
        if len(all_detections) > 0:
            draw = ImageDraw.Draw(image)
            
            for score, box, prompt, source_img in all_detections:
                x_min, y_min, x_max, y_max = map(int, box)
                
                # Draw rectangle
                draw.rectangle(
                    [(x_min, y_min), (x_max, y_max)],
                    outline=BOX_COLOR,
                    width=BOX_WIDTH
                )
        
        annotated_images.append((img_path, image))
    
    print("done")
    
    return annotated_images, detection_info


def process_image(image_path: Path, processor: Sam3Processor) -> tuple:
    """
    DEPRECATED: This function is kept for backward compatibility but is not used
    in the new group-based processing workflow.
    """
    pass


def main():
    total_start = time.perf_counter()

    # ── Validate JSON file ──────────────────────────────────────────────────
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"JSON file not found:\n  {JSON_PATH}")

    # ── Create output root ─────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load model once ────────────────────────────────────────────────────
    print("Loading SAM3 model … ", end="", flush=True)
    model     = build_sam3_image_model()
    processor = Sam3Processor(model)
    print("done.")

    # ── Load image groups from JSON ────────────────────────────────────────
    print(f"\nLoading image groups from:\n  {JSON_PATH}")
    image_groups = load_image_groups_from_json(JSON_PATH)
    total_groups = len(image_groups)
    total_images = sum(len(group['image_paths']) for group in image_groups)
    print(f"Found {total_groups} group(s) with {total_images} total image(s)\n")

    if total_groups == 0:
        print("No valid image groups found in JSON. Exiting.")
        return

    failed_groups = []   # list of (group_id, error_message)
    no_detect_groups = []   # list of group_ids where no animal was found
    detected_groups = []   # list of group_ids where detections were found
    
    total_images_processed = 0
    total_images_failed = 0
    
    # ── Open processing log CSV ────────────────────────────────────────────
    log_file = open(PROCESSING_LOG, "w")
    log_file.write("group_id,label,night_time,location,image_path,status,num_detections_in_group,max_score,min_score,avg_score,processing_time_sec,error_message\n")

    # ── Process each image group ───────────────────────────────────────────
    for idx, group in enumerate(image_groups, start=1):
        group_id = group['group_id']
        label = group['label']
        night_time = group['night_time']
        location = group['location']
        image_paths = group['image_paths']
        
        print(f"\n[{idx:>5}/{total_groups}] Group: {group_id}")
        print(f"      Label: {label} | Night: {night_time} | Location: {location} | Images: {len(image_paths)}")

        group_start = time.perf_counter()
        status = "UNKNOWN"
        num_dets = 0
        max_score = 0.0
        min_score = 0.0
        avg_score = 0.0
        error_msg = ""
        
        try:
            # Process the entire group
            annotated_images, detection_info = process_image_group(
                image_paths, processor, group_id
            )

            # Save all annotated images
            for img_path, annotated_img in annotated_images:
                dst_path = mirror_path(img_path, SOURCE_DIR, OUTPUT_DIR)
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                annotated_img.save(dst_path)

            elapsed = time.perf_counter() - group_start
            
            if detection_info['detected']:
                num_dets = detection_info['num_detections']
                max_score = detection_info['max_score']
                min_score = detection_info['min_score']
                avg_score = detection_info['avg_score']
                status = "DETECTED"
                detected_groups.append(group_id)
                
                print(f"      ✓ {num_dets} detection(s) across group | max={max_score:.3f} avg={avg_score:.3f} | ({elapsed:.1f}s)")
            else:
                status = "NO_DETECTION"
                no_detect_groups.append(group_id)
                print(f"      – No detections above threshold | ({elapsed:.1f}s)")
            
            # Write log entry for each image in the group
            for img_path in image_paths:
                rel = img_path.relative_to(SOURCE_DIR)
                log_file.write(f"{group_id},{label},{night_time},{location},{rel},{status},{num_dets},{max_score:.4f},{min_score:.4f},{avg_score:.4f},{elapsed:.2f},\n")
                total_images_processed += 1

        except Exception as exc:
            elapsed = time.perf_counter() - group_start
            error_msg = str(exc).replace(',', ';')  # Replace commas for CSV
            err_trace = traceback.format_exc()
            status = "FAILED"
            
            print(f"      ✗ ERROR | ({elapsed:.1f}s)")
            print(f"         {exc}")
            
            failed_groups.append((group_id, str(exc)))
            
            # Write log entry for each image in the failed group
            for img_path in image_paths:
                rel = img_path.relative_to(SOURCE_DIR)
                log_file.write(f"{group_id},{label},{night_time},{location},{rel},{status},{num_dets},{max_score:.4f},{min_score:.4f},{avg_score:.4f},{elapsed:.2f},{error_msg}\n")
                total_images_failed += 1

    # ── Close processing log ───────────────────────────────────────────────
    log_file.close()

    # ── Write failed-groups log ────────────────────────────────────────────
    with open(FAILED_LOG, "w") as f:
        f.write("SAM3 Batch Processing – Group-based Bounding Box Version\n")
        f.write("=" * 60 + "\n")
        f.write(f"JSON file          : {JSON_PATH}\n")
        f.write(f"Source             : {SOURCE_DIR}\n")
        f.write(f"Output             : {OUTPUT_DIR}\n")
        f.write(f"Detection threshold: {DETECTION_THRESHOLD}\n")
        f.write(f"Box color          : {BOX_COLOR}\n")
        f.write(f"Box width          : {BOX_WIDTH}\n")
        f.write(f"Total groups       : {total_groups}\n")
        f.write(f"Total images       : {total_images}\n")
        f.write(f"Groups w/ detections: {len(detected_groups)}\n")
        f.write(f"Groups w/o detections: {len(no_detect_groups)}\n")
        f.write(f"Failed groups      : {len(failed_groups)}\n")
        f.write("=" * 60 + "\n\n")

        if failed_groups:
            f.write("FAILED GROUPS\n")
            f.write("-" * 60 + "\n")
            for group_id, err in failed_groups:
                f.write(f"\n[GROUP] {group_id}\n")
                f.write(f"[ERR]   {err}\n")
        else:
            f.write("No failures – all groups processed successfully.\n")

        if no_detect_groups:
            f.write(f"\n\nGROUPS WITH NO DETECTIONS ABOVE THRESHOLD ({DETECTION_THRESHOLD})\n")
            f.write("-" * 60 + "\n")
            for group_id in no_detect_groups:
                f.write(f"  {group_id}\n")

    # ── Summary ────────────────────────────────────────────────────────────
    total_elapsed = (time.perf_counter() - total_start) / 60
    success_groups = len(detected_groups) + len(no_detect_groups)
    
    print("\n" + "=" * 60)
    print(f"  JSON file           : {JSON_PATH}")
    print(f"  Total groups        : {total_groups}")
    print(f"  Total images        : {total_images}")
    print(f"  Groups processed OK : {success_groups}")
    print(f"    ↳ With detections  : {len(detected_groups)}")
    print(f"    ↳ No detections    : {len(no_detect_groups)}")
    print(f"  Failed groups       : {len(failed_groups)}")
    print(f"  Images processed    : {total_images_processed}")
    print(f"  Images failed       : {total_images_failed}")
    print(f"  Detection threshold : {DETECTION_THRESHOLD}")
    print(f"  Total time          : {total_elapsed:.1f} minutes")
    print(f"  Failed log          : {FAILED_LOG}")
    print(f"  Processing log      : {PROCESSING_LOG}")
    print("=" * 60)


if __name__ == "__main__":
    main()