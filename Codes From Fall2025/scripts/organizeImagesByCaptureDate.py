# !/usr/bin/env python3
#
# Organize camera trap images by capture date using EXIF metadata.
# This script will only account for JPG files under subfolders SM_1 to SM_5.
#
# 1. Place this script in the same directory as the `Camera Trap Photos` folder.
# 2. `Camera Trap Photos` folder is expected to contain subfolders `SM_1` to `SM_5`.
# 2. Install dependencies: `pip install Pillow piexif` (piexif is optional).
# 3. Run it with Python 3: `python organizeImagesByCaptureDate.py --inDir "Camera Trap Photos" --outDir "output"`
#
# Notes
#   - EXIF priority: DateTimeOriginal -> Modify Date (DateTime) -> DateTimeDigitized -> FS mtime
#   - Output structure: outDir/SM_X/MM-DD-YYYY/original_filename.jpg
#   - This script is compatible with Python 3.9 (default Python version of PACE ICE clusters)

import argparse
import concurrent.futures as futures
import os
import re
import shutil
import sys
import threading
import time
from datetime import datetime
from typing import List, Tuple, Optional, Union

# Optional EXIF helpers (Pillow first, then piexif if available)
try:
    from PIL import Image  # type: ignore
    PIL_OK = True
except Exception:
    PIL_OK = False

try:
    import piexif  # type: ignore
    PIEXIF_OK = True
except Exception:
    PIEXIF_OK = False

SM_DIR_PATTERN = re.compile(r"^SM_([1-5])$")
JPG_EXTS = {".jpg", ".jpeg"}


def ts() -> str:
    return f"[{datetime.now().strftime('%Y%m%d %H:%M:%S')}]"


def is_jpg(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in JPG_EXTS


def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_exif_to_mmddyyyy(raw: Union[str, bytes, None]) -> Optional[str]:
    """
    Convert EXIF-like date strings to 'MM/DD/YYYY'.
    Accepts:
      '2023:02:01 14:33:22'
      '2023-02-01 14:33:22'
      '2023/02/01 14:33:22'
      '2023:02:01' (date only)
    """
    if not raw:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "ignore")
    raw = raw.strip()

    date_part = raw.split()[0].replace(":", "-").replace("/", "-")
    try:
        dt = datetime.strptime(date_part, "%Y-%m-%d")
        return dt.strftime("%m/%d/%Y")
    except Exception:
        return None


def extract_capture_date_mmddyyyy(path: str) -> Tuple[Optional[str], str]:
    """
    Return (date_str 'MM/DD/YYYY', source_label) with priority:
      1) EXIF DateTimeOriginal (36867)
      2) EXIF Modify Date / DateTime (306)
      3) EXIF DateTimeDigitized (36868)
      4) FS mtime
    """
    # Pillow route
    if PIL_OK:
        try:
            with Image.open(path) as im:
                exif = im.getexif()
                if exif:
                    # 1) DateTimeOriginal
                    dto = exif.get(36867)
                    d = parse_exif_to_mmddyyyy(dto)
                    if d:
                        return d, "EXIF:DateTimeOriginal"

                    # 2) Modify Date / DateTime
                    dt = exif.get(306)
                    d = parse_exif_to_mmddyyyy(dt)
                    if d:
                        return d, "EXIF:ModifyDate"

                    # 3) DateTimeDigitized
                    dtd = exif.get(36868)
                    d = parse_exif_to_mmddyyyy(dtd)
                    if d:
                        return d, "EXIF:DateTimeDigitized"
        except Exception:
            pass

    # piexif route
    if PIEXIF_OK:
        try:
            exif_dict = piexif.load(path)
            # 1) DateTimeOriginal
            dto = exif_dict.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal)
            d = parse_exif_to_mmddyyyy(dto)
            if d:
                return d, "piexif:DateTimeOriginal"

            # 2) Modify Date / DateTime (0th IFD)
            dt = exif_dict.get("0th", {}).get(piexif.ImageIFD.DateTime)
            d = parse_exif_to_mmddyyyy(dt)
            if d:
                return d, "piexif:DateTime"

            # 3) DateTimeDigitized
            dtd = exif_dict.get("Exif", {}).get(piexif.ExifIFD.DateTimeDigitized)
            d = parse_exif_to_mmddyyyy(dtd)
            if d:
                return d, "piexif:DateTimeDigitized"
        except Exception:
            pass

    # 4) Filesystem mtime fallback
    try:
        mtime = os.path.getmtime(path)
        dt = datetime.fromtimestamp(mtime)
        return dt.strftime("%m/%d/%Y"), "FS:mtime"
    except Exception:
        return None, "unknown"


def unique_dest_path(dest_path: str) -> str:
    if not os.path.exists(dest_path):
        return dest_path
    root, ext = os.path.splitext(dest_path)
    i = 1
    while True:
        candidate = f"{root}_{i}{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1


def discover_jobs(in_dir: str) -> List[Tuple[str, str]]:
    """
    Returns a list of (abs_path, sm_top) for jpg files under SM_1..SM_5.
    sm_top is 'SM_X' (top-level SM directory name).
    """
    jobs: List[Tuple[str, str]] = []
    in_dir_abs = os.path.abspath(in_dir)

    for root, _, files in os.walk(in_dir_abs):
        rel = os.path.relpath(root, in_dir_abs)
        parts = [] if rel == "." else rel.split(os.sep)

        sm_top: Optional[str] = None
        if parts and SM_DIR_PATTERN.match(parts[0]):
            sm_top = parts[0]
        if not sm_top:
            continue

        for fn in files:
            if is_jpg(fn):
                jobs.append((os.path.join(root, fn), sm_top))
    return jobs


def main():
    parser = argparse.ArgumentParser(
        description="Copy JPGs into output/SM_X/MM/DD/YYYY/ using EXIF DateTimeOriginal first."
    )
    parser.add_argument("--inDir", required=True, help="Input root (contains SM_1..SM_5)")
    parser.add_argument("--outDir", required=True, help="Output root")
    parser.add_argument("--workers", type=int, default=2, help="Max worker threads (1-2). Default: 2")
    parser.add_argument("--dry-run", action="store_true", help="Show actions without copying")
    args = parser.parse_args()

    workers = max(1, min(2, args.workers))
    in_dir = args.inDir
    out_dir = args.outDir

    start = time.perf_counter()

    jobs = discover_jobs(in_dir)
    total = len(jobs)
    done = 0
    lock = threading.Lock()

    if total == 0:
        print(f"{ts()} No JPG files found under SM_1..SM_5 in {in_dir}")
        return

    def process_one(job: Tuple[str, str]) -> None:
        nonlocal done
        src, sm_top = job
        print(f"{ts()} found: {src}")

        date_str, source = extract_capture_date_mmddyyyy(src)
        if not date_str:
            print(f"{ts()} WARNING: could not extract date, skipping: {src}")
            with lock:
                done += 1
                pct = int(done * 100 / total)
                print(f"{ts()} progress: {done}/{total} ({pct}%)")
            return

        print(f"{ts()} metadata: {date_str} ({source})")

        # Build destination: outDir/SM_X/MM/DD/YYYY/original_filename
        mm, dd, yyyy = date_str.split("/")
        folder_name = f"{mm}-{dd}-{yyyy}"
        year_dir = os.path.join(out_dir, sm_top, folder_name)

        safe_makedirs(year_dir)
        dest = os.path.join(year_dir, os.path.basename(src))
        dest_final = unique_dest_path(dest)

        print(f"{ts()} copying to: {dest_final}")
        if not args.dry_run:
            try:
                shutil.copy2(src, dest_final)
            except Exception as e:
                print(f"{ts()} ERROR copying '{src}' -> '{dest_final}': {e}")

        with lock:
            done += 1
            pct = int(done * 100 / total)
            print(f"{ts()} progress: {done}/{total} ({pct}%)")

    print(f"{ts()} Found {total} JPG files. Starting with {workers} thread(s).")

    with futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process_one, job) for job in jobs]
        for f in futures.as_completed(futs):
            exc = f.exception()
            if exc:
                print(f"{ts()} ERROR: {exc}", file=sys.stderr)

    elapsed = time.perf_counter() - start
    hrs = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)
    secs = int(elapsed % 60)
    print(f"{ts()} DONE. Total time spent: {hrs:02d}:{mins:02d}:{secs:02d}")


if __name__ == "__main__":
    main()
