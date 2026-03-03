from pathlib import Path
from PIL import Image

VALID_EXTS = {".jpg", ".jpeg", ".png"}

def expand_input_dirs(inputs_cfg: dict):
    out = []

    # Highest priority: explicit root_dir, if set
    root_dir = inputs_cfg.get("root_dir", "")
    if root_dir:
        p = Path(root_dir)
        if p.exists():
            out.append(str(p))

    # SM_* expansion
    if inputs_cfg.get("run_all_sm", False):
        sm_root = Path(inputs_cfg.get("sm_root", ""))
        if sm_root.exists():
            for d in sm_root.iterdir():
                if d.is_dir() and d.name.startswith("SM_"):
                    out.append(str(d))

    # Best Photos
    if inputs_cfg.get("run_best_photos", False):
        bp = Path(inputs_cfg.get("best_photos_dir", ""))
        if bp.exists():
            out.append(str(bp))

    # Deduplicate while preserving order
    seen = set()
    dedup = []
    for d in out:
        if d not in seen:
            dedup.append(d)
            seen.add(d)
    return dedup

def iter_images(root_dir: str):
    root = Path(root_dir)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            try:
                with Image.open(p) as im:
                    img = im.convert("RGB")
                yield str(p), img
            except Exception as e:
                # Could log this path; for now, skip corrupted images
                continue