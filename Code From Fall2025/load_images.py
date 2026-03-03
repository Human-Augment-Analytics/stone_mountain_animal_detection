from pathlib import Path
from PIL import Image

# use the storage path (or os.environ["SCRATCH"])
DATA_ROOT = "/storage/ice1/1/8/kpanchal30/stone mt camera full/ProjectInfo/Best Photos"
VALID_EXTS = {".jpg", ".jpeg", ".png"}  # lower-case set

def load_images(root):
    root = Path(root)
    if not root.exists():
        print(f"Path does not exist: {root}")
        return []

    img_paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS]
    print(f"Found {len(img_paths)} images")

    images = []
    for p in img_paths:
        try:
            with Image.open(p) as im:
                img = im.convert("RGB")
            images.append((p, img))
            print(f"Loaded {p} | size={img.size}")
        except Exception as e:
            print(f"Failed to load {p}: {e}")
    return images

if __name__ == "__main__":
    imgs = load_images(DATA_ROOT)
    print(f"\nTotal successfully loaded: {len(imgs)}")