from datetime import datetime
from PIL import Image, ExifTags

def extract_exif_datetime(path: str):
    """
    Returns a datetime if EXIF DateTimeOriginal or DateTime exists; otherwise None.
    """
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            if not exif:
                return None
            # map tag ids to names
            tag2name = {ExifTags.TAGS.get(k, k): k for k in exif.keys()}
            for key in ("DateTimeOriginal", "DateTime"):
                tag_id = tag2name.get(key)
                if tag_id is None:
                    continue
                raw = exif.get(tag_id)
                if not raw:
                    continue
                # common EXIF format: "YYYY:MM:DD HH:MM:SS"
                raw = str(raw)
                raw = raw.replace("-", ":")  # sometimes hyphens appear
                try:
                    return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
                except Exception:
                    pass
    except Exception:
        return None
    return None