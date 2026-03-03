from .base import BaseModel

# NOTE: The HF artifact may need a specific wrapper; YOLOv7 often uses Ultralytics/Darknet forks.
# If you prefer, switch to YOLOv5/8 which have simpler Python APIs.
class YOLOv7Model(BaseModel):
    def __init__(self, model_path, settings):
        super().__init__(model_path, settings)
        self.target_classes = set(settings.get("target_classes", []))
        # TODO: load detector weights from model_path, set device

    def predict(self, img_path, pil_image):
        # TODO: run detection -> filter by target_classes if provided
        # Derive Number as sum of detections (or per-class counts)
        return {
            "common_name": "",
            "species": ";".join(sorted(self.target_classes)) if self.target_classes else "",
            "number": "",
            "same_individual": "",
            "sex": "",
            "notes": "yolov7:stub",
            "best_photo": "",
        }