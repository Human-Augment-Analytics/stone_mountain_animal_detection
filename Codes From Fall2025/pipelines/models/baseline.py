from .base import BaseModel

class BaselineModel(BaseModel):
    def predict(self, img_path: str, pil_image):
        # No ML: just a stub so the plumbing runs end-to-end.
        return {
            "common_name": "",
            "species": "",
            "number": "",
            "same_individual": "",
            "sex": "",
            "notes": "baseline:no-op",
            "best_photo": "",
        }