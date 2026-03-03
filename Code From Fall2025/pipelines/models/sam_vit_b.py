from .base import BaseModel

# SAM is for segmentation; you might count instances or export masks.
class SAMViTBModel(BaseModel):
    def __init__(self, model_path, settings):
        super().__init__(model_path, settings)
        # TODO: load SAM from model_path; decide prompt mode (everything vs point/box)

    def predict(self, img_path, pil_image):
        # TODO: produce masks; optionally count mask instances
        return {
            "common_name": "",
            "species": "",
            "number": "",
            "same_individual": "",
            "sex": "",
            "notes": "sam:stub",
            "best_photo": "",
        }