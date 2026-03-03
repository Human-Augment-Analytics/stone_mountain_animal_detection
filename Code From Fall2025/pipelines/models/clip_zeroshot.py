from .base import BaseModel

# TODO: import transformers or open_clip; tokenize text prompts from settings["text_queries"]

class CLIPZeroShotModel(BaseModel):
    def __init__(self, model_path, settings):
        super().__init__(model_path, settings)
        self.text_queries = settings.get("text_queries", [])
        # TODO: load CLIP model & processor from model_path (local_files_only=True)

    def predict(self, img_path, pil_image):
        # TODO: encode image + text, compute similarity, choose best label
        label = self.text_queries[0] if self.text_queries else ""
        return {
            "common_name": label,
            "species": label,
            "number": "",
            "same_individual": "",
            "sex": "",
            "notes": "clip:stub",
            "best_photo": "",
        }