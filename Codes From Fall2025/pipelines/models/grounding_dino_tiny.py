from .base import BaseModel

# GroundingDINO: open-vocab detection with text prompts in settings["text_queries"].
class GroundingDinoTinyModel(BaseModel):
    def __init__(self, model_path, settings):
        super().__init__(model_path, settings)
        self.text_queries = settings.get("text_queries", [])
        # TODO: load model from model_path; prepare text tokenizer

    def predict(self, img_path, pil_image):
        # TODO: run detection with text queries; compute counts per query
        species = ";".join(self.text_queries) if self.text_queries else ""
        return {
            "common_name": "",
            "species": species,
            "number": "",
            "same_individual": "",
            "sex": "",
            "notes": "grounding-dino:stub",
            "best_photo": "",
        }