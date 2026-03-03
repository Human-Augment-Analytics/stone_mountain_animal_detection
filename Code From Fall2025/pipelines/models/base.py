from typing import Any, Dict

class BaseModel:
    def __init__(self, model_path: str, settings: Dict[str, Any] | None = None):
        self.model_path = model_path
        self.settings = settings or {}

    def predict(self, img_path: str, pil_image) -> Dict[str, Any]:
        """Return a dict with fields used by run_models:
           common_name, species, number, same_individual, sex, notes, best_photo
        """
        raise NotImplementedError