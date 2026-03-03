from .base import BaseModel

# TODO: from torchvision import models, transforms
# TODO: class index mapping for ImageNet or your custom classes

class ResNet50Model(BaseModel):
    def __init__(self, model_path, settings):
        super().__init__(model_path, settings)
        # TODO: load weights from self.model_path (torchvision or HF), set eval(), device, transforms

    def predict(self, img_path, pil_image):
        # TODO: preprocess -> forward -> top-1 label
        label = ""  # placeholder
        return {
            "common_name": label,
            "species": label,
            "number": "",
            "same_individual": "",
            "sex": "",
            "notes": "resnet50:stub",
            "best_photo": "",
        }