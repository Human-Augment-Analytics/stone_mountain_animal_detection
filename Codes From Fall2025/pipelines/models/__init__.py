from .baseline import BaselineModel
from .resnet50 import ResNet50Model
from .clip_zeroshot import CLIPZeroShotModel
from .yolo_v7 import YOLOv7Model
from .sam_vit_b import SAMViTBModel
from .grounding_dino_tiny import GroundingDinoTinyModel

def make_model(model_cfg: dict):
    name = (model_cfg.get("name") or "baseline").lower()
    paths = model_cfg.get("paths", {})
    settings = model_cfg.get("settings", {})

    if name == "resnet50":
        return ResNet50Model(paths.get("resnet50", ""), settings)
    if name == "clip":
        return CLIPZeroShotModel(paths.get("clip", ""), settings)
    if name == "yolov7":
        return YOLOv7Model(paths.get("yolov7", ""), settings)
    if name == "sam_vit_b":
        return SAMViTBModel(paths.get("sam_vit_b", ""), settings)
    if name == "grounding_dino_tiny":
        return GroundingDinoTinyModel(paths.get("grounding_dino_tiny", ""), settings)
    # default
    return BaselineModel("", settings)