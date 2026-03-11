# transformers 5.1.0 requires huggingface-hub<2.0,>=1.3.0, but you have huggingface-hub 0.36.2 which is incompatible.

#################################### Approach 1 ####################################
# https://huggingface.co/facebook/sam3
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
image = Image.open("<YOUR_IMAGE_PATH.jpg>")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="<YOUR_TEXT_PROMPT>")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]




# #################################### Approach 2 ####################################
# # https://huggingface.co/docs/transformers/en/model_doc/sam3
# from transformers import Sam3Processor, Sam3Model
# import torch
# from PIL import Image
# import requests
# from huggingface_hub import login

# # Login with your token
# login(token="....")

# device = "cuda" if torch.cuda.is_available() else "cpu"

# model = Sam3Model.from_pretrained("facebook/sam3").to(device)
# processor = Sam3Processor.from_pretrained("facebook/sam3")

# # Load image
# image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
# image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

# # Segment using text prompt
# inputs = processor(images=image, text="ear", return_tensors="pt").to(device)

# with torch.no_grad():
#     outputs = model(**inputs)

# # Post-process results
# results = processor.post_process_instance_segmentation(
#     outputs,
#     threshold=0.5,
#     mask_threshold=0.5,
#     target_sizes=inputs.get("original_sizes").tolist()
# )[0]

# print(f"Found {len(results['masks'])} objects")
# # Results contain:
# # - masks: Binary masks resized to original image size
# # - boxes: Bounding boxes in absolute pixel coordinates (xyxy format)
# # - scores: Confidence scores