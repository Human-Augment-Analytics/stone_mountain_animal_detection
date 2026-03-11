# # /home/hice1/ssinha348/scratch/miniconda3/envs/sam3/bin/python /home/hice1/ssinha348/scratch/codes/Sam3Test_2.py
# run in location /home/hice1/ssinha348/sam3
# from huggingface_hub import login

# # Login with your token
# login(token="....")

import torch
import time
start_time = time.perf_counter()
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)

# Load an image
image = Image.open("/home/hice1/ssinha348/scratch/stonemt_cameratrap/Camera Trap Photos/Processed_Images/SM_1/20220505/SM_1_IMG_0001_20220505_101700__000170.JPG")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="Locate image of an animal")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

# print(boxes)

print(f"Found {len(boxes)} detections")
print(f"Boxes: {boxes}")
print(f"Scores: {scores}")

# # Crop the image based on each bounding box
# for idx, (box, score) in enumerate(zip(boxes, scores)):
#     # Boxes are typically in format [x_min, y_min, x_max, y_max]
#     x_min, y_min, x_max, y_max = box
    
#     # Convert to integers
#     x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    
#     # Crop the image
#     cropped_image = image.crop((x_min, y_min, x_max, y_max))
    
#     # Save the cropped image
#     output_path = f"/home/hice1/ssinha348/scratch/animal_crop_{idx}_score_{score:.2f}.jpg"
#     cropped_image.save(output_path)
#     print(f"Saved crop {idx} to {output_path} (score: {score:.2f})")

# # Optional: Crop only the highest scoring detection
if len(boxes) > 0:
    best_idx = scores.argmax()
    best_box = boxes[best_idx]
    x_min, y_min, x_max, y_max = map(int, best_box)
    
    best_crop = image.crop((x_min, y_min, x_max, y_max))
    best_crop.save("/home/hice1/ssinha348/scratch/Testing/1.jpg")
    print(f"Saved best detection (score: {scores[best_idx]:.2f})")

# Record the end time
end_time = time.perf_counter()

# Calculate the elapsed time
elapsed_time = (end_time - start_time)/60

# Print the time difference
print(f"Elapsed time: {elapsed_time:.0f} minutes")