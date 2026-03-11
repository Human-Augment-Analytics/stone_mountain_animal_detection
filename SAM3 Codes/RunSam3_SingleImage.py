from pathlib import Path
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load model
print("Loading SAM3 model...", flush=True)
model = build_sam3_image_model()
processor = Sam3Processor(model)
print("Model loaded.\n")

# Test on single image
image_path = "/storage/ice-shared/cs8903onl/Mcquire-animal-detection/stonemt_cameratrap/Camera Trap Photos/Processed_Images/SM_2/20221122/SM_2_IMG_0010_20221122_140600__035001.JPG"
image = Image.open(image_path).convert("RGB")

TEXT_PROMPTS = [
    "Locate image of an animal",
    "Locate image of a bird",
    "Locate image of an insect",
    "Locate image of a rodent",
    "Locate image of a reptile",
    "Locate image of an amphibian",
    "Locate image of an Species",
    "Locate image of an Creature",
    "Locate image of deer"
]

inference_state = processor.set_image(image)
all_detections = []

for prompt in TEXT_PROMPTS:
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
    boxes = output["boxes"]
    scores = output["scores"]
    print(f"{prompt}: {len(boxes)} detections")
    for box, score in zip(boxes, scores):
        x_min, y_min, x_max, y_max = box
        area = (x_max - x_min) * (y_max - y_min)
        all_detections.append((float(score), box, prompt, float(area)))
        print(f"  - score={score:.4f}, area={area:.0f}")

if all_detections:
    max_score = max(d[0] for d in all_detections)
    min_score = min(d[0] for d in all_detections)
    candidates = [d for d in all_detections if d[0] >= max_score - 0.1]
    best = max(candidates, key=lambda d: d[3])
    
    print(f"\n{'='*60}")
    print(f"Total detections: {len(all_detections)}")
    print(f"Score range: {max_score:.4f} - {min_score:.4f} = {max_score - min_score:.4f}")
    print(f"Candidates within 0.1 of max: {len(candidates)}")
    print(f"Best: score={best[0]:.4f}, area={best[3]:.0f}, prompt='{best[2]}'")
    print(f"{'='*60}")
    
    x_min, y_min, x_max, y_max = map(int, best[1])
    crop = image.crop((x_min, y_min, x_max, y_max))
    crop.save("test_output.jpg")
    print("Saved cropped image to: test_output.jpg")
else:
    print("\nNo detections found")