import os
import argparse

from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionUpscalePipeline


def extract_bbox_from_overlay(img, sat_threshold=80, val_threshold=40, padding=10):
    """
    Detect a bounding box around the overlay/mask region, regardless of color.

    Strategy:
      - Convert to HSV
      - Use high saturation (and reasonable value) to detect colored overlay pixels
      - Compute min/max x,y of those pixels as the bounding box
      - Add a small padding around the box

    Returns (left, top, right, bottom) in image coordinates,
    or None if no overlay region is found.
    """
    hsv = np.array(img.convert("HSV"))
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # High saturation and not too dark → likely overlay/mask
    mask = (s > sat_threshold) & (v > val_threshold)

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None

    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()

    # Add padding, clamped to image bounds
    height, width = s.shape
    left = max(0, xmin - padding)
    top = max(0, ymin - padding)
    right = min(width, xmax + padding)
    bottom = min(height, ymax + padding)

    if right <= left or bottom <= top:
        return None

    return left, top, right, bottom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir",
        type=str,
        default="/home/hice1/kpanchal30/scratch/stone mt camera full/Camera Trap Photos/Processed_Images/SAM3_Results/",
        help="Directory containing SAM3 result images with overlay + bbox",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="stabilityai/stable-diffusion-x4-upscaler",
        help="Diffusers upscaler model ID",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a high-resolution wildlife photo, natural colors, detailed fur or feathers",
        help="Text prompt for the upscaler",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on: 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=-1,
        help="Optional cap on number of images to process (-1 = all)",
    )
    parser.add_argument(
        "--sat-threshold",
        type=int,
        default=80,
        help="Saturation threshold for overlay detection (0–255). Lower if overlay is dull.",
    )
    parser.add_argument(
        "--val-threshold",
        type=int,
        default=40,
        help="Value/brightness threshold for overlay detection (0–255).",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Extra pixels of padding around the detected bbox.",
    )

    args = parser.parse_args()

    image_dir = args.image_dir
    files = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if args.max_images > 0:
        files = files[: args.max_images]

    print(f"Found {len(files)} images in {image_dir}")

    # Load upscaler pipeline
    print(f"Loading model: {args.model_id}")
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
    )
    pipe = pipe.to(args.device)
    pipe.enable_attention_slicing()

    # Optional: disable safety checker if it flags benign wildlife images
    # pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))

    for idx, filename in enumerate(files, start=1):
        img_path = os.path.join(image_dir, filename)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[{idx}] Skipping {filename}: cannot open ({e})")
            continue

        bbox = extract_bbox_from_overlay(
            img,
            sat_threshold=args.sat_threshold,
            val_threshold=args.val_threshold,
            padding=args.padding,
        )

        if bbox is None:
            print(f"[{idx}] Skipping {filename}: no overlay region detected")
            continue

        left, top, right, bottom = bbox
        cropped = img.crop((left, top, right, bottom))

        base_name, _ = os.path.splitext(filename)
        pre_name = f"pre_diffuser_{base_name}.jpg"
        post_name = f"post_diffuser_{base_name}.jpg"

        pre_path = os.path.join(image_dir, pre_name)
        post_path = os.path.join(image_dir, post_name)

        # Save cropped "before" image
        try:
            cropped.save(pre_path, "JPEG")
        except Exception as e:
            print(f"[{idx}] Error saving pre image for {filename}: {e}")
            continue

        # Run diffuser upscaler
        try:
            autocast_device = "cuda" if args.device == "cuda" else "cpu"
            with torch.autocast(autocast_device):
                result = pipe(
                    prompt=args.prompt,
                    image=cropped,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                )
            upscaled = result.images[0]
            upscaled.save(post_path, "JPEG")
            print(f"[{idx}] Processed {filename} -> {pre_name}, {post_name}")
        except Exception as e:
            print(f"[{idx}] Error running diffuser on {filename}: {e}")


if __name__ == "__main__":
    main()