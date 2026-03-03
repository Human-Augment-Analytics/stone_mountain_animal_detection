import os
import json
import random
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm


# ---------- CONFIG ----------

BASE_INPUT = Path(
    "/home/hice1/kpanchal30/scratch/stone mt camera full/Camera Trap Photos/Processed_Images"
)
BASE_OUTPUT = Path("/home/hice1/kpanchal30/scratch/stone_mt_sr_outputs")
METRICS_LOG = Path("/home/hice1/kpanchal30/scratch/stone_mt_sr_outputs/qwen_sr_metrics.jsonl")

SAMPLES_PER_BATCH = 1
RANDOM_SEED = 42


# ---------- MODEL SETUP ----------

def load_sr_model(device="cuda"):
    print("Loading StableDiffusionUpscalePipeline...", flush=True)
    sr_model_id = "stabilityai/stable-diffusion-x4-upscaler"
    sr_pipe = StableDiffusionUpscalePipeline.from_pretrained(
        sr_model_id,
        torch_dtype=torch.float16,
        local_files_only=True,  # <-- use local cache only
    ).to(device)
    print("-> SD Upscaler loaded.", flush=True)
    return sr_pipe


def load_qwen_model():
    print("Loading Qwen3-VL-32B model...", flush=True)
    qwen_id = "Qwen/Qwen3-VL-32B-Instruct"

    qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
        qwen_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,  # <-- use local cache only
    )
    print("-> Qwen model weights loaded.", flush=True)

    print("Loading Qwen processor...", flush=True)
    qwen_processor = AutoProcessor.from_pretrained(
        qwen_id,
        local_files_only=True,  # <-- use local cache only
    )
    print("-> Qwen processor loaded.", flush=True)

    return qwen_model, qwen_processor


# ---------- IMAGE DISCOVERY / SAMPLING ----------

def collect_images_per_sm(base_input: Path):
    sm_dirs = [d for d in base_input.iterdir() if d.is_dir() and d.name.startswith("SM_")]
    sm_dirs = sorted(sm_dirs)

    sm_to_images = {}
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    for sm_dir in sm_dirs:
        imgs = []
        for p in sm_dir.rglob("*"):
            if p.is_file() and p.suffix in exts:
                imgs.append(p)
        sm_to_images[sm_dir.name] = sorted(imgs)

    return sm_to_images


def sample_images(sm_to_images, samples_per_batch: int, seed: int = 42):
    random.seed(seed)
    sampled = []

    for sm_name, imgs in sm_to_images.items():
        if not imgs:
            continue
        k = min(samples_per_batch, len(imgs))
        picked = random.sample(imgs, k)
        for p in picked:
            sampled.append((sm_name, p))

    return sampled


# ---------- SUPER-RESOLUTION + QWEN METRICS ----------

def upscale_image(sr_pipe, in_path: Path, base_input: Path, base_output: Path):
    img = Image.open(in_path).convert("RGB")

    # OPTIONAL: downscale before upscaling to save memory
    # img.thumbnail((512, 512), Image.LANCZOS)

    result = sr_pipe(prompt="", image=img)
    enhanced = result.images[0]

    rel_path = in_path.relative_to(base_input)
    out_path = base_output / rel_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    enhanced.save(out_path)

    return out_path


def qwen_rate_pair(qwen_model, qwen_processor, low_path: Path, enhanced_path: Path):
    low_img = Image.open(low_path).convert("RGB")
    enh_img = Image.open(enhanced_path).convert("RGB")

    system_prompt = (
        "You are an image quality judge focusing on super-resolution. "
        "Compare Image A (low-res input) and Image B (enhanced output). "
        "Return a strict JSON object like:\n"
        '{"sharpness_gain": float, "noise_level": float, "artifact_level": float, '
        '"overall_improvement": float, "short_reason": \"...\"}\n'
        "Use scores from 1 (worst) to 10 (best)."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": low_img},
                {"type": "image", "image": enh_img},
                {
                    "type": "text",
                    "text": "Compare the first (low-res) and second (enhanced) images.",
                },
            ],
        },
    ]

    inputs = qwen_processor(messages, return_tensors="pt").to(qwen_model.device)

    with torch.no_grad():
        out_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2,
        )

    text = qwen_processor.batch_decode(out_ids, skip_special_tokens=True)[0]

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {"raw_response": text}

    try:
        parsed = json.loads(text[start: end + 1])
    except Exception:
        parsed = {"raw_response": text}

    return parsed


# ---------- MAIN PIPELINE (2 PHASES) ----------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    print(f"Scanning images under: {BASE_INPUT}", flush=True)
    sm_to_images = collect_images_per_sm(BASE_INPUT)

    for sm_name, imgs in sm_to_images.items():
        print(f"{sm_name}: found {len(imgs)} images", flush=True)

    sampled_pairs = sample_images(sm_to_images, SAMPLES_PER_BATCH, seed=RANDOM_SEED)
    print(f"Total sampled images: {len(sampled_pairs)}", flush=True)

    if not sampled_pairs:
        print("No images found to process. Exiting.")
        return

    METRICS_LOG.parent.mkdir(parents=True, exist_ok=True)
    if METRICS_LOG.exists():
        METRICS_LOG.unlink()  # start fresh

    # -------- PHASE 1: SUPER-RESOLUTION ONLY --------
    print("=== PHASE 1: Running SD Upscale on sampled images ===", flush=True)
    sr_pipe = load_sr_model(device=device)

    enhanced_paths = {}  # map from input path -> enhanced path

    for sm_name, img_path in tqdm(sampled_pairs, desc="Super-resolving", ncols=100):
        enhanced_path = upscale_image(sr_pipe, img_path, BASE_INPUT, BASE_OUTPUT)
        enhanced_paths[str(img_path)] = str(enhanced_path)

    # Free SR model from GPU
    del sr_pipe
    torch.cuda.empty_cache()
    print("Freed SD Upscaler from GPU.", flush=True)

    # -------- PHASE 2: QWEN SCORING ONLY --------
    print("=== PHASE 2: Scoring enhanced images with Qwen ===", flush=True)
    qwen_model, qwen_processor = load_qwen_model()

    with METRICS_LOG.open("w") as f_log:
        for sm_name, img_path in tqdm(sampled_pairs, desc="Scoring with Qwen", ncols=100):
            input_str = str(img_path)
            enhanced_path = enhanced_paths[input_str]

            scores = qwen_rate_pair(qwen_model, qwen_processor, img_path, Path(enhanced_path))

            record = {
                "batch": sm_name,
                "input_path": input_str,
                "enhanced_path": enhanced_path,
                "scores": scores,
            }
            f_log.write(json.dumps(record) + "\n")

    print(f"Done. Metrics written to: {METRICS_LOG}", flush=True)


if __name__ == "__main__":
    main()