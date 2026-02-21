#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Llama-3.2-Vision-Instruct animal detection over nested camera-trap folders and save CSV results."
    )
    parser.add_argument(
        "--input-dir",
        default="/storage/ice-shared/cs8903onl/Mcquire-animal-detection/stonemt_cameratrap/Camera Trap Photos/Processed_Images",
        help="Root folder to recursively scan for images.",
    )
    parser.add_argument(
        "--model-dir",
        default="/storage/ice-shared/cs8903onl/Mcquire-animal-detection/VLM/Llama/Llama-3.2-11B-Vision-Instruct",
        help="Local path to Llama-3.2 Vision Instruct model.",
    )
    parser.add_argument(
        "--output-csv",
        default="/storage/ice-shared/cs8903onl/Mcquire-animal-detection/stonemt_cameratrap/Camera Trap Photos/Processed_Images/llama_animal_detection_results.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum tokens generated per image.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on (e.g., cuda, cuda:0, cpu). Auto-detect when omitted.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output CSV by skipping already processed file paths.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="When resuming, re-run rows that previously have ERROR results.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split input images into this many shards for parallel jobs.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="0-based shard index for this process (must be < num-shards).",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=10,
        help="Flush CSV to disk every N processed images.",
    )
    return parser.parse_args()


def find_images(root: Path) -> list[Path]:
    images = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if Path(name).suffix.lower() in IMAGE_EXTS:
                images.append(Path(dirpath) / name)
    return sorted(images)


def load_completed(output_csv: Path, retry_errors: bool) -> set[str]:
    completed: set[str] = set()
    if not output_csv.exists():
        return completed
    with output_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_path = (row.get("file_path") or "").strip()
            result = (row.get("detection_result") or "").strip()
            if not file_path:
                continue
            if retry_errors and result.startswith("ERROR:"):
                continue
            completed.add(file_path)
    return completed


def build_prompt() -> str:
    # EXACT SAME PROMPT AS QWEN SCRIPT (do not change for fair comparison)
    return (
        "What animal can you see? Please give me the name of it. "
        "If there is no animal, say 'no'. "
        "If there is more than one animal, give the names of both animals you see. "
        "Return only the animal name(s) or 'no'. No explanation."
    )


def run_inference(
    model,
    processor: AutoProcessor,
    image_path: Path,
    max_new_tokens: int,
    device: str,
) -> str:
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_prompt()},
            ],
        }
    ]

    # EXACT SAME CHAT TEMPLATE FLOW AS QWEN SCRIPT
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )

    # EXACT SAME DECODE/SLICING LOGIC AS QWEN SCRIPT
    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # Keep a compact single-field CSV value.
    return " ".join(result.split())


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    model_dir = Path(args.model_dir)
    output_csv = Path(args.output_csv)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, num-shards)")

    images = find_images(input_dir)
    if not images:
        print(f"No images found under: {input_dir}")
        return

    # EXACT SAME SHARDING METHOD AS QWEN SCRIPT
    images = [p for i, p in enumerate(images) if i % args.num_shards == args.shard_index]
    if not images:
        print(
            f"No images assigned to shard {args.shard_index}/{args.num_shards} under: {input_dir}"
        )
        return

    # EXACT SAME DEVICE + DTYPE LOGIC AS QWEN SCRIPT (including cuda vs cuda:0 behavior)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading model from: {model_dir}")
    print(f"Using device: {device}")

    # Model swap ONLY: Llama vision model loader via AutoModelForVision2Seq
    model = AutoModelForVision2Seq.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    # Keep same style as your Qwen script
    processor = AutoProcessor.from_pretrained(str(model_dir), use_fast=False)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(output_csv, args.retry_errors) if args.resume else set()
    if completed:
        images = [p for p in images if str(p) not in completed]

    print(f"Shard {args.shard_index}/{args.num_shards} has {len(images)} pending images.")
    print(f"Writing CSV to: {output_csv}")

    mode = "a" if args.resume and output_csv.exists() else "w"
    with output_csv.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if mode == "w":
            writer.writerow(["file_path", "detection_result"])

        for idx, image_path in enumerate(images, start=1):
            try:
                result = run_inference(
                    model=model,
                    processor=processor,
                    image_path=image_path,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                )
            except Exception as exc:
                result = f"ERROR: {type(exc).__name__}: {exc}"

            writer.writerow([str(image_path), result])
            if args.flush_every > 0 and (idx % args.flush_every == 0):
                f.flush()
            if idx % 50 == 0 or idx == len(images):
                print(f"Processed {idx}/{len(images)}")

    print("Done.")


if __name__ == "__main__":
    main()