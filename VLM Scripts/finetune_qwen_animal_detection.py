#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


PROMPT = (
    "What animal can you see? Please give me the name of it. "
    "If there is no animal, say 'no'. "
    "If there is more than one animal, give the names of both animals you see. "
    "Return only the animal name(s) or 'no'. No explanation."
)


def build_user_prompt(use_night_time: bool, night_time: str) -> str:
    if not use_night_time:
        return PROMPT
    nt = "night" if str(night_time).strip().upper() == "Y" else "day"
    return f"{PROMPT} Capture time hint: {nt}."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end fine-tuning + evaluation for Qwen2.5-VL-7B-Instruct on camera-trap labels."
    )
    parser.add_argument(
        "--model-dir",
        default="/storage/ice-shared/cs8903onl/Mcquire-animal-detection/VLM/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument(
        "--labels-json",
        default="/storage/ice-shared/cs8903onl/Mcquire-animal-detection/VLM/Labels_with_paths_for_vlm_finetune_with_night_time.json",
    )
    parser.add_argument(
        "--train-json",
        default="",
        help="Optional pre-split train JSON. If set, split generation is skipped.",
    )
    parser.add_argument(
        "--validation-json",
        default="",
        help="Optional pre-split validation JSON. If set, split generation is skipped.",
    )
    parser.add_argument(
        "--test-json",
        default="",
        help="Optional pre-split unseen test JSON. If set, split generation is skipped.",
    )
    parser.add_argument(
        "--shared-root",
        default="/storage/ice-shared/cs8903onl/Mcquire-animal-detection",
        help="Dataset root used to remap label paths from other users' scratch areas.",
    )
    parser.add_argument(
        "--output-dir",
        default="/storage/ice-shared/cs8903onl/Mcquire-animal-detection/VLM/qwen_finetune_output",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (used during training).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Final unseen test split ratio (not used during training).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--num-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument(
        "--resume-from-checkpoint",
        default="auto",
        help="Checkpoint resume mode: 'auto', 'none', or a specific checkpoint path.",
    )
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--disable-night-time",
        action="store_true",
        help="Ignore `night_time` metadata even if present in labels JSON.",
    )
    parser.add_argument(
        "--time-windows",
        default="30,90",
        help="Comma-separated burst windows in seconds (e.g., 30,90).",
    )
    parser.add_argument(
        "--short-window-purity-threshold",
        type=float,
        default=0.8,
        help="When using multiple windows, use short-window target if burst purity >= threshold.",
    )
    parser.add_argument(
        "--disable-burst-consistency",
        action="store_true",
        help="Disable burst-level label consistency adjustment.",
    )
    parser.add_argument(
        "--disable-burst-group-split",
        action="store_true",
        help="Disable burst-aware split; if set, split happens at individual image level.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def norm_text(text: str) -> str:
    t = "" if text is None else str(text).lower().strip()
    t = t.replace("_", " ").replace("-", " ")
    t = re.sub(r"[`*_#'\".]+", " ", t)
    t = re.sub(r"[^a-z0-9,\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def remap_image_path(path: str, shared_root: str) -> str:
    if not path:
        return ""
    if os.path.exists(path):
        return path
    marker = "/stonemt_cameratrap/"
    idx = path.find(marker)
    if idx >= 0:
        candidate = str(Path(shared_root)) + path[idx:]
        if os.path.exists(candidate):
            return candidate
    return ""


def canonical_label(label: str) -> str:
    raw = norm_text(label)
    alias = {
        "white tailed deer": "deer",
        "eastern gray squirrel": "squirrel",
        "southern flying squirrel": "squirrel",
        "virginia opossum": "opossum",
        "canada goose": "goose",
        "barred owl": "owl",
        "great blue heron": "heron",
        "domestic dog": "dog",
        "domestic cat": "cat",
        "cooper s hawk": "hawk",
        "red shouldered hawk": "hawk",
        "american robin": "robin",
        "northern cardinal": "cardinal",
        "blue jay": "jay",
        "eastern chipmunk": "chipmunk",
        "carolina chickadee": "chickadee",
        "downy woodpecker": "woodpecker",
        "eastern bluebird": "bluebird",
        "eastern wood pewee": "wood pewee",
        "eastern whip poor will": "whip poor will",
        "house finch": "finch",
        "mourning dove": "dove",
        "northern mockingbird": "mockingbird",
        "long tailed weasel": "weasel",
        "wild boar": "boar",
        "human": "human",
        "unknown": "unknown",
    }
    return alias.get(raw, raw if raw else "unknown")


def canonical_prediction(text: str) -> list[str]:
    s = norm_text(text)
    if not s:
        return ["unknown"]
    if "no answer" in s or "no animal" in s or s in {"no", "none", "nothing"}:
        return ["no"]
    if all(tok == "no" for tok in s.split()) and s:
        return ["no"]

    parts = re.split(r"[;]| and |,|\n", s)
    cleaned = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        p = re.split(r"\b(with|based|because|in this image|provided)\b", p)[0].strip()
        label = canonical_label(p)
        if label and label not in cleaned:
            cleaned.append(label)

    if not cleaned:
        first = s.split()[0] if s.split() else "unknown"
        cleaned = [canonical_label(first)]
    return cleaned


def labels_to_target(labels: list[str]) -> str:
    labels = [canonical_label(x) for x in labels if x and canonical_label(x) != "unknown"]
    labels = sorted(set(labels))
    if not labels:
        return "unknown"
    return ", ".join(labels)


def parse_target_set(target: str) -> set[str]:
    return {canonical_label(x.strip()) for x in target.split(",") if x.strip()}


def is_correct(target: str, prediction: str) -> bool:
    tgt = parse_target_set(target)
    pred = set(canonical_prediction(prediction))
    return tgt == pred


def extract_capture_meta(path: str) -> tuple[str, str, datetime | None]:
    p = str(path)
    parts = Path(p).parts
    location = ""
    date_str = ""
    capture_dt = None

    for part in parts:
        if re.fullmatch(r"SM_\d+", part):
            location = part
        if re.fullmatch(r"\d{8}", part):
            date_str = part

    filename = Path(p).name
    m = re.search(r"_(\d{8})_(\d{6})__", filename)
    if m:
        d, t = m.group(1), m.group(2)
        if not date_str:
            date_str = d
        try:
            capture_dt = datetime.strptime(f"{d}{t}", "%Y%m%d%H%M%S")
        except ValueError:
            capture_dt = None

    return location or "unknown_location", date_str or "unknown_date", capture_dt


def choose_burst_target(targets: list[str]) -> str:
    parsed = [tuple(sorted(parse_target_set(t))) for t in targets if t]
    if not parsed:
        return "unknown"

    only_unknown = tuple(["unknown"])
    non_unknown = [x for x in parsed if x != only_unknown]
    candidates = non_unknown if non_unknown else parsed

    counts: dict[tuple[str, ...], int] = defaultdict(int)
    for c in candidates:
        counts[c] += 1

    best = sorted(counts.items(), key=lambda kv: (-kv[1], -len(kv[0]), ",".join(kv[0])))[0][0]
    return ", ".join(best)


def parse_time_windows(windows_text: str) -> list[int]:
    parts = [x.strip() for x in str(windows_text).split(",") if x.strip()]
    windows = sorted({int(x) for x in parts})
    if not windows or any(w <= 0 for w in windows):
        raise ValueError("--time-windows must be positive integers, e.g. 30,90")
    return windows


def choose_burst_target_and_purity(targets: list[str]) -> tuple[str, float]:
    parsed = [tuple(sorted(parse_target_set(t))) for t in targets if t]
    if not parsed:
        return "unknown", 0.0

    only_unknown = tuple(["unknown"])
    non_unknown = [x for x in parsed if x != only_unknown]
    candidates = non_unknown if non_unknown else parsed

    counts: dict[tuple[str, ...], int] = defaultdict(int)
    for c in candidates:
        counts[c] += 1
    best_label, best_count = sorted(
        counts.items(), key=lambda kv: (-kv[1], -len(kv[0]), ",".join(kv[0]))
    )[0]
    purity = best_count / max(1, len(candidates))
    return ", ".join(best_label), purity


def annotate_bursts_for_window(
    examples: list[dict[str, Any]],
    time_window_seconds: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    missing_time: list[dict[str, Any]] = []

    for ex in examples:
        location, date_str, capture_dt = extract_capture_meta(ex["file_path"])
        ex["location"] = location
        ex["capture_date"] = date_str
        ex["capture_dt"] = capture_dt
        if capture_dt is None:
            missing_time.append(ex)
        else:
            grouped[(location, date_str)].append(ex)

    burst_counter = 0
    burst_id_field = f"burst_{time_window_seconds}_id"
    burst_target_field = f"burst_{time_window_seconds}_target"
    burst_purity_field = f"burst_{time_window_seconds}_purity"

    for key, rows in grouped.items():
        rows.sort(key=lambda x: x["capture_dt"])
        current_burst: list[dict[str, Any]] = []
        prev_dt: datetime | None = None

        for row in rows:
            if prev_dt is None or (row["capture_dt"] - prev_dt).total_seconds() <= time_window_seconds:
                current_burst.append(row)
            else:
                burst_counter += 1
                burst_id = f"{key[0]}_{key[1]}_{time_window_seconds}s_burst_{burst_counter:06d}"
                burst_target, burst_purity = choose_burst_target_and_purity(
                    [r["target"] for r in current_burst]
                )
                for r in current_burst:
                    r[burst_id_field] = burst_id
                    r[burst_target_field] = burst_target
                    r[burst_purity_field] = burst_purity
                current_burst = [row]
            prev_dt = row["capture_dt"]

        if current_burst:
            burst_counter += 1
            burst_id = f"{key[0]}_{key[1]}_{time_window_seconds}s_burst_{burst_counter:06d}"
            burst_target, burst_purity = choose_burst_target_and_purity(
                [r["target"] for r in current_burst]
            )
            for r in current_burst:
                r[burst_id_field] = burst_id
                r[burst_target_field] = burst_target
                r[burst_purity_field] = burst_purity

    for i, row in enumerate(missing_time, start=1):
        row[burst_id_field] = f"unknown_{time_window_seconds}s_burst_{i:06d}"
        row[burst_target_field] = row.get("target", "unknown")
        row[burst_purity_field] = 1.0

    for ex in examples:
        ex.pop("capture_dt", None)
    return examples


def apply_temporal_burst_consistency(
    examples: list[dict[str, Any]],
    time_windows: list[int],
    enforce_consistency: bool,
    short_window_purity_threshold: float,
) -> list[dict[str, Any]]:
    windows = sorted(time_windows)
    for w in windows:
        annotate_bursts_for_window(examples, w)

    short_w = windows[0]
    long_w = windows[-1]
    short_target_field = f"burst_{short_w}_target"
    long_target_field = f"burst_{long_w}_target"
    short_purity_field = f"burst_{short_w}_purity"

    for ex in examples:
        if not enforce_consistency:
            ex["burst_window_used"] = 0
            ex["split_group_id"] = ex.get(f"burst_{long_w}_id", ex["file_path"])
            continue

        short_purity = float(ex.get(short_purity_field, 0.0))
        if short_purity >= short_window_purity_threshold:
            ex["target"] = ex.get(short_target_field, ex["target"])
            ex["burst_window_used"] = short_w
        else:
            ex["target"] = ex.get(long_target_field, ex["target"])
            ex["burst_window_used"] = long_w

        ex["split_group_id"] = ex.get(f"burst_{long_w}_id", ex["file_path"])

    return examples


def build_examples(labels_json: str, shared_root: str) -> list[dict[str, Any]]:
    with open(labels_json, "r", encoding="utf-8") as f:
        rows = json.load(f)

    grouped_labels: dict[str, set[str]] = defaultdict(set)
    grouped_night: dict[str, str] = {}
    for row in rows:
        p = remap_image_path(row.get("file_path", ""), shared_root)
        if not p:
            continue
        grouped_labels[p].add(row.get("common_name", "unknown"))
        # Keep Y if any row for this image is marked as night.
        night = "Y" if str(row.get("night_time", "N")).strip().upper() == "Y" else "N"
        prev = grouped_night.get(p, "N")
        grouped_night[p] = "Y" if (prev == "Y" or night == "Y") else "N"

    examples = []
    for p, names in grouped_labels.items():
        target = labels_to_target(sorted(names))
        examples.append(
            {"file_path": p, "target": target, "night_time": grouped_night.get(p, "N")}
        )
    return examples


def load_examples_from_split_json(path: str, shared_root: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    out = []
    for row in rows:
        fp = remap_image_path(row.get("file_path", ""), shared_root)
        if not fp:
            continue
        out.append(
            {
                "file_path": fp,
                "target": row.get("correct_label", "unknown"),
                "night_time": "Y"
                if str(row.get("night_time", "N")).strip().upper() == "Y"
                else "N",
                "burst_window_used": int(row.get("burst_window_used", 0) or 0),
                "split_group_id": row.get("split_group_id", fp),
            }
        )
    return out


def train_val_test_split(
    examples: list[dict[str, Any]],
    eval_ratio: float,
    test_ratio: float,
    seed: int,
    burst_group_split: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError("--eval-ratio must be in (0, 1).")
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("--test-ratio must be in (0, 1).")
    if eval_ratio + test_ratio >= 1.0:
        raise ValueError("--eval-ratio + --test-ratio must be < 1.0")
    rnd = random.Random(seed)
    if not burst_group_split:
        shuffled = examples[:]
        rnd.shuffle(shuffled)
        n_eval = max(1, int(len(shuffled) * eval_ratio))
        n_test = max(1, int(len(shuffled) * test_ratio))
        eval_set = shuffled[:n_eval]
        test_set = shuffled[n_eval : n_eval + n_test]
        train_set = shuffled[n_eval + n_test :]
        return train_set, eval_set, test_set

    by_burst: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        by_burst[ex.get("split_group_id", ex["file_path"])].append(ex)

    burst_ids = list(by_burst.keys())
    rnd.shuffle(burst_ids)
    n_total = len(examples)
    n_eval_target = max(1, int(n_total * eval_ratio))
    n_test_target = max(1, int(n_total * test_ratio))

    train_set: list[dict[str, Any]] = []
    eval_set: list[dict[str, Any]] = []
    test_set: list[dict[str, Any]] = []

    eval_count = 0
    test_count = 0
    for bid in burst_ids:
        rows = by_burst[bid]
        if eval_count < n_eval_target:
            eval_set.extend(rows)
            eval_count += len(rows)
        elif test_count < n_test_target:
            test_set.extend(rows)
            test_count += len(rows)
        else:
            train_set.extend(rows)

    return train_set, eval_set, test_set


class AnimalVisionDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


@dataclass
class MultiModalCollator:
    processor: Any
    max_seq_length: int
    use_night_time: bool

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images = []
        user_texts = []
        full_texts = []

        for feat in features:
            image = Image.open(feat["file_path"]).convert("RGB")
            images.append(image)
            user_prompt = build_user_prompt(
                use_night_time=self.use_night_time, night_time=feat.get("night_time", "N")
            )

            user_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ]
            full_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_prompt},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": feat["target"]}]},
            ]

            user_texts.append(
                self.processor.apply_chat_template(
                    user_messages, tokenize=False, add_generation_prompt=True
                )
            )
            full_texts.append(
                self.processor.apply_chat_template(
                    full_messages, tokenize=False, add_generation_prompt=False
                )
            )

        full_batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        user_batch = self.processor(
            text=user_texts,
            images=images,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

        labels = full_batch["input_ids"].clone()
        user_lens = user_batch["attention_mask"].sum(dim=1)
        for i, user_len in enumerate(user_lens.tolist()):
            labels[i, : int(user_len)] = -100
        labels[full_batch["attention_mask"] == 0] = -100
        full_batch["labels"] = labels
        return full_batch


def load_model_and_processor(args: argparse.Namespace) -> tuple[Any, Any]:
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    quantization_config = None
    device_map = None

    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        device_map = "auto"

    model = AutoModelForVision2Seq.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map=device_map,
    )
    processor = AutoProcessor.from_pretrained(args.model_dir, use_fast=False)

    if args.use_lora:
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError as exc:
            raise ImportError(
                "LoRA requested but `peft` is not installed. Install with: pip install peft"
            ) from exc

        if args.use_4bit:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False

    return model, processor


def evaluate_generation(
    model: Any,
    processor: Any,
    eval_rows: list[dict[str, Any]],
    max_new_tokens: int,
    output_dir: Path,
    use_night_time: bool,
    split_name: str,
) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    if not hasattr(model, "hf_device_map"):
        model.to(device)

    details = []
    correct = 0
    for row in eval_rows:
        image = Image.open(row["file_path"]).convert("RGB")
        user_prompt = build_user_prompt(
            use_night_time=use_night_time, night_time=row.get("night_time", "N")
        )
        user_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        ok = is_correct(row["target"], pred)
        correct += int(ok)
        details.append(
            {
                "file_path": row["file_path"],
                "correct_label": row["target"],
                "night_time": row.get("night_time", "N"),
                "burst_window_used": row.get("burst_window_used", 0),
                "split_group_id": row.get("split_group_id", ""),
                "prediction": pred,
                "correct(Y/N)": "Y" if ok else "N",
            }
        )

    accuracy = correct / max(1, len(eval_rows))
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / f"{split_name}_predictions.csv"
    metrics_path = output_dir / f"{split_name}_metrics.json"

    import csv

    with details_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_path",
                "correct_label",
                "night_time",
                "burst_window_used",
                "split_group_id",
                "prediction",
                "correct(Y/N)",
            ],
        )
        writer.writeheader()
        writer.writerows(details)

    metrics = {
        "split": split_name,
        "samples": len(eval_rows),
        "correct": correct,
        "accuracy": accuracy,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_presplit = bool(args.train_json and args.validation_json and args.test_json)
    if use_presplit:
        train_rows = load_examples_from_split_json(args.train_json, args.shared_root)
        eval_rows = load_examples_from_split_json(args.validation_json, args.shared_root)
        test_rows = load_examples_from_split_json(args.test_json, args.shared_root)
        examples = train_rows + eval_rows + test_rows
        time_windows = parse_time_windows(args.time_windows)
    else:
        time_windows = parse_time_windows(args.time_windows)
        examples = build_examples(args.labels_json, args.shared_root)
        examples = apply_temporal_burst_consistency(
            examples=examples,
            time_windows=time_windows,
            enforce_consistency=not args.disable_burst_consistency,
            short_window_purity_threshold=args.short_window_purity_threshold,
        )
        if not examples:
            raise RuntimeError("No valid training examples found after path remap.")

        train_rows, eval_rows, test_rows = train_val_test_split(
            examples,
            args.eval_ratio,
            args.test_ratio,
            args.seed,
            burst_group_split=not args.disable_burst_group_split,
        )

    if args.max_train_samples > 0:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_eval_samples > 0:
        eval_rows = eval_rows[: args.max_eval_samples]
        test_rows = test_rows[: args.max_eval_samples]

    print(f"Total unique-image examples: {len(examples)}")
    if not use_presplit:
        for w in time_windows:
            print(
                f"Unique {w}s bursts: {len({x.get(f'burst_{w}_id', x['file_path']) for x in examples})}"
            )
        print(f"Split group count: {len({x.get('split_group_id', x['file_path']) for x in examples})}")
        print(
            "Burst window used counts:",
            {
                w: sum(1 for x in examples if int(x.get("burst_window_used", 0)) == w)
                for w in time_windows
            },
        )
    else:
        print("Using pre-split JSON files for train/validation/test.")
    print(f"Train examples: {len(train_rows)}")
    print(f"Validation examples: {len(eval_rows)}")
    print(f"Unseen test examples: {len(test_rows)}")

    model, processor = load_model_and_processor(args)
    data_collator = MultiModalCollator(
        processor=processor,
        max_seq_length=args.max_seq_length,
        use_night_time=not args.disable_night_time,
    )

    if not args.skip_train:
        train_dataset = AnimalVisionDataset(train_rows)
        eval_dataset = AnimalVisionDataset(eval_rows)

        fp16 = torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

        training_args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=2,
            fp16=fp16,
            bf16=bf16,
            gradient_checkpointing=args.gradient_checkpointing,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        resume_arg = str(args.resume_from_checkpoint).strip()
        if resume_arg.lower() == "auto":
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_candidates = sorted(
                checkpoint_dir.glob("checkpoint-*"),
                key=lambda p: p.stat().st_mtime,
            )
            resume_checkpoint = str(checkpoint_candidates[-1]) if checkpoint_candidates else None
        elif resume_arg.lower() == "none":
            resume_checkpoint = None
        else:
            resume_checkpoint = resume_arg

        if resume_checkpoint:
            print(f"Resuming from checkpoint: {resume_checkpoint}")
        else:
            print("Starting training from scratch (no checkpoint resume).")

        trainer.train(resume_from_checkpoint=resume_checkpoint)
        trainer.save_model(str(output_dir / "final_model"))
        processor.save_pretrained(str(output_dir / "final_model"))
    else:
        print("Skipping training (--skip-train). Running evaluation only.")

    val_metrics = evaluate_generation(
        model=model,
        processor=processor,
        eval_rows=eval_rows,
        max_new_tokens=args.max_new_tokens,
        output_dir=output_dir,
        use_night_time=not args.disable_night_time,
        split_name="validation",
    )
    test_metrics = evaluate_generation(
        model=model,
        processor=processor,
        eval_rows=test_rows,
        max_new_tokens=args.max_new_tokens,
        output_dir=output_dir,
        use_night_time=not args.disable_night_time,
        split_name="test",
    )
    print("Validation metrics:")
    print(json.dumps(val_metrics, indent=2))
    print("Test metrics (unseen):")
    print(json.dumps(test_metrics, indent=2))
    print(f"Saved validation predictions to: {output_dir / 'validation_predictions.csv'}")
    print(f"Saved validation metrics to: {output_dir / 'validation_metrics.json'}")
    print(f"Saved test predictions to: {output_dir / 'test_predictions.csv'}")
    print(f"Saved test metrics to: {output_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
