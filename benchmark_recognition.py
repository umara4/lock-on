"""Measure MobileNetV3 CPU crop latency and, when labeled data is supplied, accuracy."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from math import ceil
import json
import os
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter
from typing import Callable, Sequence

import cv2
import numpy as np

from vision.recognition import ClassificationCandidate, MobileNetV3Classifier


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark asynchronous JARVIS crop-classifier model inference on CPU."
    )
    parser.add_argument(
        "--images",
        type=Path,
        help="Optional labeled ImageFolder directory with one class subdirectory per label.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum labeled images to evaluate, or synthetic crops to time (default: 100).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Synthetic or first-image inference calls excluded from latency statistics.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report destination.")
    return parser


def _label_variants(label: str) -> set[str]:
    aliases = {
        "computer keyboard": "keyboard",
        "mobile phone": "cell phone",
        "cellular telephone": "cell phone",
        "cellular phone": "cell phone",
        "coffee mug": "mug",
    }

    def normalize(value: str) -> str:
        normalized = " ".join(
            value.casefold().replace("_", " ").replace("-", " ").split()
        )
        return aliases.get(normalized, normalized)

    return {normalize(value) for value in label.split(",") if value.strip()}


def _labels_match(first: str, second: str) -> bool:
    return bool(_label_variants(first) & _label_variants(second))


def evaluate_predictions(
    examples: Sequence[tuple[str, tuple[ClassificationCandidate, ...]]],
) -> dict[str, object]:
    if not examples:
        return {
            "samples": 0,
            "top1_accuracy": None,
            "top5_accuracy": None,
            "per_class": {},
            "confusion_matrix": {},
        }

    per_class: dict[str, dict[str, int]] = defaultdict(
        lambda: {"samples": 0, "top1_correct": 0, "top5_correct": 0}
    )
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    top1_correct = 0
    top5_correct = 0
    for expected, candidates in examples:
        predicted = candidates[0].label if candidates else "no valid candidate"
        top1 = bool(candidates) and _labels_match(expected, predicted)
        top5 = any(
            _labels_match(expected, candidate.label)
            for candidate in candidates[:5]
        )
        top1_correct += int(top1)
        top5_correct += int(top5)
        stats = per_class[expected]
        stats["samples"] += 1
        stats["top1_correct"] += int(top1)
        stats["top5_correct"] += int(top5)
        confusion[expected][predicted] += 1

    return {
        "samples": len(examples),
        "top1_accuracy": top1_correct / len(examples),
        "top5_accuracy": top5_correct / len(examples),
        "per_class": {
            label: {
                "samples": counts["samples"],
                "top1_accuracy": counts["top1_correct"] / counts["samples"],
                "top5_accuracy": counts["top5_correct"] / counts["samples"],
            }
            for label, counts in sorted(per_class.items())
        },
        "confusion_matrix": {
            label: dict(sorted(counts.items()))
            for label, counts in sorted(confusion.items())
        },
    }


def _synthetic_crop(index: int) -> np.ndarray:
    crop = np.zeros((160, 160, 3), dtype=np.uint8)
    crop[:] = (20 + index % 20, 30, 40)
    color = (80 + index % 150, 140, 200)
    cv2.rectangle(crop, (25, 30), (135, 125), color, -1)
    cv2.circle(crop, (80, 80), 30, (30, 180, 90), 3)
    return crop


def _memory_reader() -> tuple[Callable[[], int] | None, Callable[[], float] | None]:
    try:
        import psutil
    except ImportError:
        return None, None
    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)
    return (
        lambda: process.memory_info().rss,
        lambda: process.cpu_percent(interval=None),
    )


def run_benchmark(
    images_path: Path | None,
    limit: int,
    warmup: int,
) -> dict[str, object]:
    if limit < 1:
        raise ValueError("--limit must be at least one")
    if warmup < 0:
        raise ValueError("--warmup cannot be negative")
    if images_path is not None and not images_path.is_dir():
        raise ValueError(f"ImageFolder directory does not exist: {images_path}")

    read_memory, read_cpu = _memory_reader()
    before_model_rss = read_memory() if read_memory else None
    initialization_started = perf_counter()
    classifier = MobileNetV3Classifier()
    model_initialization_ms = (perf_counter() - initialization_started) * 1_000.0
    after_model_rss = read_memory() if read_memory else None

    dataset = None
    if images_path is not None:
        try:
            from torchvision.datasets import ImageFolder
        except ImportError as error:
            raise RuntimeError(
                "Labeled evaluation requires torchvision and Pillow."
            ) from error
        dataset = ImageFolder(str(images_path))
        if not dataset:
            raise ValueError(f"No images found in ImageFolder directory: {images_path}")

    examples: list[tuple[str, tuple[ClassificationCandidate, ...]]] = []
    latency_ms: list[float] = []
    peak_rss = after_model_rss or 0
    total_samples = min(limit, len(dataset)) if dataset is not None else limit
    warmup_crop = _synthetic_crop(0)
    if dataset is not None:
        first_image, _ = dataset[0]
        warmup_crop = cv2.cvtColor(np.asarray(first_image.convert("RGB")), cv2.COLOR_RGB2BGR)

    for _ in range(warmup):
        classifier.classify(warmup_crop)

    for index in range(total_samples):
        expected_label: str | None = None
        if dataset is None:
            crop = _synthetic_crop(index)
        else:
            image, class_id = dataset[index]
            crop = cv2.cvtColor(
                np.asarray(image.convert("RGB")),
                cv2.COLOR_RGB2BGR,
            )
            expected_label = dataset.classes[class_id]
        started = perf_counter()
        candidates = classifier.classify(crop)
        latency_ms.append((perf_counter() - started) * 1_000.0)
        if expected_label is not None:
            examples.append((expected_label, candidates))
        if read_memory:
            peak_rss = max(peak_rss, read_memory())

    cpu_percent = read_cpu() if read_cpu else None
    accuracy = evaluate_predictions(examples)
    return {
        "model": classifier.model_id,
        "device": "cpu",
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "dataset": (
            {
                "path": str(images_path),
                "evaluation": "labeled ImageFolder classes, exact normalized ImageNet synonym match",
                "class_count": len(dataset.classes),
            }
            if dataset is not None
            else {
                "evaluation": "synthetic crops for latency only; classification accuracy not measured",
            }
        ),
        "model_initialization_ms": model_initialization_ms,
        "warmup_inferences_excluded": warmup,
        "inferences_measured": total_samples,
        "crop_latency": {
            "mean_ms": statistics.fmean(latency_ms),
            "median_ms": statistics.median(latency_ms),
            "p95_ms": sorted(latency_ms)[min(len(latency_ms) - 1, ceil(0.95 * len(latency_ms)) - 1)],
            "min_ms": min(latency_ms),
            "max_ms": max(latency_ms),
        },
        "accuracy": accuracy if dataset is not None else None,
        "memory": {
            "rss_before_model_mb": before_model_rss / (1024 * 1024) if before_model_rss else None,
            "rss_after_model_mb": after_model_rss / (1024 * 1024) if after_model_rss else None,
            "max_sampled_rss_mb": peak_rss / (1024 * 1024) if read_memory else None,
            "process_cpu_percent_sample": cpu_percent,
            "psutil_available": read_memory is not None,
        },
        "notes": [
            "The classifier is an ImageNet-1K crop classifier, not an object detector.",
            "Softmax values are uncalibrated model scores, not calibrated probabilities.",
            "Synthetic crops do not support accuracy claims; use a representative labeled ImageFolder dataset.",
            "CPU timings are host-, thread-, image-content-, and warm-cache-sensitive.",
        ],
    }


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    try:
        report = run_benchmark(args.images, args.limit, args.warmup)
        rendered = json.dumps(report, indent=2)
        if args.output:
            args.output.write_text(rendered + "\n", encoding="utf-8")
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Recognition benchmark failed: {error}\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
