"""Measure one-time person segmentation and cached clone compositing costs."""

from __future__ import annotations

import argparse
import json
from statistics import mean, median
from time import perf_counter

import cv2
import numpy as np
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QImage, QPainter

import config
from interface.ability_effect_renderer import AbilityEffectRenderer
from interface.ability_effects import (
    AbilityEffectEngine,
    AbilityEffectSnapshot,
    CloneSpriteSnapshot,
)


def _synthetic_person_frame(width: int, height: int) -> np.ndarray:
    frame = np.full((height, width, 3), (22, 18, 14), dtype=np.uint8)
    center_x = width // 2
    head_radius = max(10, width // 22)
    cv2.circle(
        frame,
        (center_x, height // 5),
        head_radius,
        (180, 155, 125),
        -1,
        cv2.LINE_AA,
    )
    shoulder_y = height * 2 // 5
    torso_width = width // 5
    cv2.rectangle(
        frame,
        (center_x - torso_width // 2, shoulder_y),
        (center_x + torso_width // 2, height * 4 // 5),
        (110, 75, 45),
        -1,
    )
    cv2.line(
        frame,
        (center_x - torso_width // 2, shoulder_y + 8),
        (center_x - width // 3, height * 3 // 4),
        (160, 130, 100),
        max(8, width // 30),
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        (center_x + torso_width // 2, shoulder_y + 8),
        (center_x + width // 3, height * 3 // 4),
        (160, 130, 100),
        max(8, width // 30),
        cv2.LINE_AA,
    )
    return frame


def _summary(samples: list[float]) -> dict[str, float]:
    return {
        "mean_ms": mean(samples),
        "median_ms": median(samples),
        "p95_ms": float(np.percentile(samples, 95)),
    }


def run_benchmark(
    width: int = 640,
    height: int = 480,
    iterations: int = 8,
) -> dict[str, object]:
    if width < 64 or height < 64 or iterations < 2:
        raise ValueError("use image dimensions >= 64 and at least two iterations")
    frame = _synthetic_person_frame(width, height)
    engine = AbilityEffectEngine()
    segment_times: list[float] = []
    sprites: tuple[CloneSpriteSnapshot, ...] = ()
    first_segmentation_ms = 0.0
    try:
        for index in range(iterations + 1):
            started = perf_counter()
            sprites = engine._make_clones(index + 1, frame)
            elapsed_ms = (perf_counter() - started) * 1_000.0
            if index == 0:
                first_segmentation_ms = elapsed_ms
            else:
                segment_times.append(elapsed_ms)
        used_fixture = not sprites
        if used_fixture:
            fixture = np.zeros((height // 2, width // 3, 4), dtype=np.uint8)
            fixture[..., :3] = (195, 115, 75)
            cv2.ellipse(
                fixture,
                (fixture.shape[1] // 2, fixture.shape[0] // 2),
                (max(8, fixture.shape[1] // 4), max(12, fixture.shape[0] // 2)),
                0,
                0,
                360,
                (195, 115, 75, 190),
                -1,
                cv2.LINE_AA,
            )
            sprites = tuple(
                CloneSpriteSnapshot(
                    1,
                    index,
                    fixture.tobytes(),
                    fixture.shape[1],
                    fixture.shape[0],
                    (0.32, 0.25, fixture.shape[1] / width, fixture.shape[0] / height),
                    config.ABILITY_SHADOW_CLONE_OPACITY,
                    (index - 1) * config.ABILITY_SHADOW_CLONE_SPACING_RATIO,
                )
                for index in range(3)
            )

        image_cache = {
            (sprite.effect_id, sprite.clone_index): QImage(
                sprite.rgba,
                sprite.width,
                sprite.height,
                sprite.width * 4,
                QImage.Format.Format_RGBA8888,
            ).copy()
            for sprite in sprites
        }
        effect = AbilityEffectSnapshot(
            1,
            "shadow_clone",
            "shadow_clones",
            "activated",
            (0.5, 0.5),
            (1.0, 0.0),
            1.0,
            config.ABILITY_SHADOW_CLONE_DURATION_SECONDS,
            1.0,
            37,
            (),
            sprites,
        )
        composite_times: list[float] = []
        output = QImage(width, height, QImage.Format.Format_ARGB32)
        for index in range(iterations + 1):
            output.fill(QColor("#07111d"))
            painter = QPainter(output)
            started = perf_counter()
            AbilityEffectRenderer.draw_clones(
                painter,
                QRectF(0, 0, width, height),
                (effect,),
                2.0 + index / 30.0,
                image_cache,
            )
            painter.end()
            elapsed_ms = (perf_counter() - started) * 1_000.0
            if index:
                composite_times.append(elapsed_ms)
        return {
            "input": "generated synthetic silhouette; not a human-recognition accuracy test",
            "dimensions": [width, height],
            "iterations": iterations,
            "segmentation_plus_mask_crop": _summary(segment_times),
            "clone_compositing": _summary(composite_times),
            "sprites_produced_by_segmentation": len(sprites) if not used_fixture else 0,
            "compositing_fixture_used": used_fixture,
            "first_segmentation_including_model_initialization_ms": first_segmentation_ms,
            "note": (
                "Compositing uses cached RGBA sprites and the same QPainter path as the GUI. "
                "Times exclude camera capture and YOLO/hand inference."
            ),
        }
    finally:
        engine.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--iterations", type=int, default=8)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_benchmark(arguments.width, arguments.height, arguments.iterations),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
