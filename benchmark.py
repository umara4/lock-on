"""Reproducible CPU/GPU pipeline benchmark and synthetic accuracy check."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from math import ceil
import os
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter
from typing import Callable

import cv2
import numpy as np

import config
from application.pipeline import VisionPipeline
from prediction.motion import MotionEstimator
from robotics.interceptor import calculate_interception
from interface.effect_renderer import EffectRenderer
from interface.effects import EffectEngine, EffectType
from interface.ability_effect_renderer import AbilityEffectRenderer
from interface.ability_effects import AbilityEffectEngine
from utils.video import VideoSourceError, open_video_source
from vision.anime_gestures import AnimeGestureRecognizer, AnimeRecognitionSnapshot
from vision.ability_recognition import (
    AbilityEvent,
    AbilityRecognitionSnapshot,
    AbilityRecognizer,
)
from vision.gesture_features import extract_hand_features, score_finger_states
from vision.gesture_sequences import GestureSequence, GestureSequenceRecognizer
from vision.gestures import DetectedHand, GestureType
from vision.tracker import TrackedObject


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark JARVIS on a video file or deterministic synthetic frames."
    )
    parser.add_argument("--source", help="Prerecorded video path (default: synthetic frames).")
    parser.add_argument("--model", help="Detector weights (defaults according to --detector).")
    parser.add_argument(
        "--detector",
        choices=("coco", "world"),
        default=config.DETECTOR_TYPE,
        help="Closed COCO detector or optional YOLO-World detector.",
    )
    parser.add_argument(
        "--classes",
        help="Comma-separated text vocabulary used by the YOLO-World detector.",
    )
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. cpu or 0.")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=config.TRACKING_INFERENCE_SIZE,
        help="YOLO inference image size (default: 640).",
    )
    parser.add_argument("--frames", type=int, default=120, help="Maximum frames to process.")
    parser.add_argument("--warmup", type=int, default=5, help="Frames excluded from stage averages.")
    parser.add_argument("--output", type=Path, help="Optional path for a JSON report.")
    return parser


def _synthetic_frame(index: int, width: int = 640, height: int = 480) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (14, 20, 25)
    center = (80 + (index * 5) % max(1, width - 160), height // 2)
    cv2.rectangle(frame, (center[0] - 25, center[1] - 45), (center[0] + 25, center[1] + 45), (180, 180, 180), -1)
    cv2.circle(frame, (width - center[0], height // 3), 24, (90, 130, 180), -1)
    return frame


def _summarize(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": ordered[min(len(ordered) - 1, ceil(0.95 * len(ordered)) - 1)],
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _synthetic_accuracy() -> dict[str, object]:
    """Measure pixel error for noisy constant-velocity data and analytic interception."""
    estimator = MotionEstimator(
        measurement_noise=9.0,
        process_noise=20.0,
        max_speed=1_000.0,
        low_confidence_residual=100.0,
    )
    rng = np.random.default_rng(2027)
    horizons: dict[float, list[float]] = defaultdict(list)
    timestamp = 0.0
    intervals = (0.08, 0.13, 0.1, 0.17, 0.09)
    for index in range(100):
        if index:
            timestamp += intervals[index % len(intervals)]
        true_x, true_y = 120.0 + 65.0 * timestamp, 90.0 + 25.0 * timestamp
        measured = (
            round(true_x + float(rng.normal(0.0, 3.0))),
            round(true_y + float(rng.normal(0.0, 3.0))),
        )
        track = TrackedObject(
            track_id=1,
            label="synthetic",
            confidence=1.0,
            bbox=(measured[0] - 1, measured[1] - 1, measured[0] + 1, measured[1] + 1),
            center=measured,
            position_history=(measured,),
            timestamp=timestamp,
            age_seconds=timestamp,
            frames_tracked=index + 1,
            missed_frames=0,
        )
        estimate = estimator.update([track], timestamp)[1]
        if estimate.measurement_count < config.MOTION_MIN_PREDICTION_MEASUREMENTS:
            continue
        for horizon, predicted in estimate.predictions.items():
            ground_truth = (true_x + 65.0 * horizon, true_y + 25.0 * horizon)
            horizons[horizon].append(float(np.hypot(
                predicted[0] - ground_truth[0],
                predicted[1] - ground_truth[1],
            )))

    prediction_errors = {
        f"{horizon:g}s": {
            "samples": len(errors),
            "mean_euclidean_position_error_px": statistics.fmean(errors),
            "rmse_px": float(np.sqrt(np.mean(np.square(errors)))),
        }
        for horizon, errors in sorted(horizons.items())
        if errors
    }

    feasible = calculate_interception((0.0, 0.0), (100.0, 0.0), (0.0, 0.0), 100.0)
    impossible = calculate_interception((0.0, 0.0), (100.0, 0.0), (200.0, 0.0), 100.0)
    return {
        "trajectory": "65,25 px/s constant velocity; seeded 3 px Gaussian measurement noise; variable timestamps",
        "prediction_error": prediction_errors,
        "analytic_interception_cases": {
            "reachable_plan_found": feasible is not None,
            "reachable_plan_time_seconds": feasible.time_seconds if feasible else None,
            "unreachable_plan_rejected": impossible is None,
        },
    }


def _synthetic_focus_hand(
    center_x: float,
    handedness: str,
    scale: float = 0.45,
) -> DetectedHand:
    """Create a deterministic, explicitly geometric two-finger pose for timing."""
    points = (
        (0.50, 0.85, 0.0), (0.45, 0.76, 0.0), (0.41, 0.71, 0.0),
        (0.48, 0.80, 0.0), (0.50, 0.85, 0.0), (0.42, 0.62, 0.0),
        (0.40, 0.48, 0.0), (0.39, 0.36, 0.0), (0.38, 0.20, 0.0),
        (0.48, 0.59, 0.0), (0.47, 0.44, 0.0), (0.47, 0.32, 0.0),
        (0.47, 0.16, 0.0), (0.55, 0.60, 0.0), (0.56, 0.46, 0.0),
        (0.48, 0.77, 0.0), (0.49, 0.79, 0.0), (0.62, 0.63, 0.0),
        (0.65, 0.52, 0.0), (0.48, 0.77, 0.0), (0.49, 0.79, 0.0),
    )
    transformed = tuple(
        (
            center_x + (x - 0.5) * scale,
            0.5 + (y - 0.5) * scale,
            z * scale,
        )
        for x, y, z in points
    )
    return DetectedHand(transformed, handedness, GestureType.OTHER, 0.9, 0.95)


def _synthetic_anime_benchmark(iterations: int = 100) -> dict[str, object]:
    """Time Phase 10 and Phase 11 geometry, movement, and effect stages."""
    if iterations < 10:
        raise ValueError("anime benchmark needs at least ten iterations")

    left = _synthetic_focus_hand(0.43, "Left")
    right = _synthetic_focus_hand(0.57, "Right")
    feature = extract_hand_features(left.landmarks, 1.0)
    if feature is None:
        raise RuntimeError("synthetic landmark fixture did not produce features")
    expected = (False, True, True, False, False)
    measurements: dict[str, list[float]] = defaultdict(list)

    def time_call(name: str, action: Callable[[], object]) -> None:
        for index in range(iterations + 5):
            started = perf_counter()
            action()
            elapsed = (perf_counter() - started) * 1_000.0
            if index >= 5:
                measurements[name].append(elapsed)

    time_call(
        "landmark_feature_extraction",
        lambda: extract_hand_features(left.landmarks, 1.0),
    )
    time_call(
        "static_finger_template_match",
        lambda: score_finger_states(feature, expected),
    )

    single = AnimeGestureRecognizer(cooldown_seconds=0.0)
    for index in range(iterations + 5):
        started = perf_counter()
        single.update((left,), index * 0.02, (640, 640))
        elapsed = (perf_counter() - started) * 1_000.0
        if index >= 5:
            measurements["single_hand_recognition"].append(elapsed)

    two_hand = AnimeGestureRecognizer(cooldown_seconds=0.0)
    for index in range(iterations + 5):
        started = perf_counter()
        two_hand.update((left, right), index * 0.02, (640, 640))
        elapsed = (perf_counter() - started) * 1_000.0
        if index >= 5:
            measurements["two_hand_recognition"].append(elapsed)

    sequence = GestureSequenceRecognizer(
        (
            GestureSequence(
                "benchmark_sequence",
                "Benchmark",
                ("two_finger_focus", "two_hand_seal"),
                3.0,
                "fireball",
            ),
        )
    )

    def match_sequence() -> None:
        sequence.reset()
        sequence.update((("recognized", "two_finger_focus"),), 0.0)
        sequence.update((("released", "two_finger_focus"),), 0.1)
        sequence.update((("recognized", "two_hand_seal"),), 0.2)

    time_call("sequence_recognition", match_sequence)

    ability_source = AnimeGestureRecognizer(cooldown_seconds=0.0)
    ability_observation = ability_source.update((left, right), 0.0, (640, 480))
    ability_observation = ability_source.update((left, right), 0.2, (640, 480))
    ability_recognizer = AbilityRecognizer()
    ability_clock = [0.2]

    def recognize_ability_movement() -> None:
        ability_clock[0] += 0.02
        ability_recognizer.update(ability_observation, ability_clock[0])

    time_call("ability_pose_and_movement_recognition", recognize_ability_movement)

    effects = EffectEngine(maximum_effects=8)
    empty_recognition = AnimeRecognitionSnapshot()

    def update_effects() -> None:
        effects.reset()
        timestamp = perf_counter()
        for effect_type in EffectType:
            effects.trigger(effect_type, timestamp, (0.5, 0.5))
        effects.update(timestamp + 0.001, empty_recognition)

    time_call("effect_state_update", update_effects)
    update_effects()
    effect_snapshots = effects.snapshot

    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QImage, QPainter

    canvas = QImage(640, 480, QImage.Format.Format_ARGB32)

    def render_effects() -> None:
        painter = QPainter(canvas)
        EffectRenderer.draw(painter, QRectF(0, 0, 640, 480), effect_snapshots, 0.5)
        painter.end()

    time_call("effect_vector_render", render_effects)

    ability_event = AbilityEvent(
        "kamehameha",
        "KAMEHAMEHA",
        "Synthetic benchmark",
        "activated",
        (0.5, 0.5),
        (1.0, 0.0),
        (),
        1.0,
    )
    ability_recognition = AbilityRecognitionSnapshot(
        selected_ability_id="kamehameha",
        phase="ACTIVATED",
        events=(ability_event,),
    )
    ability_effects = AbilityEffectEngine(maximum_effects=6)

    def update_ability_effects() -> None:
        ability_effects.update(perf_counter(), ability_recognition)

    time_call("ability_effect_state_update", update_ability_effects)
    ability_effects.update(perf_counter(), ability_recognition)
    ability_effect_snapshots = ability_effects.snapshot

    def render_ability_effects() -> None:
        painter = QPainter(canvas)
        AbilityEffectRenderer.draw(
            painter,
            QRectF(0, 0, 640, 480),
            ability_effect_snapshots,
            perf_counter(),
        )
        painter.end()

    time_call("ability_effect_vector_render", render_ability_effects)
    return {
        "input": "deterministic synthetic landmarks and QImage canvas; not live-camera performance",
        "iterations": iterations,
        "mean_median_p95_latency_ms": {
            name: _summarize(values) for name, values in measurements.items()
        },
        "accuracy_claim": "none; timings use synthetic fixture landmarks, not labeled real-hand samples",
    }


def _read_optional_process_metrics() -> tuple[Callable[[], int] | None, Callable[[], float] | None]:
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
    source: str | None,
    model_path: str,
    device: str | None,
    frame_limit: int,
    warmup_frames: int,
    inference_size: int = config.TRACKING_INFERENCE_SIZE,
    detector_type: str = config.DETECTOR_TYPE,
    vocabulary: tuple[str, ...] = config.OPEN_VOCABULARY_CLASSES,
) -> dict[str, object]:
    if frame_limit < 1:
        raise ValueError("frames must be at least one")
    if not 0 <= warmup_frames < frame_limit:
        raise ValueError("warmup must be non-negative and smaller than frames")

    started = perf_counter()
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    from interface.desktop import VideoCanvas

    app = QApplication.instance() or QApplication(["jarvis-benchmark"])
    canvas = VideoCanvas()
    canvas.resize(960, 600)
    canvas.show()
    app.processEvents()
    visible_at = perf_counter()

    try:
        capture = open_video_source(source) if source else None
    except VideoSourceError:
        canvas.close()
        if QApplication.instance() is app:
            app.quit()
        raise
    source_opened = perf_counter()
    initialization: dict[str, float] = {}
    model_init_started = perf_counter()
    gesture_init_started: float | None = None

    def report_initialization(state: str) -> None:
        nonlocal gesture_init_started
        now = perf_counter()
        if state == "Initializing gesture recognition":
            initialization["yolo_model_load"] = (now - model_init_started) * 1_000.0
            gesture_init_started = now
        elif state == "System ready" and gesture_init_started is not None:
            initialization["gesture_model_load"] = (now - gesture_init_started) * 1_000.0

    pipeline: VisionPipeline | None = None
    stage_samples: dict[str, list[float]] = defaultdict(list)
    frame_count = 0
    peak_rss_bytes = 0
    first_processed_seconds: float | None = None
    read_rss, read_cpu_percent = _read_optional_process_metrics()
    try:
        pipeline = VisionPipeline(
            model_path=model_path,
            confidence=config.CONFIDENCE_THRESHOLD,
            device=device,
            tracker_config=config.BYTE_TRACKER_CONFIG,
            inference_size=inference_size,
            detector_type=detector_type,
            vocabulary=vocabulary,
            classification_enabled=False,
            on_initialization=report_initialization,
        )
        initialization["total_model_and_pipeline_init"] = (perf_counter() - model_init_started) * 1_000.0
        if read_rss:
            peak_rss_bytes = read_rss()

        while frame_count < frame_limit:
            frame_started = perf_counter()
            read_started = perf_counter()
            if capture is None:
                frame = _synthetic_frame(frame_count)
                has_frame = True
            else:
                has_frame, frame = capture.read()
            input_seconds = perf_counter() - read_started
            if not has_frame:
                break

            timestamp = perf_counter()
            snapshot = pipeline.process_frame(frame, timestamp)
            timing = pipeline.last_timing
            render_started = perf_counter()
            image = QImage(
                frame.data,
                frame.shape[1],
                frame.shape[0],
                int(frame.strides[0]),
                QImage.Format.Format_BGR888,
            ).copy()
            canvas.set_image(image, (frame.shape[1], frame.shape[0]))
            canvas.set_snapshot(snapshot, 0.0, True, True)
            canvas.repaint()
            app.processEvents()
            render_seconds = perf_counter() - render_started
            frame_seconds = perf_counter() - frame_started
            if first_processed_seconds is None:
                first_processed_seconds = perf_counter() - started

            frame_count += 1
            if frame_count > warmup_frames:
                for name, duration in (
                    ("input_read_or_generation", input_seconds),
                    ("ultralytics_track_detection_and_association", timing.tracking_seconds),
                    ("tracker_output_adapter_and_history", timing.tracking_adapter_seconds),
                    ("gesture", timing.gesture_seconds),
                    ("motion_estimation", timing.motion_seconds),
                    ("qt_camera_canvas_update_render", render_seconds),
                    ("end_to_end_frame", frame_seconds),
                ):
                    stage_samples[name].append(duration * 1_000.0)
            if read_rss:
                peak_rss_bytes = max(peak_rss_bytes, read_rss())

        finished = perf_counter()
    finally:
        try:
            if capture is not None:
                capture.release()
        finally:
            try:
                if pipeline is not None:
                    pipeline.close()
            finally:
                canvas.close()
                if QApplication.instance() is app:
                    app.quit()

    if frame_count == 0:
        raise RuntimeError("benchmark input produced no readable frames")
    steady_state_latency = _summarize(stage_samples["end_to_end_frame"])
    render_latency = _summarize(stage_samples["qt_camera_canvas_update_render"])
    tracking_latency = _summarize(
        stage_samples["ultralytics_track_detection_and_association"]
    )
    cpu_percent = read_cpu_percent() if read_cpu_percent else None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_device = torch.cuda.get_device_name(0) if cuda_available else None
        torch_version = torch.__version__
    except ImportError:
        cuda_available, cuda_device, torch_version = False, None, None

    return {
        "input": source if source else "generated synthetic frames (640x480)",
        "model": model_path,
        "detector": detector_type,
        "open_vocabulary": list(vocabulary) if detector_type == "world" else None,
        "inference_size": inference_size,
        "requested_device": device or "Ultralytics default",
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "opencv": cv2.__version__,
            "torch": torch_version,
            "cuda_available": cuda_available,
            "cuda_device_0": cuda_device,
        },
        "startup_ms": {
            "qt_canvas_visible_from_benchmark_start": (visible_at - started) * 1_000.0,
            "source_open": (source_opened - visible_at) * 1_000.0,
            **initialization,
            "time_to_first_processed_and_rendered_frame": (
                first_processed_seconds * 1_000.0 if first_processed_seconds is not None else None
            ),
        },
        "frames_processed": frame_count,
        "warmup_frames_excluded_from_stage_statistics": warmup_frames,
        "wall_time_seconds": finished - started,
        "fps_including_startup": frame_count / max(finished - started, 1e-9),
        "steady_state_processed_fps": (
            1_000.0 / steady_state_latency["mean_ms"]
            if steady_state_latency is not None
            else None
        ),
        "yolo_tracking_call_fps": (
            1_000.0 / tracking_latency["mean_ms"]
            if tracking_latency is not None and tracking_latency["mean_ms"] > 0
            else None
        ),
        "qt_canvas_update_render_fps": (
            1_000.0 / render_latency["mean_ms"]
            if render_latency is not None
            else None
        ),
        "stage_latency": {
            name: _summarize(values) for name, values in stage_samples.items()
        },
        "memory": {
            "max_sampled_rss_mb": peak_rss_bytes / (1024 * 1024) if read_rss else None,
            "process_cpu_percent_sample": cpu_percent,
            "psutil_available": read_rss is not None,
        },
        "anime_gesture_effect_benchmark": _synthetic_anime_benchmark(),
        "timing_notes": [
            "The Qt camera canvas is rendered offscreen; its update/render throughput is not the full application's GUI refresh rate.",
            "Ultralytics model.track() bundles detector inference and ByteTrack association; they are reported together.",
            "Tracker adapter timing covers conversion of model results and local position-history updates.",
            "Timing statistics exclude the configured warmup frames; model initialization is separately reported.",
            "Input acquisition timing includes synthetic frame generation when no video source is specified.",
            "RSS is sampled after initialization and once per frame; a brief peak between samples may be missed.",
            "Prediction accuracy below is a deterministic synthetic test, not a real-camera accuracy claim.",
            "Anime feature/recognition/effect timings use generated landmarks and an offscreen vector canvas, not webcam input.",
        ],
        "synthetic_accuracy": _synthetic_accuracy(),
    }


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    model_path = args.model or (
        config.OPEN_VOCABULARY_MODEL_PATH
        if args.detector == "world"
        else config.MODEL_PATH
    )
    vocabulary = (
        tuple(item.strip() for item in args.classes.split(",") if item.strip())
        if args.classes
        else config.OPEN_VOCABULARY_CLASSES
    )
    try:
        report = run_benchmark(
            source=args.source,
            model_path=model_path,
            device=args.device,
            frame_limit=args.frames,
            warmup_frames=args.warmup,
            inference_size=args.imgsz,
            detector_type=args.detector,
            vocabulary=vocabulary,
        )
        rendered = json.dumps(report, indent=2)
        if args.output:
            args.output.write_text(rendered + "\n", encoding="utf-8")
    except (OSError, RuntimeError, ValueError, VideoSourceError) as error:
        parser.exit(1, f"Benchmark failed: {error}\n")

    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
