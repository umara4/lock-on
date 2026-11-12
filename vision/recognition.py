"""Asynchronous, per-track classification verification for detected objects."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
import logging
from math import isfinite
from queue import Empty, Full, PriorityQueue
import threading
import time
from typing import Callable, Protocol

import cv2
import numpy as np

import config
from vision.tracker import TrackedObject

logger = logging.getLogger(__name__)


class ClassificationStatus(str, Enum):
    DETECTED = "DETECTED"
    CLASSIFYING = "CLASSIFYING"
    CONFIRMED = "CONFIRMED"
    REFINED = "REFINED"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class ClassificationCandidate:
    """One uncalibrated model score and label; score is not a probability."""

    label: str
    score: float


@dataclass(frozen=True)
class ClassificationPrediction:
    candidates: tuple[ClassificationCandidate, ...]
    model_id: str
    timestamp: float


@dataclass(frozen=True)
class TrackRecognition:
    """Detection provenance and optional temporally-consistent crop classification."""

    track_id: int
    detector_label: str
    detector_confidence: float
    refined_label: str | None
    classification_score: float | None
    model_id: str | None
    status: ClassificationStatus
    timestamp: float
    alternatives: tuple[ClassificationCandidate, ...] = ()
    error: str | None = None

    @property
    def display_label(self) -> str:
        return (
            self.refined_label
            if self.status == ClassificationStatus.REFINED and self.refined_label
            else self.detector_label
        )


class CropClassifier(Protocol):
    """Backend contract for image-crop classifiers."""

    model_id: str

    def classify(self, crop: np.ndarray) -> tuple[ClassificationCandidate, ...]: ...


class MobileNetV3Classifier:
    """Torchvision MobileNetV3-Large ImageNet-1K crop classifier."""

    model_id = "torchvision/mobilenet_v3_large.imagenet1k_v2"

    def __init__(self) -> None:
        try:
            import torch
            from PIL import Image
            from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
        except ImportError as error:
            raise RuntimeError(
                "Torchvision classification requires torch, torchvision, and Pillow."
            ) from error

        self._torch = torch
        self._image_type = Image
        self._weights = MobileNet_V3_Large_Weights.DEFAULT
        self._model = mobilenet_v3_large(weights=self._weights).eval()
        self._transform = self._weights.transforms()
        self._categories = self._weights.meta["categories"]

    def classify(self, crop: np.ndarray) -> tuple[ClassificationCandidate, ...]:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        input_tensor = self._transform(self._image_type.fromarray(rgb)).unsqueeze(0)
        with self._torch.inference_mode():
            scores = self._torch.softmax(self._model(input_tensor)[0], dim=0)
            values, indices = self._torch.topk(
                scores,
                min(config.CLASSIFICATION_TOP_K, len(self._categories)),
            )
        return tuple(
            ClassificationCandidate(
                label=self._categories[int(index)],
                score=float(score),
            )
            for score, index in zip(values.tolist(), indices.tolist())
        )


def extract_detection_crop(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    minimum_size: int = config.CLASSIFICATION_CROP_MINIMUM_SIZE,
    padding_ratio: float = 0.06,
) -> np.ndarray | None:
    """Return a copied, padded BGR crop or ``None`` for invalid/tiny boxes."""
    if frame.ndim != 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
        return None
    if minimum_size < 1 or not 0.0 <= padding_ratio <= 0.5:
        raise ValueError("crop size and padding ratio must be valid")
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    if not all(isinstance(value, (int, np.integer)) for value in bbox):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    box_width = x2 - x1
    box_height = y2 - y1
    padding_x = round(box_width * padding_ratio)
    padding_y = round(box_height * padding_ratio)
    left = max(0, x1 - padding_x)
    top = max(0, y1 - padding_y)
    right = min(width, x2 + padding_x)
    bottom = min(height, y2 + padding_y)
    if right - left < minimum_size or bottom - top < minimum_size:
        return None
    return frame[top:bottom, left:right].copy()


def _normalized_label(label: str) -> str:
    return " ".join(label.casefold().replace("_", " ").split())


def _labels_agree(first: str, second: str) -> bool:
    aliases = {
        "computer keyboard": "keyboard",
        "mobile phone": "cell phone",
        "cellular telephone": "cell phone",
        "cellular phone": "cell phone",
        "coffee mug": "mug",
    }

    def canonical(label: str) -> str:
        choices = (_normalized_label(value) for value in label.split(","))
        normalized = next(choices, "")
        return aliases.get(normalized, normalized)

    return canonical(first) == canonical(second)


@dataclass
class _TrackRecognitionState:
    generation: int
    result: TrackRecognition
    last_request_time: float = float("-inf")
    last_classification_time: float = float("-inf")
    last_signature_check: float = float("-inf")
    appearance_signature: np.ndarray | None = None
    observations: deque[ClassificationPrediction] = field(
        default_factory=lambda: deque(maxlen=config.CLASSIFICATION_HISTORY_LENGTH)
    )


@dataclass(frozen=True)
class _ClassificationJob:
    epoch: int
    track_id: int
    generation: int
    detector_label: str
    detector_confidence: float
    crop: np.ndarray
    submitted_at: float


class TrackRecognitionManager:
    """Keep classification work bounded and reject results for stale track identities."""

    def __init__(
        self,
        classifier_factory: Callable[[], CropClassifier] = MobileNetV3Classifier,
        enabled: bool = config.CLASSIFICATION_ENABLED,
        max_pending_jobs: int = config.CLASSIFICATION_MAX_PENDING_JOBS,
        requests_per_second: float = config.CLASSIFICATION_MAX_REQUESTS_PER_SECOND,
    ) -> None:
        if max_pending_jobs < 1 or not isfinite(requests_per_second) or requests_per_second <= 0:
            raise ValueError("queue size and classification rate must be positive")
        self._classifier_factory = classifier_factory
        self._enabled = enabled
        self._requests_per_second = requests_per_second
        self._minimum_request_interval = 1.0 / requests_per_second
        self._jobs: PriorityQueue[tuple[int, int, _ClassificationJob]] = PriorityQueue(
            maxsize=max_pending_jobs
        )
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._classifier: CropClassifier | None = None
        self._classifier_error: str | None = None
        self._sequence = 0
        self._epoch = 0
        self._next_generation: dict[int, int] = {}
        self._states: dict[int, _TrackRecognitionState] = {}
        self._pending: set[tuple[int, int, int]] = set()
        self._last_request_time = float("-inf")
        self.last_inference_seconds = 0.0
        self.completed_count = 0
        self.dropped_count = 0

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        if enabled == self.enabled:
            return
        self.reset()
        with self._lock:
            self._enabled = enabled
            if enabled:
                self._classifier_error = None

    def reset(self) -> None:
        with self._lock:
            self._epoch += 1
            self._states.clear()
            self._next_generation.clear()
            self._pending.clear()
            while True:
                try:
                    self._jobs.get_nowait()
                    self._jobs.task_done()
                except Empty:
                    break

    def snapshot(self) -> dict[int, TrackRecognition]:
        with self._lock:
            return {
                track_id: state.result
                for track_id, state in self._states.items()
            }

    def update(
        self,
        frame: np.ndarray,
        tracks: list[TrackedObject],
        retained_track_ids: set[int],
        timestamp: float,
        selected_id: int | None,
    ) -> dict[int, TrackRecognition]:
        """Schedule bounded crop classifications and return current per-ID state."""
        with self._lock:
            if not self._enabled:
                return {
                    track_id: state.result
                    for track_id, state in self._states.items()
                }

            for track_id in tuple(self._states):
                if track_id not in retained_track_ids:
                    state = self._states.pop(track_id)
                    self._pending.discard((self._epoch, track_id, state.generation))
                    self._next_generation[track_id] = state.generation + 1

            ordered_tracks = sorted(
                tracks,
                key=lambda track: (
                    track.track_id != selected_id,
                    -track.frames_tracked,
                    track.track_id,
                ),
            )
            eligible: list[tuple[TrackedObject, _TrackRecognitionState, np.ndarray]] = []
            for track in ordered_tracks:
                if track.frames_tracked < config.CLASSIFICATION_MIN_TRACK_FRAMES:
                    continue
                state = self._states.get(track.track_id)
                if state is None:
                    generation = self._next_generation.get(track.track_id, 0)
                    state = _TrackRecognitionState(
                        generation,
                        TrackRecognition(
                            track_id=track.track_id,
                            detector_label=track.label,
                            detector_confidence=track.confidence,
                            refined_label=None,
                            classification_score=None,
                            model_id=None,
                            status=ClassificationStatus.DETECTED,
                            timestamp=timestamp,
                        ),
                    )
                    self._states[track.track_id] = state
                elif state.result.detector_label != track.label:
                    old_identity = (self._epoch, track.track_id, state.generation)
                    self._pending.discard(old_identity)
                    state.generation += 1
                    self._next_generation[track.track_id] = state.generation
                    state.observations.clear()
                    state.last_request_time = float("-inf")
                    state.last_classification_time = float("-inf")
                    state.appearance_signature = None
                    state.result = TrackRecognition(
                        track_id=track.track_id,
                        detector_label=track.label,
                        detector_confidence=track.confidence,
                        refined_label=None,
                        classification_score=None,
                        model_id=None,
                        status=ClassificationStatus.DETECTED,
                        timestamp=timestamp,
                    )
                else:
                    state.result = TrackRecognition(
                        track_id=track.track_id,
                        detector_label=track.label,
                        detector_confidence=track.confidence,
                        refined_label=state.result.refined_label,
                        classification_score=state.result.classification_score,
                        model_id=state.result.model_id,
                        status=state.result.status,
                        timestamp=state.result.timestamp,
                        alternatives=state.result.alternatives,
                        error=state.result.error,
                    )

                identity = (self._epoch, track.track_id, state.generation)
                if identity in self._pending:
                    continue
                crop = extract_detection_crop(frame, track.bbox)
                if crop is None:
                    if state.result.status != ClassificationStatus.DETECTED:
                        state.result = self._status_result(
                            state.result,
                            ClassificationStatus.UNCERTAIN,
                            timestamp,
                            error="Detection crop is too small or invalid.",
                        )
                    continue
                changed = self._appearance_changed(state, crop, timestamp)
                age = timestamp - state.last_classification_time
                never_classified = state.last_classification_time == float("-inf")
                uncertain = state.result.status in (
                    ClassificationStatus.UNCERTAIN,
                )
                refresh_interval = (
                    config.CLASSIFICATION_RETRY_SECONDS
                    if uncertain
                    else config.CLASSIFICATION_REFRESH_SECONDS
                )
                if (
                    timestamp - state.last_request_time
                    >= config.CLASSIFICATION_MIN_REQUEST_INTERVAL_SECONDS
                    and (
                        never_classified
                        or (
                            changed
                            and age >= config.CLASSIFICATION_RETRY_SECONDS
                        )
                        or (
                            age >= refresh_interval
                            and state.last_request_time
                            + config.CLASSIFICATION_MIN_REQUEST_INTERVAL_SECONDS
                            <= timestamp
                        )
                    )
                ):
                    eligible.append((track, state, crop))

            for track, state, crop in eligible:
                if timestamp - self._last_request_time < self._minimum_request_interval:
                    break
                identity = (self._epoch, track.track_id, state.generation)
                if identity in self._pending:
                    continue
                self._sequence += 1
                job = _ClassificationJob(
                    epoch=self._epoch,
                    track_id=track.track_id,
                    generation=state.generation,
                    detector_label=track.label,
                    detector_confidence=track.confidence,
                    crop=crop,
                    submitted_at=timestamp,
                )
                try:
                    self._jobs.put_nowait(
                        (0 if track.track_id == selected_id else 1, self._sequence, job)
                    )
                except Full:
                    self.dropped_count += 1
                    state.last_request_time = timestamp
                    self._last_request_time = timestamp
                    break
                self._pending.add(identity)
                self._last_request_time = timestamp
                state.last_request_time = timestamp
                state.result = self._status_result(
                    state.result,
                    ClassificationStatus.CLASSIFYING,
                    timestamp,
                    error=None,
                )
                if self._worker is None or not self._worker.is_alive():
                    self._stop.clear()
                    self._worker = threading.Thread(
                        target=self._run,
                        name="jarvis-crop-classifier",
                        daemon=True,
                    )
                    self._worker.start()

            return {
                track_id: state.result
                for track_id, state in self._states.items()
            }

    def close(self, timeout: float = 5.0) -> bool:
        self._stop.set()
        with self._lock:
            while True:
                try:
                    self._jobs.get_nowait()
                    self._jobs.task_done()
                except Empty:
                    break
            self._pending.clear()
        worker = self._worker
        if worker is None or not worker.is_alive():
            return True
        worker.join(timeout)
        return not worker.is_alive()

    def _appearance_changed(
        self,
        state: _TrackRecognitionState,
        crop: np.ndarray,
        timestamp: float,
    ) -> bool:
        if timestamp - state.last_signature_check < config.CLASSIFICATION_MIN_REQUEST_INTERVAL_SECONDS:
            return False
        state.last_signature_check = timestamp
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        signature = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        if state.appearance_signature is None:
            state.appearance_signature = signature
            return False
        distance = float(np.mean(np.abs(signature - state.appearance_signature)))
        if distance >= config.CLASSIFICATION_APPEARANCE_CHANGE_THRESHOLD:
            state.appearance_signature = signature
            return True
        return False

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                _, _, job = self._jobs.get(timeout=0.05)
            except Empty:
                continue
            identity = (job.epoch, job.track_id, job.generation)
            try:
                with self._lock:
                    current_state = self._states.get(job.track_id)
                    if (
                        job.epoch != self._epoch
                        or current_state is None
                        or current_state.generation != job.generation
                    ):
                        self._pending.discard(identity)
                        continue
                    classifier_error = self._classifier_error

                if classifier_error is not None:
                    self._finish_error(job, classifier_error)
                    continue
                if self._classifier is None:
                    try:
                        self._classifier = self._classifier_factory()
                    except Exception as error:
                        logger.exception("Could not initialize crop classifier")
                        with self._lock:
                            self._classifier_error = str(error)
                        self._finish_error(job, str(error))
                        continue

                started = time.perf_counter()
                try:
                    candidates = self._classifier.classify(job.crop)
                except Exception as error:
                    logger.exception("Crop classification failed")
                    self._finish_error(job, str(error))
                    continue
                elapsed = time.perf_counter() - started
                self.last_inference_seconds = elapsed
                self._finish_success(job, candidates, elapsed)
            finally:
                self._jobs.task_done()

    def _finish_error(self, job: _ClassificationJob, message: str) -> None:
        identity = (job.epoch, job.track_id, job.generation)
        with self._lock:
            self._pending.discard(identity)
            state = self._states.get(job.track_id)
            if job.epoch == self._epoch and state is not None and state.generation == job.generation:
                state.last_classification_time = job.submitted_at
                state.result = self._status_result(
                    state.result,
                    ClassificationStatus.UNCERTAIN,
                    time.perf_counter(),
                    error=message,
                )

    def _finish_success(
        self,
        job: _ClassificationJob,
        candidates: tuple[ClassificationCandidate, ...],
        elapsed: float,
    ) -> None:
        identity = (job.epoch, job.track_id, job.generation)
        completed_at = time.perf_counter()
        with self._lock:
            self._pending.discard(identity)
            state = self._states.get(job.track_id)
            if job.epoch != self._epoch or state is None or state.generation != job.generation:
                return
            valid = tuple(
                candidate
                for candidate in candidates[: config.CLASSIFICATION_TOP_K]
                if candidate.label and isfinite(candidate.score) and 0.0 <= candidate.score <= 1.0
            )
            state.last_classification_time = job.submitted_at
            state.observations.append(
                ClassificationPrediction(
                    candidates=valid,
                    model_id=self._classifier.model_id if self._classifier else "unknown",
                    timestamp=completed_at,
                )
            )
            status, refined_label, score = self._resolve_consensus(
                job.detector_label,
                state.observations,
            )
            alternatives = valid
            state.result = TrackRecognition(
                track_id=job.track_id,
                detector_label=job.detector_label,
                detector_confidence=job.detector_confidence,
                refined_label=refined_label,
                classification_score=score,
                model_id=self._classifier.model_id if self._classifier else None,
                status=status,
                timestamp=completed_at,
                alternatives=alternatives,
                error=(
                    "Classifier returned no valid candidate labels."
                    if not valid
                    else None
                ),
            )
            self.completed_count += 1
            self.last_inference_seconds = elapsed

    @staticmethod
    def _resolve_consensus(
        detector_label: str,
        observations: deque[ClassificationPrediction],
    ) -> tuple[ClassificationStatus, str | None, float | None]:
        if not observations or not observations[-1].candidates:
            return ClassificationStatus.UNCERTAIN, None, None
        latest = observations[-1].candidates
        top = latest[0]
        runner_up_score = latest[1].score if len(latest) > 1 else 0.0
        winners = [
            prediction.candidates[0].label
            for prediction in observations
            if prediction.candidates
        ]
        counts = Counter(_normalized_label(label) for label in winners)
        top_count = counts[_normalized_label(top.label)]
        if (
            top.score >= config.CLASSIFICATION_ACCEPTANCE_THRESHOLD
            and top.score - runner_up_score >= config.CLASSIFICATION_MINIMUM_MARGIN
            and top_count >= config.CLASSIFICATION_MINIMUM_AGREEMENTS
        ):
            if _labels_agree(top.label, detector_label):
                return ClassificationStatus.CONFIRMED, top.label, top.score
            return ClassificationStatus.REFINED, top.label, top.score
        return ClassificationStatus.UNCERTAIN, None, top.score

    @staticmethod
    def _status_result(
        previous: TrackRecognition,
        status: ClassificationStatus,
        timestamp: float,
        error: str | None,
    ) -> TrackRecognition:
        return TrackRecognition(
            track_id=previous.track_id,
            detector_label=previous.detector_label,
            detector_confidence=previous.detector_confidence,
            refined_label=previous.refined_label,
            classification_score=previous.classification_score,
            model_id=previous.model_id,
            status=status,
            timestamp=timestamp,
            alternatives=previous.alternatives,
            error=error,
        )
