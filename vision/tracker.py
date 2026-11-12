"""Persistent ByteTrack integration and per-object history management."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrackObservation:
    """One ByteTrack observation for a single frame."""

    track_id: int
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    timestamp: float


@dataclass(frozen=True)
class TrackedObject:
    """Current object state plus a bounded history of measured centers."""

    track_id: int
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    position_history: tuple[tuple[int, int], ...]
    timestamp: float
    age_seconds: float
    frames_tracked: int
    missed_frames: int


@dataclass(frozen=True)
class TrackerTiming:
    """Measured durations for Ultralytics tracking and local history bookkeeping."""

    ultralytics_track_seconds: float = 0.0
    adapter_seconds: float = 0.0


@dataclass
class _TrackState:
    track_id: int
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    first_seen: float
    timestamp: float
    positions: deque[tuple[int, int]]
    consecutive_frames: int = 0
    missed_frames: int = 0


class TrackHistoryManager:
    """Maintain bounded measured-position histories and selection state."""

    def __init__(self, history_length: int, max_missed_frames: int) -> None:
        if history_length < 1:
            raise ValueError("history_length must be at least 1")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must not be negative")

        self.history_length = history_length
        self.max_missed_frames = max_missed_frames
        self._states: dict[int, _TrackState] = {}
        self.selected_id: int | None = None
        self._last_timestamp = 0.0

    def update(
        self,
        observations: list[TrackObservation],
        timestamp: float,
    ) -> list[TrackedObject]:
        """Update cached state and return objects observed in this frame."""
        self._last_timestamp = timestamp
        visible_ids: set[int] = set()
        ordered_visible_ids: list[int] = []
        for observation in observations:
            track_id = observation.track_id
            visible_ids.add(track_id)
            if track_id not in ordered_visible_ids:
                ordered_visible_ids.append(track_id)
            state = self._states.get(track_id)
            if state is None:
                state = _TrackState(
                    track_id=track_id,
                    label=observation.label,
                    confidence=observation.confidence,
                    bbox=observation.bbox,
                    center=observation.center,
                    first_seen=timestamp,
                    timestamp=timestamp,
                    positions=deque(maxlen=self.history_length),
                )
                self._states[track_id] = state
            elif state.missed_frames:
                state.consecutive_frames = 0

            state.label = observation.label
            state.confidence = observation.confidence
            state.bbox = observation.bbox
            state.center = observation.center
            state.timestamp = timestamp
            state.positions.append(observation.center)
            state.consecutive_frames += 1
            state.missed_frames = 0

        for track_id, state in tuple(self._states.items()):
            if track_id not in visible_ids:
                state.missed_frames += 1
                state.consecutive_frames = 0
                if state.missed_frames > self.max_missed_frames:
                    del self._states[track_id]
                    if self.selected_id == track_id:
                        self.selected_id = None

        return [
            self._snapshot(self._states[track_id], timestamp)
            for track_id in ordered_visible_ids
        ]

    @property
    def position_histories(self) -> dict[int, tuple[tuple[int, int], ...]]:
        """Return copies of retained position histories, including missed tracks."""
        return {
            track_id: tuple(state.positions)
            for track_id, state in self._states.items()
        }

    @property
    def retained_tracks(self) -> list[TrackedObject]:
        """Return snapshots for active and temporarily missing tracks."""
        return [
            self._snapshot(state, self._last_timestamp)
            for state in self._states.values()
        ]

    def select(self, track_id: int | None) -> bool:
        """Select a retained track ID, or clear selection with ``None``."""
        if track_id is None:
            self.selected_id = None
            return True
        if track_id not in self._states:
            return False
        self.selected_id = track_id
        return True

    def cycle_selection(self, visible_ids: list[int]) -> int | None:
        """Select the next visible ID, retaining selection if none are visible."""
        ordered_ids = sorted(set(visible_ids))
        if not ordered_ids:
            return self.selected_id
        if self.selected_id not in ordered_ids:
            self.selected_id = ordered_ids[0]
            return self.selected_id

        current_index = ordered_ids.index(self.selected_id)
        self.selected_id = ordered_ids[(current_index + 1) % len(ordered_ids)]
        return self.selected_id

    @staticmethod
    def _snapshot(state: _TrackState, timestamp: float) -> TrackedObject:
        return TrackedObject(
            track_id=state.track_id,
            label=state.label,
            confidence=state.confidence,
            bbox=state.bbox,
            center=state.center,
            position_history=tuple(state.positions),
            timestamp=state.timestamp,
            age_seconds=max(0.0, timestamp - state.first_seen),
            frames_tracked=state.consecutive_frames,
            missed_frames=state.missed_frames,
        )


class ByteTrackTracker:
    """Run Ultralytics ByteTrack and retain history between returned detections."""

    def __init__(
        self,
        model: Any,
        tracker_config: str | Path,
        inference_confidence: float,
        history_length: int,
        max_missed_frames: int,
        device: str | None = None,
        inference_size: int = 640,
    ) -> None:
        if not 0.0 <= inference_confidence <= 1.0:
            raise ValueError("inference_confidence must be between 0.0 and 1.0")
        if inference_size < 32:
            raise ValueError("inference_size must be at least 32 pixels")
        self.model = model
        self.tracker_config = Path(tracker_config)
        if not self.tracker_config.is_file():
            raise RuntimeError(
                f"ByteTrack configuration not found: {self.tracker_config}"
            )
        self.inference_confidence = inference_confidence
        self.device = device
        self.inference_size = inference_size
        self.history = TrackHistoryManager(history_length, max_missed_frames)
        self._history_length = history_length
        self._max_missed_frames = max_missed_frames
        self.last_timing = TrackerTiming()

    @property
    def selected_id(self) -> int | None:
        return self.history.selected_id

    @property
    def position_histories(self) -> dict[int, tuple[tuple[int, int], ...]]:
        return self.history.position_histories

    @property
    def retained_tracks(self) -> list[TrackedObject]:
        return self.history.retained_tracks

    def cycle_selection(self, visible_ids: list[int]) -> int | None:
        return self.history.cycle_selection(visible_ids)

    def select(self, track_id: int | None) -> bool:
        """Select an ID directly, for keyboard or future gesture controls."""
        return self.history.select(track_id)

    def reset(self) -> None:
        """Clear local tracks and reset Ultralytics' persistent ByteTrack state."""
        self.history = TrackHistoryManager(
            self._history_length,
            self._max_missed_frames,
        )
        predictor = getattr(self.model, "predictor", None)
        for byte_tracker in getattr(predictor, "trackers", ()):
            byte_tracker.reset()

    def update(self, frame: np.ndarray, timestamp: float) -> list[TrackedObject]:
        """Track one frame using the persistent Ultralytics tracking API."""
        call_started = perf_counter()
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=str(self.tracker_config),
            conf=self.inference_confidence,
            imgsz=self.inference_size,
            device=self.device,
            verbose=False,
        )
        call_finished = perf_counter()

        adapter_started = call_finished
        observations: list[TrackObservation] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue
            coordinates = _as_list(boxes.xyxy)
            confidences = _as_list(boxes.conf)
            class_ids = _as_list(boxes.cls)
            track_ids = _as_list(boxes.id)
            names = result.names
            for box, confidence, class_id, track_id in zip(
                coordinates, confidences, class_ids, track_ids
            ):
                x1, y1, x2, y2 = (int(value) for value in box)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                class_index = int(class_id)
                label = (
                    names.get(class_index, str(class_index))
                    if isinstance(names, dict)
                    else names[class_index]
                )
                observations.append(
                    TrackObservation(
                        track_id=int(track_id),
                        label=label,
                        confidence=float(confidence),
                        bbox=(x1, y1, x2, y2),
                        center=center,
                        timestamp=timestamp,
                    )
                )
        tracks = self.history.update(observations, timestamp)
        adapter_finished = perf_counter()
        self.last_timing = TrackerTiming(
            ultralytics_track_seconds=call_finished - call_started,
            adapter_seconds=adapter_finished - adapter_started,
        )
        return tracks


def _as_list(values: Any) -> list[Any]:
    """Convert NumPy arrays or Torch tensors to ordinary Python lists."""
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)
