"""Feature-only gesture sample recording for future template calibration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import cos, isfinite, radians
from pathlib import Path

from vision.gesture_features import extract_hand_features
from vision.gestures import DetectedHand


@dataclass(frozen=True)
class RecordedGestureSample:
    gesture_id: str
    display_name: str
    timestamp: float
    handedness: tuple[str | None, ...]
    feature_vectors: tuple[dict[str, object], ...]
    relative_hand_features: dict[str, float] | None


class GestureSampleRecorder:
    """Persist derived hand features only; never save camera frames or raw video."""

    def __init__(self, maximum_samples_per_gesture: int = 64) -> None:
        if maximum_samples_per_gesture < 1:
            raise ValueError("maximum_samples_per_gesture must be positive")
        self.maximum_samples_per_gesture = maximum_samples_per_gesture
        self._samples: dict[str, list[RecordedGestureSample]] = {}

    @property
    def sample_counts(self) -> dict[str, int]:
        return {gesture_id: len(samples) for gesture_id, samples in self._samples.items()}

    def clear(self, gesture_id: str | None = None) -> None:
        if gesture_id is None:
            self._samples.clear()
        else:
            self._samples.pop(gesture_id, None)

    def record(
        self,
        gesture_id: str,
        display_name: str,
        hands: tuple[DetectedHand, ...],
        timestamp: float,
        aspect_ratio: float = 1.0,
    ) -> RecordedGestureSample:
        if not gesture_id.strip() or not display_name.strip():
            raise ValueError("gesture id and display name must not be empty")
        if len(hands) not in (1, 2):
            raise ValueError("gesture samples must contain one or two detected hands")
        if not isfinite(timestamp):
            raise ValueError("sample timestamp must be finite")
        features = tuple(
            extract_hand_features(hand.landmarks, aspect_ratio)
            for hand in hands
        )
        if any(feature is None for feature in features):
            raise ValueError("every recorded hand must have 21 finite landmarks")
        valid_features = tuple(feature for feature in features if feature is not None)
        vector_records = tuple(
            {
                "finger_extension_ratios": list(feature.finger_extension_ratios),
                "finger_states": list(feature.finger_states),
                "fingertip_distances": list(feature.fingertip_distances),
                "joint_angles_degrees": list(feature.joint_angles_degrees),
                "palm_orientation_degrees": feature.palm_orientation_degrees,
            }
            for feature in valid_features
        )
        relative = None
        if len(valid_features) == 2:
            first, second = valid_features
            dx = first.center[0] - second.center[0]
            dy = first.center[1] - second.center[1]
            mean_scale = max((first.palm_scale + second.palm_scale) / 2.0, 1e-8)
            relative = {
                "center_distance_scales": ((dx * aspect_ratio) ** 2 + dy**2) ** 0.5
                / mean_scale,
                "vertical_offset_scales": abs(dy) / mean_scale,
                "orientation_alignment": abs(
                    cos(
                        radians(
                            first.palm_orientation_degrees
                            - second.palm_orientation_degrees
                        )
                    )
                ),
            }
        sample = RecordedGestureSample(
            gesture_id,
            display_name,
            timestamp,
            tuple(hand.handedness for hand in hands),
            vector_records,
            relative,
        )
        samples = self._samples.setdefault(gesture_id, [])
        if len(samples) >= self.maximum_samples_per_gesture:
            samples.pop(0)
        samples.append(sample)
        return sample

    def to_payload(self) -> dict[str, object]:
        """Return a versioned feature-only format extensible to actions/sequences."""
        return {
            "schema": "jarvis.gesture-features.v1",
            "samples": [
                {
                    "gesture_id": sample.gesture_id,
                    "display_name": sample.display_name,
                    "timestamp": sample.timestamp,
                    "handedness": list(sample.handedness),
                    "hands": list(sample.feature_vectors),
                    "relative_hand_features": sample.relative_hand_features,
                }
                for samples in self._samples.values()
                for sample in samples
            ],
            "sequences": [],
            "action_mappings": {},
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_payload(), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
