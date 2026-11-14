"""Persistent calibration from derived hand features; camera frames are never saved."""

from __future__ import annotations

from itertools import permutations
import json
from math import hypot, isfinite, sqrt
from pathlib import Path

from vision.anime_gestures import HandObservation


class AbilityCalibrationStore:
    """Store pose samples and score runtime hand features against them."""

    def __init__(
        self,
        path: str | Path,
        maximum_samples_per_ability: int = 48,
        minimum_similarity: float = 0.72,
    ) -> None:
        if maximum_samples_per_ability < 1:
            raise ValueError("maximum samples must be positive")
        if not 0.0 < minimum_similarity <= 1.0:
            raise ValueError("minimum similarity must be in (0, 1]")
        self.path = Path(path)
        self.maximum_samples_per_ability = maximum_samples_per_ability
        self.minimum_similarity = minimum_similarity
        self._samples: dict[str, list[tuple[tuple[float, ...], ...]]] = {}
        self._landmark_samples: dict[
            str,
            list[tuple[tuple[float, ...], ...] | None],
        ] = {}
        self._movement_samples: dict[str, list[list[dict[str, object]]]] = {}
        self._recording_ability: str | None = None
        self._movement_started_at: float | None = None
        self._movement_buffer: list[dict[str, object]] = []
        if self.path.is_file():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("schema") != "jarvis.ability-calibration.v1":
                raise ValueError(f"unsupported ability calibration file: {self.path}")
            stored_similarity = payload.get("minimum_similarity")
            if stored_similarity is not None:
                stored_similarity = float(stored_similarity)
                if not isfinite(stored_similarity) or not 0.0 < stored_similarity <= 1.0:
                    raise ValueError("stored minimum similarity must be in (0, 1]")
                self.minimum_similarity = stored_similarity
            for sample in payload.get("samples", []):
                ability_id = str(sample["ability_id"])
                hands = tuple(
                    tuple(float(value) for value in hand["finger_extension_ratios"])
                    for hand in sample["hands"]
                )
                if len(hands) not in (1, 2) or any(
                    len(features) != 5
                    or not all(isfinite(value) for value in features)
                    for features in hands
                ):
                    raise ValueError("calibration samples require one or two valid hands")
                self._samples.setdefault(ability_id, []).append(hands)
                normalized = sample.get("normalized_landmarks")
                if normalized is None:
                    self._landmark_samples.setdefault(ability_id, []).append(None)
                else:
                    landmark_hands = tuple(
                        tuple(float(value) for value in hand)
                        for hand in normalized
                    )
                    if len(landmark_hands) not in (1, 2) or any(
                        len(points) != 63
                        or not all(isfinite(value) for value in points)
                        for points in landmark_hands
                    ):
                        raise ValueError("calibration samples require valid 21-point hands")
                    self._landmark_samples.setdefault(ability_id, []).append(
                        landmark_hands
                    )
            for samples in self._samples.values():
                del samples[maximum_samples_per_ability:]
            for samples in self._landmark_samples.values():
                del samples[maximum_samples_per_ability:]
            for sample in payload.get("movement_samples", []):
                ability_id = str(sample["ability_id"])
                frames = sample["frames"]
                if not isinstance(frames, list):
                    raise ValueError("calibrated movement sample frames must be a list")
                self._movement_samples.setdefault(ability_id, []).append(frames)
            for samples in self._movement_samples.values():
                del samples[maximum_samples_per_ability:]

    @property
    def sample_counts(self) -> dict[str, int]:
        return {ability_id: len(samples) for ability_id, samples in self._samples.items()}

    def has_samples(self, ability_id: str) -> bool:
        return bool(self._samples.get(ability_id))

    @property
    def movement_sample_counts(self) -> dict[str, int]:
        return {
            ability_id: len(samples)
            for ability_id, samples in self._movement_samples.items()
        }

    @property
    def recording_ability(self) -> str | None:
        return self._recording_ability

    @property
    def recording_frames(self) -> int:
        return len(self._movement_buffer)

    def set_minimum_similarity(self, value: float) -> None:
        if not isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError("minimum similarity must be finite and in (0, 1]")
        self.minimum_similarity = value
        self.save()

    def begin_movement_recording(self, ability_id: str, timestamp: float) -> None:
        if not ability_id:
            raise ValueError("ability id must not be empty")
        if not isfinite(timestamp):
            raise ValueError("recording timestamp must be finite")
        self._recording_ability = ability_id
        self._movement_started_at = None
        self._movement_buffer = []

    def append_movement_frame(
        self,
        hands: tuple[HandObservation, ...],
        timestamp: float,
    ) -> None:
        if self._recording_ability is None:
            return
        if not isfinite(timestamp):
            raise ValueError("movement sample timestamps must be finite and monotonic")
        if self._movement_started_at is None:
            self._movement_started_at = timestamp
        if timestamp < self._movement_started_at:
            raise ValueError("movement sample timestamps must be finite and monotonic")
        if len(self._movement_buffer) >= 128:
            return
        visible = tuple(hand for hand in hands if hand.visible)
        self._movement_buffer.append(
            {
                "time_seconds": timestamp - self._movement_started_at,
                "hands": [
                    {
                        "center": list(hand.center),
                        "center_3d": list(hand.features.center),
                        "palm_scale": hand.palm_scale,
                        "finger_extension_ratios": list(
                            hand.features.finger_extension_ratios
                        ),
                    }
                    for hand in visible
                ],
            }
        )

    def finish_movement_recording(self) -> int:
        ability_id = self._recording_ability
        frames = self._movement_buffer
        self._recording_ability = None
        self._movement_started_at = None
        self._movement_buffer = []
        valid_frames = sum(bool(frame["hands"]) for frame in frames)
        if ability_id is None or len(frames) < 3 or valid_frames < 3:
            raise ValueError("record at least three movement observations with visible hands")
        samples = self._movement_samples.setdefault(ability_id, [])
        if len(samples) >= self.maximum_samples_per_ability:
            samples.pop(0)
        samples.append(frames)
        self.save()
        return len(samples)

    def record(
        self,
        ability_id: str,
        hands: tuple[HandObservation, ...],
        expected_hand_count: int | None = None,
        aspect_ratio: float = 1.0,
    ) -> int:
        if not ability_id:
            raise ValueError("ability id must not be empty")
        visible = tuple(hand for hand in hands if hand.visible)
        if expected_hand_count is not None and len(visible) != expected_hand_count:
            raise ValueError(
                f"calibration requires exactly {expected_hand_count} visible hand(s)"
            )
        if len(visible) not in (1, 2) or not isfinite(aspect_ratio) or aspect_ratio <= 0.0:
            raise ValueError("calibration requires one or two valid visible hands")
        features = tuple(
            tuple(hand.features.finger_extension_ratios)
            for hand in visible
        )
        landmarks = tuple(
            self._normalized_landmarks(hand, aspect_ratio)
            for hand in visible
        )
        samples = self._samples.setdefault(ability_id, [])
        landmark_samples = self._landmark_samples.setdefault(ability_id, [])
        if len(samples) >= self.maximum_samples_per_ability:
            samples.pop(0)
            landmark_samples.pop(0)
        samples.append(features)
        landmark_samples.append(landmarks)
        self.save()
        return len(samples)

    def landmark_similarity(
        self,
        ability_id: str,
        hands: tuple[HandObservation, ...],
        aspect_ratio: float = 1.0,
    ) -> float | None:
        visible = tuple(hand for hand in hands if hand.visible)
        if (
            len(visible) not in (1, 2)
            or not isfinite(aspect_ratio)
            or aspect_ratio <= 0.0
        ):
            return None
        current = tuple(
            self._normalized_landmarks(hand, aspect_ratio)
            for hand in visible
        )
        best = 0.0
        for sample in self._landmark_samples.get(ability_id, ()):
            if sample is None or len(sample) != len(current):
                continue
            for ordering in permutations(current):
                rmse = sqrt(
                    sum(
                        (actual - expected) ** 2
                        for actual_hand, expected_hand in zip(ordering, sample)
                        for actual, expected in zip(actual_hand, expected_hand)
                    )
                    / (len(sample) * 63)
                )
                best = max(best, max(0.0, 1.0 - rmse / 0.55))
        return best if best > 0.0 else None

    def similarity(
        self,
        ability_id: str,
        hands: tuple[HandObservation, ...],
    ) -> float | None:
        samples = self._samples.get(ability_id, ())
        visible = tuple(hand for hand in hands if hand.visible)
        if not samples or len(visible) not in (1, 2):
            return None
        current = tuple(hand.features.finger_extension_ratios for hand in visible)
        best = 0.0
        for sample in samples:
            if len(sample) != len(current):
                continue
            for ordering in permutations(current):
                squared_error = sum(
                    (actual - expected) ** 2
                    for actual_hand, expected_hand in zip(ordering, sample)
                    for actual, expected in zip(actual_hand, expected_hand)
                )
                rmse = sqrt(squared_error / (len(sample) * 5))
                best = max(best, max(0.0, 1.0 - rmse / 0.8))
        return best

    def accepts(self, ability_id: str, hands: tuple[HandObservation, ...]) -> bool:
        score = self.similarity(ability_id, hands)
        return score is None or score >= self.minimum_similarity

    def movement_threshold(self, ability_id: str, fallback: float) -> float:
        """Estimate a conservative palm-relative trigger from recorded sequences."""
        magnitudes: list[float] = []
        for sample in self._movement_samples.get(ability_id, ()):
            valid = [frame for frame in sample if frame.get("hands")]
            if len(valid) < 3:
                continue
            start_hands = valid[0]["hands"]
            end_hands = valid[-1]["hands"]
            if not isinstance(start_hands, list) or not isinstance(end_hands, list):
                continue
            if not start_hands or not end_hands:
                continue
            start_center = self._mean_center(start_hands)
            end_center = self._mean_center(end_hands)
            scale = (
                self._mean_scale(start_hands) + self._mean_scale(end_hands)
            ) / 2.0
            if scale > 1e-8:
                dx = end_center[0] - start_center[0]
                dy = end_center[1] - start_center[1]
                depth = 0.0
                if ability_id == "kamehameha":
                    start_depth = self._mean_depth(start_hands)
                    end_depth = self._mean_depth(end_hands)
                    depth = end_depth - start_depth
                magnitudes.append(hypot(dx, dy, depth) / scale)
        if not magnitudes or max(magnitudes) < 0.1:
            return fallback
        ordered = sorted(magnitudes)
        median = ordered[len(ordered) // 2]
        return min(2.5, max(0.2, median * 0.6))

    @staticmethod
    def _mean_center(hands: list[dict[str, object]]) -> tuple[float, float]:
        centers = [hand["center"] for hand in hands if isinstance(hand.get("center"), list)]
        if not centers:
            return 0.0, 0.0
        return (
            sum(float(center[0]) for center in centers) / len(centers),
            sum(float(center[1]) for center in centers) / len(centers),
        )

    @staticmethod
    def _mean_scale(hands: list[dict[str, object]]) -> float:
        scales = [float(hand["palm_scale"]) for hand in hands if "palm_scale" in hand]
        return sum(scales) / len(scales) if scales else 0.0

    @staticmethod
    def _mean_depth(hands: list[dict[str, object]]) -> float:
        depths = [
            float(center[2])
            for hand in hands
            if isinstance((center := hand.get("center_3d")), list)
            and len(center) == 3
        ]
        return sum(depths) / len(depths) if depths else 0.0

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        samples: list[dict[str, object]] = []
        for ability_id, values in sorted(self._samples.items()):
            landmarks = self._landmark_samples.get(ability_id, [])
            for index, hands in enumerate(values):
                landmark_hands = landmarks[index] if index < len(landmarks) else None
                samples.append(
                    {
                        "ability_id": ability_id,
                        "hands": [
                            {"finger_extension_ratios": list(hand)}
                            for hand in hands
                        ],
                        "normalized_landmarks": (
                            [list(hand) for hand in landmark_hands]
                            if landmark_hands is not None
                            else None
                        ),
                    }
                )
        self.path.write_text(
            json.dumps(
                {
                    "schema": "jarvis.ability-calibration.v1",
                    "minimum_similarity": self.minimum_similarity,
                    "samples": samples,
                    "movement_samples": [
                        {
                            "ability_id": ability_id,
                            "frames": frames,
                        }
                        for ability_id, values in sorted(self._movement_samples.items())
                        for frames in values
                    ],
                },
                indent=2,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _normalized_landmarks(
        hand: HandObservation,
        aspect_ratio: float,
    ) -> tuple[float, ...]:
        if len(hand.landmarks) != 21:
            raise ValueError("pose calibration requires all 21 hand landmarks")
        wrist = hand.landmarks[0]
        middle_mcp = hand.landmarks[9]
        scale = sqrt(
            ((middle_mcp[0] - wrist[0]) * aspect_ratio) ** 2
            + (middle_mcp[1] - wrist[1]) ** 2
            + ((middle_mcp[2] - wrist[2]) * aspect_ratio) ** 2
        )
        if scale <= 1e-8:
            raise ValueError("pose calibration has a degenerate palm scale")
        values: list[float] = []
        for x, y, z in hand.landmarks:
            values.extend(
                (
                    (x - wrist[0]) * aspect_ratio / scale,
                    (y - wrist[1]) / scale,
                    (z - wrist[2]) * aspect_ratio / scale,
                )
            )
        return tuple(values)
