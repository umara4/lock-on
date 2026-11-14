"""Configurable geometric hand-pose matching using existing MediaPipe results."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
from math import cos, hypot, radians
from pathlib import Path
from typing import Any

import config
from vision.gesture_features import (
    HandFeatures,
    extract_hand_features,
    score_finger_states,
)
from vision.gesture_sequences import (
    GestureSequence,
    GestureSequenceRecognizer,
    SequenceEvent,
    SequenceProgress,
)
from vision.gestures import DetectedHand, GestureType


@dataclass(frozen=True)
class HandObservation:
    identity: str
    handedness: str | None
    landmarks: tuple[tuple[float, float, float], ...]
    center: tuple[float, float]
    palm_scale: float
    confidence: float
    features: HandFeatures
    command_gesture: GestureType
    history: tuple[tuple[float, float, float, float], ...]
    seconds_since_detection: float
    visible: bool


class PoseStatus(str):
    UNKNOWN = "UNKNOWN"
    RECOGNIZING = "RECOGNIZING"
    RECOGNIZED = "RECOGNIZED"
    HELD = "HELD"
    RELEASED = "RELEASED"


@dataclass(frozen=True)
class PoseMatch:
    pose_id: str
    display_name: str
    status: str
    confidence: float
    held_seconds: float
    hand_ids: tuple[str, ...]
    center: tuple[float, float]


@dataclass(frozen=True)
class AnimeGestureEvent:
    gesture_id: str
    display_name: str
    kind: str
    confidence: float
    hand_ids: tuple[str, ...]
    position: tuple[float, float]


@dataclass(frozen=True)
class AnimeRecognitionSnapshot:
    hands: tuple[HandObservation, ...] = ()
    pose: PoseMatch | None = None
    events: tuple[AnimeGestureEvent, ...] = ()
    sequence_progress: tuple[SequenceProgress, ...] = ()
    sequence_events: tuple[SequenceEvent, ...] = ()


@dataclass(frozen=True)
class _PoseTemplate:
    pose_id: str
    display_name: str
    description: str
    kind: str
    finger_states: tuple[bool, ...]
    min_hold_seconds: float
    min_score: float
    priority: int
    max_center_distance_scales: float = 0.0
    max_vertical_offset_scales: float = 0.0
    min_orientation_alignment: float = 0.0


@dataclass
class _HandTrack:
    identity: str
    handedness: str | None
    center: tuple[float, float]
    palm_scale: float
    last_seen: float
    landmarks: tuple[tuple[float, float, float], ...]
    features: HandFeatures
    command_gesture: GestureType
    confidence: float
    history: deque[tuple[float, float, float, float]] = field(
        default_factory=lambda: deque(maxlen=32)
    )


class AnimeGestureRecognizer:
    """Track hands and recognize conservative single-/two-hand templates."""

    def __init__(
        self,
        template_path: str | Path | None = None,
        maximum_lost_seconds: float = config.ANIME_HAND_LOST_SECONDS,
        cooldown_seconds: float = config.ANIME_GESTURE_COOLDOWN_SECONDS,
        sequence_timeout_seconds: float = config.ANIME_SEQUENCE_TIMEOUT_SECONDS,
    ) -> None:
        if maximum_lost_seconds <= 0 or cooldown_seconds < 0:
            raise ValueError("hand loss timeout must be positive and cooldown non-negative")
        self.maximum_lost_seconds = maximum_lost_seconds
        self.cooldown_seconds = cooldown_seconds
        selected_path = Path(template_path or config.ANIME_GESTURE_TEMPLATES_PATH)
        if not selected_path.is_absolute() and not selected_path.is_file():
            selected_path = Path(__file__).with_name(selected_path.name)
        payload = json.loads(selected_path.read_text(encoding="utf-8"))
        self._templates = tuple(self._parse_template(item) for item in payload["templates"])
        self._templates = tuple(sorted(self._templates, key=lambda item: -item.priority))
        sequence_values = payload.get("sequences", [])
        self._sequences = GestureSequenceRecognizer(
            GestureSequence(
                sequence_id=item["id"],
                display_name=item["display_name"],
                steps=tuple(item["steps"]),
                max_gap_seconds=min(
                    float(item.get("max_gap_seconds", sequence_timeout_seconds)),
                    sequence_timeout_seconds,
                ),
                completion_effect=item.get("completion_effect", ""),
            )
            for item in sequence_values
        )
        self._hand_tracks: dict[str, _HandTrack] = {}
        self._next_identity = 1
        self._active_pose: str | None = None
        self._candidate_pose: str | None = None
        self._candidate_since: float | None = None
        self._last_activation: dict[str, float] = {}
        self._active_hand_ids: tuple[str, ...] = ()
        self._active_center = (0.5, 0.5)
        self._aspect_ratio = 1.0
        self._last_snapshot = AnimeRecognitionSnapshot()

    @staticmethod
    def _parse_template(item: dict[str, Any]) -> _PoseTemplate:
        finger_states = tuple(item["finger_states"])
        if len(finger_states) != 5 or any(not isinstance(value, bool) for value in finger_states):
            raise ValueError(f"template {item.get('id')} must define five boolean finger states")
        kind = item.get("kind")
        if kind not in {"single", "two_hand"}:
            raise ValueError(f"unsupported hand template kind: {kind!r}")
        return _PoseTemplate(
            pose_id=item["id"],
            display_name=item["display_name"],
            description=item["description"],
            kind=kind,
            finger_states=finger_states,
            min_hold_seconds=float(item["min_hold_seconds"]),
            min_score=float(item["min_score"]),
            priority=int(item.get("priority", 0)),
            max_center_distance_scales=float(item.get("max_center_distance_scales", 0.0)),
            max_vertical_offset_scales=float(item.get("max_vertical_offset_scales", 0.0)),
            min_orientation_alignment=float(item.get("min_orientation_alignment", 0.0)),
        )

    @property
    def snapshot(self) -> AnimeRecognitionSnapshot:
        return self._last_snapshot

    def reset(self) -> None:
        self._hand_tracks.clear()
        self._active_pose = None
        self._candidate_pose = None
        self._candidate_since = None
        self._last_activation.clear()
        self._active_hand_ids = ()
        self._active_center = (0.5, 0.5)
        self._sequences.reset()
        self._last_snapshot = AnimeRecognitionSnapshot()

    def update(
        self,
        detected_hands: tuple[DetectedHand, ...],
        timestamp: float,
        frame_size: tuple[int, int] | None = None,
    ) -> AnimeRecognitionSnapshot:
        if frame_size is not None:
            width, height = frame_size
            if width < 1 or height < 1:
                raise ValueError("frame dimensions must be positive")
            self._aspect_ratio = width / height
        hands = self._track_hands(detected_hands, timestamp)
        matches = [
            (template, confidence, hand_ids, center)
            for template in self._templates
            if (
                matched := self._match_template(template, hands)
            ) is not None
            for confidence, hand_ids, center in (matched,)
            if confidence >= max(template.min_score, config.ANIME_MIN_GESTURE_CONFIDENCE)
        ]
        best = max(matches, key=lambda match: match[0].priority) if matches else None
        events: list[AnimeGestureEvent] = []
        pose: PoseMatch | None = None
        sequence_input: list[tuple[str, str]] = []

        if best is not None:
            template, confidence, hand_ids, center = best
            if self._candidate_pose != template.pose_id:
                if self._active_pose is not None:
                    events.append(
                        self._event(
                            self._active_pose,
                            "released",
                            1.0,
                            self._active_hand_ids,
                            self._active_center,
                        )
                    )
                    sequence_input.append(("released", self._active_pose))
                    self._active_hand_ids = ()
                self._candidate_pose = template.pose_id
                self._candidate_since = timestamp
                self._active_pose = None
            held_seconds = (
                max(0.0, timestamp - self._candidate_since)
                if self._candidate_since is not None
                else 0.0
            )
            status = PoseStatus.RECOGNIZING
            if held_seconds >= template.min_hold_seconds:
                last_activation = self._last_activation.get(template.pose_id, float("-inf"))
                if self._active_pose is None and timestamp - last_activation >= self.cooldown_seconds:
                    self._active_pose = template.pose_id
                    self._active_hand_ids = hand_ids
                    self._active_center = center
                    self._last_activation[template.pose_id] = timestamp
                    events.append(
                        self._event(template.pose_id, "recognized", confidence, hand_ids, center)
                    )
                    sequence_input.append(("recognized", template.pose_id))
                    status = PoseStatus.RECOGNIZED
                elif self._active_pose == template.pose_id:
                    status = PoseStatus.HELD
            pose = PoseMatch(
                template.pose_id,
                template.display_name,
                status,
                confidence,
                held_seconds,
                hand_ids,
                center,
            )
        else:
            if self._candidate_pose is not None:
                if self._active_pose is not None:
                    events.append(
                        self._event(
                            self._active_pose,
                            "released",
                            1.0,
                            self._active_hand_ids,
                            self._active_center,
                        )
                    )
                    sequence_input.append(("released", self._active_pose))
                    self._active_hand_ids = ()
                self._candidate_pose = None
                self._candidate_since = None
                self._active_pose = None
                pose = PoseMatch(
                    "",
                    "RELEASED",
                    PoseStatus.RELEASED,
                    1.0,
                    0.0,
                    (),
                    (0.5, 0.5),
                )

        progress, sequence_events = self._sequences.update(sequence_input, timestamp)
        self._last_snapshot = AnimeRecognitionSnapshot(
            hands=hands,
            pose=pose,
            events=tuple(events),
            sequence_progress=progress,
            sequence_events=sequence_events,
        )
        return self._last_snapshot

    def _track_hands(
        self,
        detected_hands: tuple[DetectedHand, ...],
        timestamp: float,
    ) -> tuple[HandObservation, ...]:
        tracked = tuple(
            (hand, extract_hand_features(hand.landmarks, self._aspect_ratio))
            for hand in detected_hands
        )
        tracked = tuple((hand, features) for hand, features in tracked if features is not None)
        candidates: list[tuple[float, str, int]] = []
        for identity, prior in self._hand_tracks.items():
            age = timestamp - prior.last_seen
            if age < 0 or age > self.maximum_lost_seconds:
                continue
            for index, (hand, features) in enumerate(tracked):
                assert features is not None
                distance = hypot(
                    (features.center[0] - prior.center[0]) * self._aspect_ratio,
                    features.center[1] - prior.center[1],
                ) / max(prior.palm_scale, features.palm_scale, 1e-6)
                if hand.handedness and prior.handedness and hand.handedness != prior.handedness:
                    distance += 0.75
                candidates.append((distance, identity, index))

        assignments: dict[int, str] = {}
        used_identities: set[str] = set()
        used_observations: set[int] = set()
        for distance, identity, index in sorted(candidates):
            if distance > 3.0 or identity in used_identities or index in used_observations:
                continue
            assignments[index] = identity
            used_identities.add(identity)
            used_observations.add(index)

        for index in range(len(tracked)):
            if index not in assignments:
                identity = f"hand-{self._next_identity}"
                self._next_identity += 1
                assignments[index] = identity

        observations: list[HandObservation] = []
        visible_ids: set[str] = set()
        for index, (hand, features) in enumerate(tracked):
            assert features is not None
            identity = assignments[index]
            visible_ids.add(identity)
            prior = self._hand_tracks.get(identity)
            history = prior.history if prior is not None else deque(maxlen=32)
            history.append(
                (
                    timestamp,
                    features.center[0],
                    features.center[1],
                    features.center[2],
                )
            )
            state = _HandTrack(
                identity,
                hand.handedness,
                features.center[:2],
                features.palm_scale,
                timestamp,
                hand.landmarks,
                features,
                hand.gesture,
                hand.tracking_confidence,
                history,
            )
            self._hand_tracks[identity] = state
            observations.append(
                HandObservation(
                    identity,
                    hand.handedness,
                    hand.landmarks,
                    features.center[:2],
                    features.palm_scale,
                    hand.tracking_confidence,
                    features,
                    hand.gesture,
                    tuple(history),
                    0.0,
                    True,
                )
            )

        for identity, state in tuple(self._hand_tracks.items()):
            age = timestamp - state.last_seen
            if age > self.maximum_lost_seconds:
                del self._hand_tracks[identity]
        observations.extend(
            HandObservation(
                state.identity,
                state.handedness,
                state.landmarks,
                state.center,
                state.palm_scale,
                state.confidence,
                state.features,
                state.command_gesture,
                tuple(state.history),
                max(0.0, timestamp - state.last_seen),
                False,
            )
            for identity, state in self._hand_tracks.items()
            if identity not in visible_ids
        )
        return tuple(observations)

    def _match_template(
        self,
        template: _PoseTemplate,
        hands: tuple[HandObservation, ...],
    ) -> tuple[float, tuple[str, ...], tuple[float, float]] | None:
        visible = tuple(hand for hand in hands if hand.visible)
        visible = tuple(
            hand
            for hand in visible
            if hand.command_gesture in (GestureType.NONE, GestureType.OTHER)
        )
        if template.kind == "single":
            candidates = [
                hand
                for hand in visible
                if self._finger_score(hand.features, template.finger_states) >= template.min_score
            ]
            if not candidates:
                return None
            hand = max(
                candidates,
                key=lambda item: self._finger_score(item.features, template.finger_states),
            )
            return (
                self._finger_score(hand.features, template.finger_states),
                (hand.identity,),
                hand.center,
            )
        if len(visible) != 2:
            return None
        first, second = visible
        first_score = self._finger_score(first.features, template.finger_states)
        second_score = self._finger_score(second.features, template.finger_states)
        if min(first_score, second_score) < template.min_score:
            return None
        mean_scale = max((first.palm_scale + second.palm_scale) / 2.0, 1e-6)
        dx = (first.center[0] - second.center[0]) * self._aspect_ratio
        dy = first.center[1] - second.center[1]
        if hypot(dx, dy) / mean_scale > template.max_center_distance_scales:
            return None
        if abs(dy) / mean_scale > template.max_vertical_offset_scales:
            return None
        orientation_a = first.features.palm_orientation_degrees
        orientation_b = second.features.palm_orientation_degrees
        angle_delta = abs(orientation_a - orientation_b) % 180.0
        alignment = abs(cos(radians(angle_delta)))
        if alignment < template.min_orientation_alignment:
            return None
        center = (
            (first.center[0] + second.center[0]) / 2.0,
            (first.center[1] + second.center[1]) / 2.0,
        )
        return min(first_score, second_score), (first.identity, second.identity), center

    @staticmethod
    def _finger_score(features: HandFeatures, expected: tuple[bool, ...]) -> float:
        return score_finger_states(features, expected)

    def _event(
        self,
        pose_id: str,
        kind: str,
        confidence: float,
        hand_ids: tuple[str, ...],
        position: tuple[float, float],
    ) -> AnimeGestureEvent:
        template = next(item for item in self._templates if item.pose_id == pose_id)
        return AnimeGestureEvent(
            pose_id,
            template.display_name,
            kind,
            confidence,
            hand_ids,
            position,
        )
