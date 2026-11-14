"""Timestamped, release-gated recognition for the initial ability library."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, sqrt
from typing import Literal

import config
from vision.ability_calibration import AbilityCalibrationStore
from vision.ability_library import AbilityDefinition, load_ability_library
from vision.anime_gestures import AnimeRecognitionSnapshot, HandObservation
from vision.gestures import GestureType
from vision.gesture_sequences import SequenceProgress

AbilityPhase = Literal["charging", "activated", "released"]


@dataclass(frozen=True)
class AbilityEvent:
    ability_id: str
    display_name: str
    franchise: str
    phase: AbilityPhase
    position: tuple[float, float]
    direction: tuple[float, float]
    hand_ids: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class AbilityRecognitionSnapshot:
    selected_ability_id: str | None = None
    phase: str = "READY"
    position: tuple[float, float] = (0.5, 0.5)
    direction: tuple[float, float] = (1.0, 0.0)
    hand_ids: tuple[str, ...] = ()
    confidence: float = 0.0
    events: tuple[AbilityEvent, ...] = ()
    sequence_progress: tuple[SequenceProgress, ...] = ()


class AbilityRecognizer:
    """Recognize conservative image-space poses and movement transitions.

    Movement is measured from MediaPipe landmark histories using monotonic
    timestamps and palm-relative screen coordinates. It does not infer a 3D
    forward direction or body pose.
    """

    _FOCUS = (False, True, True, False, False)
    _OPEN = (True, True, True, True, True)
    _FIST = (None, False, False, False, False)
    _SYSTEM_GESTURES = {
        GestureType.POINTING,
        GestureType.OPEN_PALM,
        GestureType.PINCH,
        GestureType.FIST,
    }

    def __init__(
        self,
        calibration: AbilityCalibrationStore | None = None,
        abilities: tuple[AbilityDefinition, ...] | None = None,
    ) -> None:
        self.abilities = abilities or load_ability_library()
        self._by_id = {ability.ability_id: ability for ability in self.abilities}
        self.calibration = calibration
        self._current_id: str | None = None
        self._started_at: float | None = None
        self._last_activation: dict[str, float] = {}
        self._latched = False
        self._last_snapshot = AbilityRecognitionSnapshot()

    @property
    def snapshot(self) -> AbilityRecognitionSnapshot:
        return self._last_snapshot

    def reset(self) -> None:
        self._current_id = None
        self._started_at = None
        self._last_activation.clear()
        self._latched = False
        self._last_snapshot = AbilityRecognitionSnapshot()

    def update(
        self,
        recognition: AnimeRecognitionSnapshot,
        timestamp: float,
        pose_landmarks: tuple[tuple[float, float, float, float], ...] = (),
        frame_size: tuple[int, int] | None = None,
    ) -> AbilityRecognitionSnapshot:
        visible_hands = tuple(hand for hand in recognition.hands if hand.visible)
        hands = visible_hands
        aspect_ratio = (
            frame_size[0] / frame_size[1]
            if frame_size is not None and frame_size[1] > 0
            else 1.0
        )
        sequence_definition = next(
            (
                definition
                for event in recognition.sequence_events
                for definition in self.abilities
                if definition.sequence_id == event.sequence_id
            ),
            None,
        )
        if sequence_definition is not None:
            definition = sequence_definition
            if self._current_id is not None:
                events = [self._event("released", hands=())]
                self._current_id = None
                self._started_at = None
                self._latched = False
            else:
                events = []
            last_activation = self._last_activation.get(
                definition.ability_id,
                float("-inf"),
            )
            if timestamp - last_activation >= definition.cooldown_seconds:
                position, hand_ids = self._sequence_anchor(recognition.hands)
                events.append(
                    AbilityEvent(
                        definition.ability_id,
                        definition.display_name,
                        definition.franchise,
                        "activated",
                        position,
                        (1.0, 0.0),
                        hand_ids,
                        1.0,
                    )
                )
                self._last_activation[definition.ability_id] = timestamp
            return self._publish(events, recognition)
        candidate = self._candidate(hands, timestamp, pose_landmarks, aspect_ratio)
        events: list[AbilityEvent] = []
        if candidate is None:
            if self._current_id is not None:
                events.append(self._event("released", hands=()))
            self._current_id = None
            self._started_at = None
            self._latched = False
            return self._publish(
                events,
                recognition,
            )

        definition, matched_hands, direction, confidence = candidate
        if definition.ability_id != self._current_id:
            if self._current_id is not None:
                events.append(self._event("released", hands=()))
            self._current_id = definition.ability_id
            self._started_at = timestamp
            self._latched = False
            events.append(self._event("charging", matched_hands, direction, confidence))

        held_seconds = (
            timestamp - self._started_at
            if self._started_at is not None
            else 0.0
        )
        last_activation = self._last_activation.get(definition.ability_id, float("-inf"))
        cooldown_over = timestamp - last_activation >= definition.cooldown_seconds
        activate = False
        movement_threshold = (
            self.calibration.movement_threshold(
                definition.ability_id,
                definition.movement_threshold_scales,
            )
            if self.calibration is not None
            else definition.movement_threshold_scales
        )
        if not self._latched and cooldown_over and held_seconds >= definition.minimum_hold_seconds:
            if definition.movement_gate == "cupped_synchronized_thrust":
                activate = confidence >= 1.0
            elif definition.movement_gate in {"synchronized_thrust", "single_hand_thrust"}:
                activate = hypot(*direction) >= movement_threshold
            else:
                activate = True
        if activate:
            self._latched = True
            self._last_activation[definition.ability_id] = timestamp
            events.append(self._event("activated", matched_hands, direction, confidence))

        return self._publish(
            events,
            recognition,
            position=self._center(matched_hands),
            direction=direction,
            hand_ids=tuple(hand.identity for hand in matched_hands),
            confidence=confidence,
        )

    def _candidate(
        self,
        hands: tuple[HandObservation, ...],
        timestamp: float,
        pose_landmarks: tuple[tuple[float, float, float, float], ...],
        aspect_ratio: float,
    ) -> tuple[AbilityDefinition, tuple[HandObservation, ...], tuple[float, float], float] | None:
        if not hands:
            return None
        focus = tuple(hand for hand in hands if self._states_match(hand, self._FOCUS))
        open_hands = tuple(hand for hand in hands if self._states_match(hand, self._OPEN))
        fists = tuple(hand for hand in hands if self._states_match(hand, self._FIST))

        domain = next(
            (
                definition
                for definition in self.abilities
                if definition.movement_gate == "single_calibrated_hold"
            ),
            None,
        )
        if domain is not None and len(hands) == 1 and self.calibration is not None:
            score = self.calibration.landmark_similarity(
                domain.ability_id,
                hands,
                aspect_ratio,
            )
            if score is not None and score >= self.calibration.minimum_similarity:
                return domain, hands, (0.0, 0.0), score

        for definition in self.abilities:
            if definition.movement_gate == "crossed_arms_hold":
                if (
                    len(hands) == 2
                    and self._arms_crossed(pose_landmarks, hands, aspect_ratio)
                ):
                    return definition, hands, (1.0, 0.0), 0.98
            elif definition.movement_gate == "single_calibrated_hold":
                continue
            elif definition.movement_gate == "cupped_synchronized_thrust":
                if (
                    len(hands) == 2
                    and self._cupped_hands(hands)
                    and self._hands_close(hands)
                    and self._calibrated_pose_matches(definition, hands, aspect_ratio)
                    and (
                        self._current_id == definition.ability_id
                        or self._hands_near_body_side(pose_landmarks, hands)
                    )
                ):
                    movement_threshold = (
                        self.calibration.movement_threshold(
                            definition.ability_id,
                            definition.movement_threshold_scales,
                        )
                        if self.calibration is not None
                        else definition.movement_threshold_scales
                    )
                    motion, confidence = self._shared_thrust(
                        hands,
                        timestamp,
                        definition,
                        pose_landmarks,
                        aspect_ratio,
                        movement_threshold,
                    )
                    return definition, hands, motion, confidence
            elif definition.movement_gate == "synchronized_thrust":
                if self._pose_matches(definition, hands, focus, 2) and self._hands_close(hands):
                    motion, confidence = self._shared_motion(hands, timestamp, definition)
                    return definition, hands, motion, confidence
            elif definition.movement_gate == "single_hand_thrust":
                if self._pose_matches(definition, hands, focus, 1):
                    motion, confidence = self._hand_motion(hands[0], timestamp, definition)
                    return definition, hands, motion, confidence
            elif definition.movement_gate == "rise_and_hold":
                if (
                    self._pose_matches(definition, hands, open_hands, 2)
                    and self._hands_raised(hands)
                ):
                    return definition, hands, (0.0, -1.0), 0.95
            elif definition.movement_gate == "stable_hold":
                if self._pose_matches(definition, hands, fists, 2) and self._hands_close(hands):
                    return definition, hands, (0.0, 0.0), 0.98
        return None

    def _calibrated_pose_matches(
        self,
        definition: AbilityDefinition,
        hands: tuple[HandObservation, ...],
        aspect_ratio: float,
    ) -> bool:
        if self.calibration is None or not self.calibration.has_samples(definition.ability_id):
            return False
        score = self.calibration.landmark_similarity(
            definition.ability_id,
            hands,
            aspect_ratio,
        )
        return score is not None and score >= self.calibration.minimum_similarity

    @staticmethod
    def _cupped_hands(hands: tuple[HandObservation, ...]) -> bool:
        for hand in hands:
            ratios = hand.features.finger_extension_ratios
            if len(ratios) != 5:
                return False
            curled = sum(0.60 <= ratio <= 1.35 for ratio in ratios[1:])
            if curled < 3:
                return False
        return True

    @staticmethod
    def _hands_near_body_side(
        pose: tuple[tuple[float, float, float, float], ...],
        hands: tuple[HandObservation, ...],
    ) -> bool:
        if len(pose) <= 12:
            return False
        left, right = pose[11], pose[12]
        if min(left[3], right[3]) < config.ABILITY_POSE_MIN_VISIBILITY:
            return False
        shoulder_span = abs(right[0] - left[0])
        if shoulder_span < 0.05:
            return False
        midpoint_x = (left[0] + right[0]) / 2.0
        midpoint_y = (left[1] + right[1]) / 2.0
        hands_x = sum(hand.center[0] for hand in hands) / len(hands)
        hands_y = sum(hand.center[1] for hand in hands) / len(hands)
        lateral_distance = abs(hands_x - midpoint_x)
        return (
            shoulder_span * 0.12 <= lateral_distance <= shoulder_span * 1.8
            and midpoint_y - shoulder_span * 0.25
            <= hands_y
            <= midpoint_y + shoulder_span * 1.8
        )

    @staticmethod
    def _arms_crossed(
        pose: tuple[tuple[float, float, float, float], ...],
        hands: tuple[HandObservation, ...],
        aspect_ratio: float,
    ) -> bool:
        if len(pose) <= 16:
            return False
        left_shoulder, right_shoulder = pose[11], pose[12]
        left_elbow, right_elbow = pose[13], pose[14]
        left_wrist, right_wrist = pose[15], pose[16]
        if min(
            left_shoulder[3],
            right_shoulder[3],
            left_elbow[3],
            right_elbow[3],
            left_wrist[3],
            right_wrist[3],
        ) < config.ABILITY_POSE_MIN_VISIBILITY:
            return False
        shoulder_span = abs(right_shoulder[0] - left_shoulder[0])
        if shoulder_span < 0.05:
            return False
        midline = (left_shoulder[0] + right_shoulder[0]) / 2.0
        orientation = 1.0 if right_shoulder[0] > left_shoulder[0] else -1.0
        margin = shoulder_span * 0.12

        def forearm_crosses(elbow: tuple[float, ...], wrist: tuple[float, ...]) -> bool:
            elbow_side = (elbow[0] - midline) * orientation
            wrist_side = (wrist[0] - midline) * orientation
            return elbow_side * wrist_side < -(margin * margin)

        if not (
            forearm_crosses(left_elbow, left_wrist)
            and forearm_crosses(right_elbow, right_wrist)
            and (left_elbow[0] - midline) * orientation < 0
            and (right_elbow[0] - midline) * orientation > 0
            and min(
                (left_wrist[0] - midline) * orientation,
                (right_wrist[0] - midline) * orientation,
            ) < 0
            < max(
                (left_wrist[0] - midline) * orientation,
                (right_wrist[0] - midline) * orientation,
            )
        ):
            return False

        wrist_points = (left_wrist, right_wrist)
        hand_points = tuple(hand.center for hand in hands)
        shoulder_span_pixels = shoulder_span * aspect_ratio
        max_distance = shoulder_span_pixels * 0.42
        assignments = (
            (0, 1),
            (1, 0),
        )
        associated = any(
            all(
                hypot(
                    (wrist_points[index][0] - hand_points[hand_index][0])
                    * aspect_ratio,
                    wrist_points[index][1] - hand_points[hand_index][1],
                )
                <= max_distance
                for index, hand_index in enumerate(order)
            )
            for order in assignments
        )
        wrist_gap = hypot(
            (left_wrist[0] - right_wrist[0]) * aspect_ratio,
            left_wrist[1] - right_wrist[1],
        )
        return associated and wrist_gap > shoulder_span_pixels * 0.12

    def _pose_matches(
        self,
        definition: AbilityDefinition,
        hands: tuple[HandObservation, ...],
        defaults: tuple[HandObservation, ...],
        required_count: int,
    ) -> bool:
        if len(hands) != required_count:
            return False
        if self.calibration is not None and self.calibration.has_samples(definition.ability_id):
            return self.calibration.accepts(definition.ability_id, hands)
        default_ids = {hand.identity for hand in defaults}
        return len(defaults) == required_count and all(
            hand.identity in default_ids for hand in hands
        )

    @staticmethod
    def _states_match(
        hand: HandObservation,
        expected: tuple[bool | None, ...],
    ) -> bool:
        actual = hand.features.finger_states
        return len(actual) == len(expected) and all(
            value is not None and (target is None or value == target)
            for value, target in zip(actual, expected)
        )

    @staticmethod
    def _hands_close(hands: tuple[HandObservation, ...]) -> bool:
        first, second = hands
        mean_scale = max((first.palm_scale + second.palm_scale) / 2.0, 1e-6)
        return hypot(
            first.center[0] - second.center[0],
            first.center[1] - second.center[1],
        ) / mean_scale <= 3.5

    @staticmethod
    def _hands_raised(hands: tuple[HandObservation, ...]) -> bool:
        if any(hand.center[1] > 0.42 for hand in hands):
            return False
        return all(
            len(hand.history) >= 2
            and hand.history[-1][2] <= hand.history[0][2] - 0.04
            for hand in hands
        )

    @staticmethod
    def _recent_displacement(
        hand: HandObservation,
        timestamp: float,
        window: float,
    ) -> tuple[float, float]:
        if not hand.history:
            return 0.0, 0.0
        latest = hand.history[-1]
        earlier = next(
            (sample for sample in hand.history if timestamp - sample[0] <= window),
            hand.history[-1],
        )
        return latest[1] - earlier[1], latest[2] - earlier[2]

    @classmethod
    def _hand_motion(
        cls,
        hand: HandObservation,
        timestamp: float,
        definition: AbilityDefinition,
    ) -> tuple[tuple[float, float], float]:
        dx, dy = cls._recent_displacement(
            hand,
            timestamp,
            definition.movement_window_seconds,
        )
        scale = max(hand.palm_scale, 1e-6)
        movement = (dx / scale, dy / scale)
        return movement, min(1.0, hypot(*movement) / max(definition.movement_threshold_scales, 1e-6))

    @classmethod
    def _shared_motion(
        cls,
        hands: tuple[HandObservation, ...],
        timestamp: float,
        definition: AbilityDefinition,
    ) -> tuple[tuple[float, float], float]:
        movement = [
            cls._recent_displacement(hand, timestamp, definition.movement_window_seconds)
            for hand in hands
        ]
        scales = [max(hand.palm_scale, 1e-6) for hand in hands]
        normalized = [
            (delta[0] / scale, delta[1] / scale)
            for delta, scale in zip(movement, scales)
        ]
        first, second = normalized
        first_length = hypot(*first)
        second_length = hypot(*second)
        if first_length == 0.0 or second_length == 0.0:
            return (0.0, 0.0), 0.0
        alignment = (first[0] * second[0] + first[1] * second[1]) / (
            first_length * second_length
        )
        if alignment < 0.5:
            return (0.0, 0.0), max(0.0, alignment)
        average = ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)
        return average, min(
            1.0,
            hypot(*average) / max(definition.movement_threshold_scales, 1e-6),
        )

    @classmethod
    def _shared_thrust(
        cls,
        hands: tuple[HandObservation, ...],
        timestamp: float,
        definition: AbilityDefinition,
        pose: tuple[tuple[float, float, float, float], ...],
        aspect_ratio: float,
        movement_threshold: float,
    ) -> tuple[tuple[float, float], float]:
        vectors: list[tuple[float, float, float]] = []
        for hand in hands:
            if not hand.history:
                return (0.0, 0.0), 0.0
            latest = hand.history[-1]
            earlier = next(
                (
                    sample
                    for sample in hand.history
                    if timestamp - sample[0] <= definition.movement_window_seconds
                ),
                latest,
            )
            scale = max(hand.palm_scale, 1e-6)
            vectors.append(
                (
                    (latest[1] - earlier[1]) * aspect_ratio / scale,
                    (latest[2] - earlier[2]) / scale,
                    -(latest[3] - earlier[3]) / scale,
                )
            )
        first, second = vectors
        if first[2] < 0.12 or second[2] < 0.12:
            return (0.0, 0.0), 0.0
        lengths = [sqrt(sum(value * value for value in vector)) for vector in vectors]
        if min(lengths) <= 1e-8:
            return (0.0, 0.0), 0.0
        alignment = sum(
            first[index] * second[index] for index in range(3)
        ) / (lengths[0] * lengths[1])
        if alignment < 0.45:
            return (0.0, 0.0), max(0.0, alignment)
        average = tuple((first[index] + second[index]) / 2.0 for index in range(3))
        movement_magnitude = sqrt(sum(value * value for value in average))
        threshold = max(movement_threshold, 1e-6)
        confidence = min(1.0, movement_magnitude / threshold)
        screen_direction = (average[0] / aspect_ratio, average[1])
        if hypot(*screen_direction) < 0.08:
            screen_direction = cls._forearm_direction(pose)
        return screen_direction, confidence

    @staticmethod
    def _forearm_direction(
        pose: tuple[tuple[float, float, float, float], ...],
    ) -> tuple[float, float]:
        if len(pose) <= 16:
            return 0.0, 0.0
        vectors = []
        for elbow_index, wrist_index in ((13, 15), (14, 16)):
            elbow, wrist = pose[elbow_index], pose[wrist_index]
            if min(elbow[3], wrist[3]) < config.ABILITY_POSE_MIN_VISIBILITY:
                continue
            vectors.append(
                (
                    wrist[0] - elbow[0],
                    wrist[1] - elbow[1],
                )
            )
        if not vectors:
            return 0.0, 0.0
        return (
            sum(vector[0] for vector in vectors) / len(vectors),
            sum(vector[1] for vector in vectors) / len(vectors),
        )

    def _event(
        self,
        phase: AbilityPhase,
        hands: tuple[HandObservation, ...] | None = None,
        direction: tuple[float, float] | None = None,
        confidence: float | None = None,
    ) -> AbilityEvent:
        assert self._current_id is not None
        definition = self._by_id[self._current_id]
        selected_hands = hands or ()
        selected_direction = direction or (0.0, 0.0)
        return AbilityEvent(
            definition.ability_id,
            definition.display_name,
            definition.franchise,
            phase,
            self._center(selected_hands),
            selected_direction,
            tuple(hand.identity for hand in selected_hands),
            confidence if confidence is not None else 1.0,
        )

    @staticmethod
    def _center(hands: tuple[HandObservation, ...]) -> tuple[float, float]:
        if not hands:
            return (0.5, 0.5)
        return (
            sum(hand.center[0] for hand in hands) / len(hands),
            sum(hand.center[1] for hand in hands) / len(hands),
        )

    @staticmethod
    def _sequence_anchor(
        hands: tuple[HandObservation, ...],
    ) -> tuple[tuple[float, float], tuple[str, ...]]:
        visible = tuple(hand for hand in hands if hand.visible)
        if not visible:
            return (0.5, 0.5), ()
        return AbilityRecognizer._center(visible), tuple(hand.identity for hand in visible)

    def _publish(
        self,
        events: list[AbilityEvent],
        recognition: AnimeRecognitionSnapshot,
        *,
        position: tuple[float, float] = (0.5, 0.5),
        direction: tuple[float, float] = (1.0, 0.0),
        hand_ids: tuple[str, ...] = (),
        confidence: float = 0.0,
    ) -> AbilityRecognitionSnapshot:
        if self._current_id is None:
            state = "READY"
            display_id = None
        else:
            state = "ACTIVATED" if self._latched else "CHARGING"
            display_id = self._current_id
        self._last_snapshot = AbilityRecognitionSnapshot(
            display_id,
            state,
            position,
            direction,
            hand_ids,
            confidence,
            tuple(events),
            recognition.sequence_progress,
        )
        return self._last_snapshot
