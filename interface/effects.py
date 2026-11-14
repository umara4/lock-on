"""Bounded, timestamp-driven procedural visual-effect state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite
from pathlib import Path

import config
from vision.anime_gestures import AnimeRecognitionSnapshot


class EffectType(str, Enum):
    ENERGY_CHARGE = "energy_charge"
    LIGHTNING = "lightning"
    FIREBALL = "fireball"
    ENERGY_BURST = "energy_burst"
    AURA = "aura"


@dataclass(frozen=True)
class EffectSnapshot:
    effect_id: int
    effect_type: EffectType
    position: tuple[float, float]
    age_seconds: float
    duration_seconds: float
    intensity: float
    follow_hand_id: str | None
    velocity: tuple[float, float]
    seed: int


@dataclass
class _Effect:
    effect_id: int
    effect_type: EffectType
    position: tuple[float, float]
    started_at: float
    duration_seconds: float
    intensity: float
    follow_hand_id: str | None
    velocity: tuple[float, float]
    seed: int
    released_at: float | None = None
    last_motion_at: float | None = None


class EffectEngine:
    """Map recognized pose events to lightweight bounded effect instances."""

    def __init__(
        self,
        mapping_path: str | Path | None = None,
        maximum_effects: int = config.ANIME_MAX_ACTIVE_EFFECTS,
        enabled: bool = config.ANIME_EFFECTS_ENABLED,
    ) -> None:
        if maximum_effects < 1:
            raise ValueError("maximum_effects must be positive")
        selected_path = (
            Path(mapping_path)
            if mapping_path is not None
            else Path(__file__).with_name("effect_library.json")
        )
        mapping = json.loads(selected_path.read_text(encoding="utf-8"))
        self._pose_actions: dict[str, dict[str, tuple[EffectType, ...]]] = {
            pose_id: {
                kind: tuple(EffectType(action) for action in actions)
                for kind, actions in action_map.items()
            }
            for pose_id, action_map in mapping.get("pose_actions", {}).items()
        }
        self._sequence_actions = {
            sequence_id: EffectType(effect_type)
            for sequence_id, effect_type in mapping.get("sequence_actions", {}).items()
        }
        self.maximum_effects = maximum_effects
        self.enabled = enabled
        self._effects: list[_Effect] = []
        self._continuous: dict[tuple[str, EffectType], int] = {}
        self._next_id = 1
        self._last_update = 0.0
        self._snapshot: tuple[EffectSnapshot, ...] = ()

    @property
    def snapshot(self) -> tuple[EffectSnapshot, ...]:
        return self._snapshot

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        if not enabled:
            self._effects.clear()
            self._continuous.clear()
            self._snapshot = ()

    def reset(self) -> None:
        self._effects.clear()
        self._continuous.clear()
        self._snapshot = ()

    def consume(
        self,
        recognition: AnimeRecognitionSnapshot,
        timestamp: float,
    ) -> None:
        if not self.enabled:
            return
        positions = {
            hand.identity: hand.center
            for hand in recognition.hands
            if hand.visible
        }
        for event in recognition.events:
            actions = self._pose_actions.get(event.gesture_id, {}).get(event.kind, ())
            for effect_type in actions:
                if event.kind == "recognized" and effect_type in (
                    EffectType.ENERGY_CHARGE,
                    EffectType.AURA,
                ):
                    if not event.hand_ids:
                        continue
                    hand_id = event.hand_ids[0]
                    self._start_continuous(
                        hand_id,
                        effect_type,
                        timestamp,
                        positions.get(hand_id, event.position),
                        event.confidence,
                    )
                elif event.kind == "released":
                    if effect_type == EffectType.LIGHTNING:
                        self.trigger(effect_type, timestamp, event.position)
                    for hand_id in event.hand_ids:
                        self._release_continuous(hand_id, timestamp)
        for event in recognition.sequence_events:
            effect_type = self._sequence_actions.get(event.sequence_id)
            if effect_type is not None:
                visible_hands = tuple(hand for hand in recognition.hands if hand.visible)
                position = (
                    (
                        sum(hand.center[0] for hand in visible_hands) / len(visible_hands),
                        sum(hand.center[1] for hand in visible_hands) / len(visible_hands),
                    )
                    if visible_hands
                    else (0.5, 0.5)
                )
                self.trigger(effect_type, timestamp, position)

    def trigger(
        self,
        effect_type: EffectType,
        timestamp: float,
        position: tuple[float, float],
        *,
        follow_hand_id: str | None = None,
        velocity: tuple[float, float] | None = None,
        intensity: float = 1.0,
    ) -> int | None:
        if not self.enabled:
            return None
        if not isfinite(timestamp) or not all(isfinite(value) for value in position):
            raise ValueError("effect timestamp and position must be finite")
        if len(self._effects) >= self.maximum_effects:
            oldest = self._effects.pop(0)
            self._continuous = {
                key: value for key, value in self._continuous.items()
                if value != oldest.effect_id
            }
        durations = {
            EffectType.ENERGY_CHARGE: 1.2,
            EffectType.LIGHTNING: 0.42,
            EffectType.FIREBALL: 2.4,
            EffectType.ENERGY_BURST: 0.72,
            EffectType.AURA: 1.2,
        }
        effect_id = self._next_id
        self._next_id += 1
        self._effects.append(
            _Effect(
                effect_id,
                effect_type,
                position,
                timestamp,
                durations[effect_type],
                max(0.0, min(1.0, intensity)),
                follow_hand_id,
                velocity or ((0.62, -0.08) if effect_type == EffectType.FIREBALL else (0.0, 0.0)),
                effect_id * 7919,
                last_motion_at=timestamp,
            )
        )
        return effect_id

    def update(
        self,
        timestamp: float,
        recognition: AnimeRecognitionSnapshot,
    ) -> tuple[EffectSnapshot, ...]:
        if timestamp < self._last_update:
            raise ValueError("effect clock must not move backwards")
        self._last_update = timestamp
        if not self.enabled:
            return ()
        positions = {
            hand.identity: hand.center
            for hand in recognition.hands
            if hand.visible
        }
        for effect in self._effects:
            if effect.effect_type == EffectType.FIREBALL and effect.released_at is None:
                last_motion_at = (
                    effect.last_motion_at
                    if effect.last_motion_at is not None
                    else timestamp
                )
                elapsed = max(0.0, timestamp - last_motion_at)
                effect.position = (
                    effect.position[0] + effect.velocity[0] * elapsed,
                    effect.position[1] + effect.velocity[1] * elapsed,
                )
                effect.last_motion_at = timestamp
            elif effect.follow_hand_id is not None:
                if effect.follow_hand_id in positions:
                    effect.position = positions[effect.follow_hand_id]
                elif effect.released_at is None:
                    effect.released_at = timestamp
            if effect.effect_type in (EffectType.ENERGY_CHARGE, EffectType.AURA):
                if effect.released_at is None:
                    effect.duration_seconds = max(
                        effect.duration_seconds,
                        timestamp - effect.started_at + 0.5,
                    )
                effect.intensity = (
                    1.0
                    if effect.released_at is None
                    else max(0.0, 1.0 - (timestamp - effect.released_at) / 0.45)
                )

        self._effects = [
            effect
            for effect in self._effects
            if timestamp - effect.started_at < effect.duration_seconds
            and effect.intensity > 0.01
            and -0.2 <= effect.position[0] <= 1.2
            and -0.2 <= effect.position[1] <= 1.2
        ]
        self._snapshot = tuple(
            EffectSnapshot(
                effect.effect_id,
                effect.effect_type,
                effect.position,
                max(0.0, timestamp - effect.started_at),
                effect.duration_seconds,
                effect.intensity,
                effect.follow_hand_id,
                effect.velocity,
                effect.seed,
            )
            for effect in self._effects
        )
        return self._snapshot

    def _start_continuous(
        self,
        hand_id: str,
        effect_type: EffectType,
        timestamp: float,
        position: tuple[float, float],
        intensity: float,
    ) -> None:
        key = (hand_id, effect_type)
        if key in self._continuous:
            effect = next(
                (effect for effect in self._effects if effect.effect_id == self._continuous[key]),
                None,
            )
            if effect is not None:
                effect.position = position
                effect.intensity = intensity
            return
        effect_id = self.trigger(
            effect_type,
            timestamp,
            position,
            follow_hand_id=hand_id,
            intensity=intensity,
        )
        if effect_id is not None:
            self._continuous[key] = effect_id

    def _release_continuous(self, hand_id: str, timestamp: float) -> None:
        for key, effect_id in tuple(self._continuous.items()):
            if key[0] != hand_id:
                continue
            effect = next(
                (effect for effect in self._effects if effect.effect_id == effect_id),
                None,
            )
            if effect is not None:
                effect.released_at = timestamp
                effect.follow_hand_id = None
            del self._continuous[key]
