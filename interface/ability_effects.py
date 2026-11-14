"""Bounded lifecycle for camera-space ability effects."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from math import hypot, isfinite
from typing import Protocol
import cv2
import numpy as np

import config
from vision.ability_library import AbilityDefinition, load_ability_library
from vision.ability_recognition import AbilityRecognitionSnapshot
from vision.anime_gestures import HandObservation


class _SegmentationResult(Protocol):
    segmentation_mask: np.ndarray | None


class _SelfieSegmenter(Protocol):
    def process(self, image: np.ndarray) -> _SegmentationResult: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class CloneSpriteSnapshot:
    effect_id: int
    clone_index: int
    rgba: bytes
    width: int
    height: int
    bounds: tuple[float, float, float, float]
    opacity: float
    spacing_ratio: float


@dataclass(frozen=True)
class AbilityEffectSnapshot:
    effect_id: int
    ability_id: str
    effect_type: str
    phase: str
    position: tuple[float, float]
    direction: tuple[float, float]
    age_seconds: float
    duration_seconds: float
    intensity: float
    seed: int
    hand_trail: tuple[tuple[float, float], ...]
    clone_sprites: tuple[CloneSpriteSnapshot, ...] = ()
    origin: tuple[float, float] | None = None


@dataclass
class _AbilityEffect:
    effect_id: int
    ability_id: str
    effect_type: str
    phase: str
    position: tuple[float, float]
    direction: tuple[float, float]
    started_at: float
    duration_seconds: float
    intensity: float
    seed: int
    hand_ids: tuple[str, ...]
    trail: list[tuple[float, float]]
    clone_sprites: tuple[CloneSpriteSnapshot, ...] = ()
    origin: tuple[float, float] | None = None


class AbilityEffectEngine:
    """Translate recognition transitions into time-based, bounded animations."""

    def __init__(
        self,
        maximum_effects: int = config.ANIME_MAX_ACTIVE_EFFECTS,
        abilities: tuple[AbilityDefinition, ...] | None = None,
    ) -> None:
        if maximum_effects < 1:
            raise ValueError("maximum ability effects must be positive")
        self.maximum_effects = maximum_effects
        self.abilities = {
            item.ability_id: item
            for item in (abilities or load_ability_library())
        }
        self._effects: list[_AbilityEffect] = []
        self._next_id = 1
        self._last_update = float("-inf")
        self._snapshot: tuple[AbilityEffectSnapshot, ...] = ()
        self.enabled = True
        self._segmenter: _SelfieSegmenter | None = None
        self._segmentation_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="jarvis-person-mask",
        )
        self._warmup_future: Future[None] | None = None
        self._segmentation_future: Future[tuple[CloneSpriteSnapshot, ...]] | None = None
        self._segmentation_effect_id: int | None = None
        self._pending_clone: tuple[int, np.ndarray] | None = None

    @property
    def snapshot(self) -> tuple[AbilityEffectSnapshot, ...]:
        return self._snapshot

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        if not enabled:
            self.reset()

    def reset(self) -> None:
        self._effects.clear()
        self._snapshot = ()
        self._pending_clone = None

    def close(self) -> None:
        self._segmentation_executor.shutdown(wait=True)
        try:
            if self._warmup_future is not None:
                self._warmup_future.result()
            if self._segmentation_future is not None:
                self._segmentation_future.result()
        finally:
            self._warmup_future = None
            self._segmentation_future = None
            self._segmentation_effect_id = None
            self._pending_clone = None
            segmenter = self._segmenter
            self._segmenter = None
            try:
                if segmenter is not None:
                    segmenter.close()
            finally:
                self.reset()

    def prepare_segmentation(self) -> None:
        if self._segmenter is None and self._warmup_future is None:
            self._warmup_future = self._segmentation_executor.submit(
                self._initialize_segmenter
            )

    def update(
        self,
        timestamp: float,
        recognition: AbilityRecognitionSnapshot,
        visible_hands: tuple[HandObservation, ...] = (),
        frame: np.ndarray | None = None,
    ) -> tuple[AbilityEffectSnapshot, ...]:
        if not isfinite(timestamp) or timestamp < self._last_update:
            raise ValueError("ability effect clock must be finite and monotonic")
        self._last_update = timestamp
        self._poll_segmentation()
        if not self.enabled:
            return ()

        hands = {
            hand.identity: hand
            for hand in visible_hands
            if hand.visible
        }
        for event in recognition.events:
            definition = self.abilities[event.ability_id]
            effect = next(
                (item for item in self._effects if item.ability_id == event.ability_id),
                None,
            )
            if event.phase == "charging":
                if effect is None:
                    self._add(
                        definition,
                        "charging",
                        event.position,
                        event.direction,
                        timestamp,
                        20.0,
                        event.confidence,
                        event.hand_ids,
                        frame,
                    )
            elif event.phase == "activated":
                if effect is None:
                    effect = self._add(
                        definition,
                        "activated",
                        event.position,
                        event.direction,
                        timestamp,
                        self._duration(definition),
                        event.confidence,
                        event.hand_ids,
                        frame,
                    )
                else:
                    effect.phase = "activated"
                    effect.position = event.position
                    effect.direction = event.direction
                    effect.started_at = timestamp
                    effect.duration_seconds = self._duration(definition)
                    effect.intensity = max(effect.intensity, event.confidence)
                    effect.hand_ids = event.hand_ids
                    effect.origin = event.position
                    if effect.effect_type == "shadow_clones" and frame is not None:
                        self._schedule_clones(effect.effect_id, frame)
            else:
                if effect is not None:
                    if effect.phase == "activated":
                        continue
                    effect.phase = "released"
                    effect.started_at = timestamp
                    effect.duration_seconds = 0.35
                    effect.hand_ids = ()

        visible_by_id = {key: value for key, value in hands.items()}
        for effect in self._effects:
            if effect.phase == "charging" or (
                effect.phase == "activated" and effect.effect_type == "afterimage"
            ):
                followed = [
                    visible_by_id[identity]
                    for identity in effect.hand_ids
                    if identity in visible_by_id
                ]
                if followed:
                    effect.position = (
                        sum(hand.center[0] for hand in followed) / len(followed),
                        sum(hand.center[1] for hand in followed) / len(followed),
                    )
                    effect.trail.append(effect.position)
                    del effect.trail[:-16]
        self._effects = [
            effect
            for effect in self._effects
            if timestamp - effect.started_at < effect.duration_seconds
        ]
        self._snapshot = tuple(
            AbilityEffectSnapshot(
                effect.effect_id,
                effect.ability_id,
                effect.effect_type,
                effect.phase,
                effect.position,
                effect.direction,
                max(0.0, timestamp - effect.started_at),
                effect.duration_seconds,
                effect.intensity,
                effect.seed,
                tuple(effect.trail),
                effect.clone_sprites,
                effect.origin,
            )
            for effect in self._effects
        )
        return self._snapshot

    def _add(
        self,
        definition: AbilityDefinition,
        phase: str,
        position: tuple[float, float],
        direction: tuple[float, float],
        timestamp: float,
        duration: float,
        intensity: float,
        hand_ids: tuple[str, ...],
        frame: np.ndarray | None = None,
    ) -> _AbilityEffect:
        if len(self._effects) >= self.maximum_effects:
            self._effects.pop(0)
        magnitude = hypot(*direction)
        normalized_direction = (
            (direction[0] / magnitude, direction[1] / magnitude)
            if magnitude > 1e-8
            else (1.0, 0.0)
        )
        effect = _AbilityEffect(
            self._next_id,
            definition.ability_id,
            definition.effect,
            phase,
            position,
            normalized_direction,
            timestamp,
            duration,
            max(0.0, min(1.0, intensity)),
            self._next_id * 37 % 997,
            hand_ids,
            [position],
            origin=position,
        )
        if definition.effect == "shadow_clones" and phase == "activated" and frame is not None:
            self._schedule_clones(effect.effect_id, frame)
        self._next_id += 1
        self._effects.append(effect)
        return effect

    def _initialize_segmenter(self) -> None:
        if self._segmenter is not None:
            return
        import mediapipe as mp

        self._segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=0,
        )
        warmup = np.zeros((64, 64, 3), dtype=np.uint8)
        self._segmenter.process(warmup)

    def _schedule_clones(self, effect_id: int, frame: np.ndarray) -> None:
        if self._segmentation_future is not None:
            self._pending_clone = (effect_id, frame.copy())
            return
        self._segmentation_effect_id = effect_id
        self._segmentation_future = self._segmentation_executor.submit(
            self._make_clones,
            effect_id,
            frame.copy(),
        )

    def _poll_segmentation(self) -> None:
        if self._warmup_future is not None and self._warmup_future.done():
            future = self._warmup_future
            self._warmup_future = None
            future.result()
        future = self._segmentation_future
        if future is None or not future.done():
            return
        effect_id = self._segmentation_effect_id
        self._segmentation_future = None
        self._segmentation_effect_id = None
        try:
            sprites = future.result()
        except Exception:
            self._pending_clone = None
            raise
        effect = next(
            (item for item in self._effects if item.effect_id == effect_id),
            None,
        )
        if effect is not None and self.enabled:
            effect.clone_sprites = sprites
        pending = self._pending_clone
        self._pending_clone = None
        if pending is not None and any(
            item.effect_id == pending[0] and item.effect_type == "shadow_clones"
            for item in self._effects
        ):
            self._schedule_clones(*pending)

    def _make_clones(
        self,
        effect_id: int,
        frame: np.ndarray,
    ) -> tuple[CloneSpriteSnapshot, ...]:
        if self._segmenter is None:
            self._initialize_segmenter()
        height, width = frame.shape[:2]
        scale = min(
            1.0,
            config.ABILITY_SHADOW_CLONE_MAX_FRAME_DIMENSION / max(height, width),
        )
        sampled = (
            cv2.resize(
                frame,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            if scale < 1.0
            else frame
        )
        result = self._segmenter.process(cv2.cvtColor(sampled, cv2.COLOR_BGR2RGB))
        mask = result.segmentation_mask
        if mask is None:
            raise RuntimeError("MediaPipe Selfie Segmentation returned no person mask")
        mask = np.asarray(mask, dtype=np.float32)
        mask = cv2.GaussianBlur(
            mask,
            (
                config.ABILITY_SHADOW_CLONE_MASK_BLUR_PIXELS | 1,
                config.ABILITY_SHADOW_CLONE_MASK_BLUR_PIXELS | 1,
            ),
            0.0,
        )
        foreground = mask >= config.ABILITY_SHADOW_CLONE_MASK_THRESHOLD
        rows, columns = np.nonzero(foreground)
        if not len(columns):
            return ()
        sampled_height, sampled_width = sampled.shape[:2]
        padding_x = max(2, int((columns.max() - columns.min() + 1) * 0.08))
        padding_y = max(2, int((rows.max() - rows.min() + 1) * 0.04))
        x1 = max(0, int(columns.min()) - padding_x)
        x2 = min(sampled_width, int(columns.max()) + 1 + padding_x)
        y1 = max(0, int(rows.min()) - padding_y)
        y2 = min(sampled_height, int(rows.max()) + 1 + padding_y)
        crop = sampled[y1:y2, x1:x2]
        alpha = (mask[y1:y2, x1:x2] * 255.0).clip(0, 255).astype(np.uint8)
        rgba_base = cv2.cvtColor(crop, cv2.COLOR_BGR2RGBA)
        count = config.ABILITY_SHADOW_CLONE_COUNT
        sprites: list[CloneSpriteSnapshot] = []
        spacing = config.ABILITY_SHADOW_CLONE_SPACING_RATIO
        tints = (
            (1.0, 0.88, 1.08),
            (0.82, 0.94, 1.18),
            (1.12, 0.88, 0.94),
        )
        for index in range(count):
            rgba = rgba_base.copy()
            blue, green, red = tints[index % len(tints)]
            rgba[..., 0] = (rgba[..., 0] * red).clip(0, 255).astype(np.uint8)
            rgba[..., 1] = (rgba[..., 1] * green).clip(0, 255).astype(np.uint8)
            rgba[..., 2] = (rgba[..., 2] * blue).clip(0, 255).astype(np.uint8)
            rgba[..., 3] = alpha
            offset = (index - (count - 1) / 2.0) * spacing
            sprites.append(
                CloneSpriteSnapshot(
                    effect_id,
                    index,
                    rgba.tobytes(),
                    x2 - x1,
                    y2 - y1,
                    (
                        x1 / sampled_width,
                        y1 / sampled_height,
                        (x2 - x1) / sampled_width,
                        (y2 - y1) / sampled_height,
                    ),
                    config.ABILITY_SHADOW_CLONE_OPACITY,
                    offset,
                )
            )
        return tuple(sprites)

    @staticmethod
    def _duration(definition: AbilityDefinition) -> float:
        if definition.effect == "shadow_clones":
            return config.ABILITY_SHADOW_CLONE_DURATION_SECONDS
        return definition.effect_duration_seconds
