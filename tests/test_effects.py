"""Tests for bounded time-based effects and Qt-only vector rendering."""

from __future__ import annotations

import unittest
from threading import Event

import numpy as np
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QImage, QPainter

from interface.effect_renderer import EffectRenderer
from interface.effects import EffectEngine, EffectType
from interface.ability_effect_renderer import AbilityEffectRenderer
from interface.ability_effects import AbilityEffectEngine
from tests.test_anime_gestures import detected, focus_pose
from vision.anime_gestures import AnimeGestureRecognizer, AnimeRecognitionSnapshot
from vision.ability_recognition import (
    AbilityEvent,
    AbilityRecognitionSnapshot,
)


class EffectEngineTests(unittest.TestCase):
    def test_charge_and_aura_follow_the_hand_and_release_cleanly(self) -> None:
        recognizer = AnimeGestureRecognizer(cooldown_seconds=0.0)
        engine = EffectEngine()
        hand = detected(focus_pose(center=(0.4, 0.5)), "Left")

        engine.consume(recognizer.update((hand,), 1.0, (640, 640)), 1.0)
        recognized = recognizer.update((hand,), 1.4, (640, 640))
        engine.consume(recognized, 1.4)
        effects = engine.update(1.4, recognized)
        self.assertEqual(
            {effect.effect_type for effect in effects},
            {EffectType.ENERGY_CHARGE, EffectType.AURA},
        )

        moved = detected(focus_pose(center=(0.55, 0.5)), "Left")
        moving_snapshot = recognizer.update((moved,), 1.5, (640, 640))
        followed = engine.update(1.5, moving_snapshot)
        self.assertTrue(all(effect.position[0] > 0.5 for effect in followed))

        released = recognizer.update((), 1.6, (640, 640))
        engine.consume(released, 1.6)
        fading = engine.update(1.6, released)
        self.assertIn(EffectType.LIGHTNING, {effect.effect_type for effect in fading})
        self.assertTrue(
            all(effect.follow_hand_id is None for effect in fading)
        )
        self.assertTrue(all(effect.intensity >= 0.0 for effect in fading))
        self.assertEqual(engine.update(3.0, released), ())

    def test_projectile_moves_independently_and_effect_count_is_bounded(self) -> None:
        engine = EffectEngine(maximum_effects=3)
        projectile_id = engine.trigger(EffectType.FIREBALL, 0.0, (0.25, 0.5))
        self.assertIsNotNone(projectile_id)
        first = engine.update(0.1, AnimeRecognitionSnapshot())
        second = engine.update(0.6, AnimeRecognitionSnapshot())
        self.assertGreater(second[0].position[0], first[0].position[0])

        for index in range(10):
            engine.trigger(EffectType.LIGHTNING, 0.7 + index * 0.01, (0.5, 0.5))
        bounded = engine.update(0.9, AnimeRecognitionSnapshot())
        self.assertLessEqual(len(bounded), 3)

    def test_effects_can_be_disabled_and_vector_rendering_changes_canvas(self) -> None:
        engine = EffectEngine()
        engine.trigger(EffectType.ENERGY_BURST, 1.0, (0.5, 0.5))
        effects = engine.update(1.1, AnimeRecognitionSnapshot())
        image = QImage(320, 240, QImage.Format.Format_ARGB32)
        image.fill(QColor("#000000"))
        painter = QPainter(image)
        EffectRenderer.draw(painter, QRectF(0, 0, 320, 240), effects, 1.1)
        painter.end()
        self.assertNotEqual(image.pixelColor(160, 120), QColor("#000000"))

        engine.set_enabled(False)
        self.assertEqual(engine.snapshot, ())
        self.assertIsNone(engine.trigger(EffectType.LIGHTNING, 2.0, (0.5, 0.5)))

    def test_kamehameha_charge_follows_hand_then_beam_renders_and_expires(self) -> None:
        engine = AbilityEffectEngine()
        hands = AnimeGestureRecognizer()
        first = hands.update(
            (detected(focus_pose(center=(0.4, 0.5)), "Left"),),
            1.0,
            (640, 480),
        )
        charging = AbilityEvent(
            "kamehameha",
            "KAMEHAMEHA",
            "Dragon Ball Z inspired",
            "charging",
            (0.4, 0.5),
            (1.0, 0.0),
            (first.hands[0].identity,),
            0.9,
        )
        initial = engine.update(
            1.0,
            AbilityRecognitionSnapshot(events=(charging,)),
            first.hands,
        )
        self.assertEqual(initial[0].phase, "charging")

        activated = AbilityEvent(
            "kamehameha",
            "KAMEHAMEHA",
            "Dragon Ball Z inspired",
            "activated",
            (0.4, 0.5),
            (1.0, 0.0),
            (first.hands[0].identity,),
            0.9,
        )
        beam = engine.update(
            1.1,
            AbilityRecognitionSnapshot(events=(activated,)),
            first.hands,
        )
        self.assertEqual(beam[0].effect_type, "beam")
        self.assertEqual(beam[0].phase, "activated")
        self.assertEqual(beam[0].duration_seconds, 1.8)
        self.assertEqual(beam[0].origin, (0.4, 0.5))
        released = AbilityEvent(
            "kamehameha",
            "KAMEHAMEHA",
            "Dragon Ball Z inspired",
            "released",
            (0.5, 0.5),
            (0.0, 0.0),
            (),
            1.0,
        )
        continued = engine.update(
            1.2,
            AbilityRecognitionSnapshot(events=(released,)),
        )
        self.assertEqual(continued[0].phase, "activated")
        self.assertEqual(continued[0].duration_seconds, 1.8)

        image = QImage(320, 240, QImage.Format.Format_ARGB32)
        image.fill(QColor("#000000"))
        painter = QPainter(image)
        AbilityEffectRenderer.draw(painter, QRectF(0, 0, 320, 240), beam, 1.1)
        painter.end()
        self.assertNotEqual(image.pixelColor(128, 120), QColor("#000000"))
        self.assertEqual(engine.update(3.0, AbilityRecognitionSnapshot()), ())

    def test_shadow_clone_segments_bounded_person_sprites_and_expires(self) -> None:
        class FakeSegmenter:
            def process(self, image: np.ndarray) -> object:
                mask = np.zeros(image.shape[:2], dtype=np.float32)
                mask[20:100, 25:75] = 1.0
                return type("Result", (), {"segmentation_mask": mask})()

            def close(self) -> None:
                pass

        engine = AbilityEffectEngine()
        engine._segmenter = FakeSegmenter()
        event = AbilityEvent(
            "shadow_clone",
            "SHADOW CLONE JUTSU",
            "Naruto inspired",
            "activated",
            (0.5, 0.5),
            (1.0, 0.0),
            ("hand-1", "hand-2"),
            0.98,
        )
        frame = np.full((120, 100, 3), 110, dtype=np.uint8)
        engine.update(
            1.0,
            AbilityRecognitionSnapshot(events=(event,)),
            frame=frame,
        )
        self.assertIsNotNone(engine._segmentation_future)
        engine._segmentation_future.result(timeout=2.0)
        effects = engine.update(1.1, AbilityRecognitionSnapshot())
        self.assertEqual(len(effects), 1)
        self.assertEqual(effects[0].duration_seconds, 7.0)
        self.assertEqual(len(effects[0].clone_sprites), 3)
        sprite = effects[0].clone_sprites[0]
        rgba = np.frombuffer(sprite.rgba, dtype=np.uint8).reshape(
            sprite.height,
            sprite.width,
            4,
        )
        self.assertTrue(np.any(rgba[..., 3] == 0))
        self.assertTrue(np.any(rgba[..., 3] > 200))

        cache = {
            (item.effect_id, item.clone_index): QImage(
                item.rgba,
                item.width,
                item.height,
                item.width * 4,
                QImage.Format.Format_RGBA8888,
            ).copy()
            for item in effects[0].clone_sprites
        }
        image = QImage(320, 240, QImage.Format.Format_ARGB32)
        image.fill(QColor("#000000"))
        painter = QPainter(image)
        AbilityEffectRenderer.draw_clones(
            painter,
            QRectF(0, 0, 320, 240),
            effects,
            1.1,
            cache,
        )
        painter.end()
        self.assertNotEqual(image.pixelColor(160, 120), QColor("#000000"))
        self.assertEqual(engine.update(8.1, AbilityRecognitionSnapshot()), ())
        engine.close()

    def test_shadow_clone_keeps_only_latest_pending_frame(self) -> None:
        started = Event()
        release = Event()

        class SlowSegmenter:
            def process(self, image: np.ndarray) -> object:
                started.set()
                if not release.wait(2.0):
                    raise TimeoutError("test segmentation was not released")
                mask = np.ones(image.shape[:2], dtype=np.float32)
                return type("Result", (), {"segmentation_mask": mask})()

            def close(self) -> None:
                pass

        engine = AbilityEffectEngine()
        engine._segmenter = SlowSegmenter()

        def activation() -> AbilityEvent:
            return AbilityEvent(
                "shadow_clone",
                "SHADOW CLONE JUTSU",
                "Naruto inspired",
                "activated",
                (0.5, 0.5),
                (1.0, 0.0),
                ("hand-1", "hand-2"),
                0.98,
            )

        frame = np.full((120, 100, 3), 110, dtype=np.uint8)
        try:
            engine.update(
                1.0,
                AbilityRecognitionSnapshot(events=(activation(),)),
                frame=frame,
            )
            self.assertTrue(started.wait(2.0))
            engine.update(
                1.2,
                AbilityRecognitionSnapshot(events=(activation(),)),
                frame=frame,
            )
            self.assertIsNotNone(engine._pending_clone)

            release.set()
            first_future = engine._segmentation_future
            self.assertIsNotNone(first_future)
            first_future.result(timeout=2.0)
            engine.update(1.3, AbilityRecognitionSnapshot())
            second_future = engine._segmentation_future
            self.assertIsNotNone(second_future)
            second_future.result(timeout=2.0)
            effects = engine.update(1.4, AbilityRecognitionSnapshot())

            self.assertEqual(len(effects), 1)
            self.assertTrue(effects[0].clone_sprites)
            self.assertTrue(
                all(
                    sprite.effect_id == effects[0].effect_id
                    for sprite in effects[0].clone_sprites
                )
            )
        finally:
            release.set()
            engine.close()

    def test_domain_effect_changes_the_environment_not_only_its_center(self) -> None:
        engine = AbilityEffectEngine()
        activation = AbilityEvent(
            "domain_expansion",
            "DOMAIN EXPANSION",
            "Jujutsu Kaisen inspired",
            "activated",
            (0.5, 0.45),
            (0.0, 0.0),
            ("hand-1",),
            1.0,
        )
        effects = engine.update(
            2.0,
            AbilityRecognitionSnapshot(events=(activation,)),
        )
        image = QImage(400, 300, QImage.Format.Format_ARGB32)
        image.fill(QColor("#808080"))
        painter = QPainter(image)
        AbilityEffectRenderer.draw(painter, QRectF(0, 0, 400, 300), effects, 2.1)
        painter.end()
        self.assertNotEqual(image.pixelColor(5, 5), QColor("#808080"))
        self.assertNotEqual(image.pixelColor(200, 150), QColor("#808080"))


if __name__ == "__main__":
    unittest.main()
