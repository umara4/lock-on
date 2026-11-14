"""Offscreen smoke tests for the PySide6 workspace and camera presentation."""

from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace
import time
import unittest
from unittest import mock

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from application.pipeline import PipelineCommand
from interface.desktop import JarvisWindow, SimulationCanvas, VideoCanvas
from vision.anime_gestures import AnimeRecognitionSnapshot
from robotics.interceptor import InterceptionSimulator
from vision.gestures import DetectedHand, GestureType, HandGesture


class DesktopWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_window_is_visible_before_background_workers_start(self) -> None:
        window = JarvisWindow(0, "yolov8n.pt", 0.35, "cpu", "bytetrack.yaml", 640)
        window.show()
        self.app.processEvents()
        self.assertTrue(window.isVisible())
        self.assertFalse(window._capture._thread.is_alive())
        self.assertFalse(window._vision._thread.is_alive())
        window.close()
        self.app.processEvents()

    def test_camera_canvas_renders_image_and_controls_enqueue_commands(self) -> None:
        window = JarvisWindow(0, "yolov8n.pt", 0.35, "cpu", "bytetrack.yaml", 640)
        window.show()
        image = QImage(320, 240, QImage.Format.Format_RGB32)
        image.fill(Qt.GlobalColor.darkGray)
        window.canvas.set_image(image, (320, 240))
        self.app.processEvents()
        self.assertFalse(window.canvas.grab().isNull())

        window.buttons["pause"].click()
        queued = window._vision._commands.get_nowait()
        self.assertEqual(queued.command, PipelineCommand.TOGGLE_PAUSE)
        window.close()
        self.app.processEvents()

    def test_toolbar_controls_dispatch_pipeline_commands(self) -> None:
        window = JarvisWindow(0, "yolov8n.pt", 0.35, "cpu", "bytetrack.yaml", 640)
        controls = {
            "pause": PipelineCommand.TOGGLE_PAUSE,
            "detection": PipelineCommand.TOGGLE_DETECTION,
            "gestures": PipelineCommand.TOGGLE_GESTURES,
            "lock": PipelineCommand.TOGGLE_LOCK,
            "interception": PipelineCommand.TOGGLE_INTERCEPTION,
            "cycle": PipelineCommand.CYCLE_TARGET,
            "reset_tracking": PipelineCommand.RESET_TRACKING,
            "reset_simulation": PipelineCommand.RESET_SIMULATION,
            "classification": PipelineCommand.TOGGLE_CLASSIFICATION,
            "anime": PipelineCommand.TOGGLE_ANIME_GESTURES,
            "effects": PipelineCommand.TOGGLE_ANIME_EFFECTS,
            "abilities": PipelineCommand.TOGGLE_ABILITIES,
        }
        for key, command in controls.items():
            with self.subTest(control=key):
                window.buttons[key].click()
                queued = window._vision._commands.get_nowait()
                self.assertEqual(queued.command, command)
        window.close()
        self.app.processEvents()

    def test_ability_selector_and_calibration_submit_worker_commands(self) -> None:
        window = JarvisWindow(0, "yolov8n.pt", 0.35, "cpu", "bytetrack.yaml", 640)
        window.ability_selector.setCurrentIndex(2)
        selected = window._vision._commands.get_nowait()
        self.assertEqual(selected.command, PipelineCommand.SELECT_ABILITY)
        self.assertEqual(selected.ability_id, "chidori")

        window.record_ability_sample.click()
        calibration = window._vision._commands.get_nowait()
        self.assertEqual(
            calibration.command,
            PipelineCommand.RECORD_ABILITY_SAMPLE,
        )
        self.assertEqual(calibration.ability_id, "chidori")
        window.record_ability_movement.click()
        movement = window._vision._commands.get_nowait()
        self.assertEqual(
            movement.command,
            PipelineCommand.TOGGLE_ABILITY_MOVEMENT_RECORDING,
        )
        self.assertEqual(movement.ability_id, "chidori")

        window.ability_similarity.setValue(0.82)
        threshold = window._vision._commands.get_nowait()
        self.assertEqual(threshold.command, PipelineCommand.SET_ABILITY_THRESHOLD)
        self.assertEqual(threshold.ability_threshold, 0.82)
        window.close()
        self.app.processEvents()

    def test_command_palette_toggles_and_displays_events(self) -> None:
        window = JarvisWindow(0, "yolov8n.pt", 0.35, "cpu", "bytetrack.yaml", 640)
        window.show()
        self.app.processEvents()
        self.assertFalse(window.command_palette.isVisible())

        window.menu_button.click()
        self.app.processEvents()
        self.assertTrue(window.command_palette.isVisible())
        self.assertIn("−", window.menu_button.text())

        window.log_toggle.click()
        self.app.processEvents()
        self.assertTrue(window.event_log.isVisible())

        window.menu_button.click()
        self.app.processEvents()
        self.assertFalse(window.command_palette.isVisible())
        window.close()
        self.app.processEvents()

    def test_video_file_control_switches_source_and_resets_tracking(self) -> None:
        window = JarvisWindow(0, "yolov8n.pt", 0.35, "cpu", "bytetrack.yaml", 640)
        video_path = r"C:\videos\demo.mp4"
        with (
            mock.patch(
                "interface.desktop.QFileDialog.getOpenFileName",
                return_value=(video_path, "Video files"),
            ),
            mock.patch.object(window._capture, "request_source") as request_source,
            mock.patch.object(window, "_submit") as submit,
        ):
            window.buttons["video"].click()

        request_source.assert_called_once_with(video_path)
        submit.assert_called_once_with(PipelineCommand.RESET_TRACKING)
        window.close()
        self.app.processEvents()

    def test_window_layout_resizes_with_the_desktop(self) -> None:
        window = JarvisWindow(0, "yolov8n.pt", 0.35, "cpu", "bytetrack.yaml", 640)
        window.show()
        window.resize(1200, 800)
        self.app.processEvents()
        first_size = window.canvas.size()
        window.resize(1320, 900)
        self.app.processEvents()
        self.assertNotEqual(window.canvas.size(), first_size)
        window.close()
        self.app.processEvents()

    def test_video_canvas_renders_without_an_input_frame(self) -> None:
        canvas = VideoCanvas()
        canvas.resize(640, 360)
        canvas.show()
        self.app.processEvents()
        self.assertFalse(canvas.grab().isNull())
        canvas.close()

    def test_video_canvas_draws_both_hands_and_clears_them_when_missing(self) -> None:
        canvas = VideoCanvas()
        canvas.resize(640, 360)
        image = QImage(640, 360, QImage.Format.Format_RGB32)
        image.fill(QColor("#000000"))
        canvas.set_image(image, (640, 360))
        pose = tuple((0.3 + index * 0.01, 0.2 + index * 0.02, 0.0) for index in range(21))
        left = DetectedHand(pose, "Left", GestureType.OPEN_PALM, 0.9)
        right_pose = tuple((x + 0.3, y, z) for x, y, z in pose)
        right = DetectedHand(right_pose, "Right", GestureType.PINCH, 0.8)
        now = time.perf_counter()
        snapshot = SimpleNamespace(
            tracks=(),
            position_histories={},
            selected_id=None,
            selected_estimate=None,
            recognition={},
            pointing_ray=None,
            anime=AnimeRecognitionSnapshot(),
            effects=(),
            anime_effects_enabled=True,
            hand=HandGesture(
                GestureType.OPEN_PALM,
                pose,
                "Left",
                0.9,
                (left, right),
            ),
            gestures_enabled=True,
            system_state="TRACKING",
            inference_fps=8.0,
            timestamp=now,
        )
        canvas.set_snapshot(snapshot, 0.0, True, True)
        self.assertEqual(len(canvas._hands), 2)
        rendered = canvas.grab().toImage()
        self.assertFalse(rendered.isNull())
        self.assertGreater(rendered.pixelColor(195, 76).green(), 80)

        missing = SimpleNamespace(**vars(snapshot))
        missing.hand = HandGesture()
        missing.timestamp = now + 0.1
        canvas.set_snapshot(missing, 0.0, True, True)
        self.assertEqual(canvas._hands, ())
        canvas.close()

    def test_hand_overlay_is_hidden_when_gesture_recognition_is_disabled(self) -> None:
        canvas = VideoCanvas()
        pose = tuple((0.5, 0.5, 0.0) for _ in range(21))
        hand = DetectedHand(pose, "Right", GestureType.FIST, 0.9)
        snapshot = SimpleNamespace(
            tracks=(),
            position_histories={},
            selected_id=None,
            selected_estimate=None,
            recognition={},
            pointing_ray=None,
            anime=AnimeRecognitionSnapshot(),
            effects=(),
            anime_effects_enabled=True,
            hand=HandGesture(GestureType.FIST, pose, "Right", 0.9, (hand,)),
            gestures_enabled=False,
            system_state="READY",
            inference_fps=0.0,
            timestamp=time.perf_counter(),
        )
        canvas.set_snapshot(snapshot, 0.0, True, True)
        self.assertEqual(canvas._hands, ())
        canvas.close()

    def test_simulation_canvas_renders_snapshot(self) -> None:
        simulator = InterceptionSimulator()
        snapshot = simulator.step(None, 1.0, False, 640, 480)
        canvas = SimulationCanvas()
        canvas.resize(640, 150)
        canvas.set_snapshot(snapshot)
        canvas.show()
        self.app.processEvents()
        self.assertFalse(canvas.grab().isNull())
        canvas.close()


if __name__ == "__main__":
    unittest.main()
