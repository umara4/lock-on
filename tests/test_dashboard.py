"""Tests for dashboard control mapping, bounded logs, and live telemetry rendering."""

from __future__ import annotations

from datetime import datetime
import unittest

import cv2
import numpy as np

from interface.dashboard import (
    DashboardAction,
    DashboardControls,
    DashboardTelemetry,
    EventLog,
    EventSeverity,
    HandTelemetry,
)


def telemetry(**overrides: object) -> DashboardTelemetry:
    values: dict[str, object] = {
        "detected_count": 2,
        "tracked_count": 2,
        "selected_id": 3,
        "selected_label": "person",
        "selected_confidence": 0.91,
        "target_locked": False,
        "tracking_paused": False,
        "detection_enabled": True,
        "gestures_enabled": True,
        "hand_detected": True,
        "gesture": "pointing",
        "stable_gesture": "pointing",
        "gesture_stability": 0.9,
        "gesture_command": "point_select",
        "hands": (
            HandTelemetry("Right", "pointing", "pointing", 0.9, "point_select"),
        ),
        "interception_active": False,
        "fps": 30.0,
        "uptime_seconds": 25.0,
        "source_label": "CAM 0",
        "source_running": True,
        "system_state": "TARGET SELECTED",
    }
    values.update(overrides)
    return DashboardTelemetry(**values)


class EventLogTests(unittest.TestCase):
    def test_log_is_bounded_and_timestamped(self) -> None:
        log = EventLog(max_events=2)
        log.add("FIRST", timestamp=datetime(2026, 10, 31, 16, 0, 0))
        log.add("SECOND", EventSeverity.WARNING)
        log.add("THIRD", EventSeverity.SUCCESS)
        self.assertEqual([event.message for event in log.events], ["SECOND", "THIRD"])
        self.assertEqual(log.events[-1].severity, EventSeverity.SUCCESS)
        self.assertRegex(log.events[-1].timestamp, r"^\d{2}:\d{2}:\d{2}$")


class DashboardControlsTests(unittest.TestCase):
    def test_controls_render_and_hit_test_to_functional_actions(self) -> None:
        controls = DashboardControls()
        frame = np.zeros((540, 1250, 3), dtype=np.uint8)
        rendered = controls.draw(frame, telemetry())
        self.assertEqual(
            rendered.shape,
            (540 + 86, 1250, 3),
        )
        action = DashboardAction.INTERCEPTION
        bounds = next(
            rect for rect in controls._bounds if rect[4] == action
        )
        x = (bounds[0] + bounds[2]) // 2
        y = (bounds[1] + bounds[3]) // 2
        self.assertEqual(controls.hit_test(x, y), action)
        controls.mouse_callback(cv2.EVENT_LBUTTONUP, x, y, 0, None)
        self.assertEqual(controls.next_action(), action)
        self.assertIsNone(controls.next_action())

    def test_toolbar_reflects_camera_and_system_toggles(self) -> None:
        controls = DashboardControls()
        frame = np.zeros((360, 900, 3), dtype=np.uint8)
        controls.draw(
            frame,
            telemetry(
                source_label="video.mp4",
                source_running=False,
                detection_enabled=False,
                gestures_enabled=False,
                target_locked=True,
                interception_active=True,
            ),
        )
        self.assertEqual(len(controls._bounds), len(controls.BUTTONS))
        self.assertIsNotNone(controls.hit_test(10, 10 + frame.shape[0]))


if __name__ == "__main__":
    unittest.main()
