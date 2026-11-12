"""Dashboard control events, bounded event history, and live telemetry values."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import cv2
import numpy as np

import config


class DashboardAction(str, Enum):
    CAMERA = "camera"
    VIDEO = "video"
    SOURCE_TOGGLE = "source_toggle"
    DETECTION = "detection"
    GESTURES = "gestures"
    TRACKING = "tracking"
    LOCK = "lock"
    NEXT_TARGET = "next_target"
    INTERCEPTION = "interception"
    RESET_TRACKING = "reset_tracking"
    RESET_SIMULATION = "reset_simulation"


class EventSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"


@dataclass(frozen=True)
class DashboardEvent:
    timestamp: str
    message: str
    severity: EventSeverity


@dataclass(frozen=True)
class HandTelemetry:
    handedness: str | None
    gesture: str
    stable_gesture: str
    stability: float
    command: str


class EventLog:
    """Keep a bounded chronological event list so long runs cannot leak memory."""

    def __init__(self, max_events: int = config.DASHBOARD_EVENT_LIMIT) -> None:
        if max_events < 1:
            raise ValueError("max_events must be at least one")
        self._events: deque[DashboardEvent] = deque(maxlen=max_events)

    @property
    def events(self) -> tuple[DashboardEvent, ...]:
        return tuple(self._events)

    def add(
        self,
        message: str,
        severity: EventSeverity = EventSeverity.INFO,
        timestamp: datetime | None = None,
    ) -> DashboardEvent:
        event = DashboardEvent(
            (timestamp or datetime.now()).strftime("%H:%M:%S"),
            message,
            severity,
        )
        self._events.append(event)
        return event


@dataclass(frozen=True)
class DashboardTelemetry:
    detected_count: int
    tracked_count: int
    selected_id: int | None
    selected_label: str | None
    selected_confidence: float | None
    target_locked: bool
    tracking_paused: bool
    detection_enabled: bool
    gestures_enabled: bool
    hand_detected: bool
    gesture: str
    stable_gesture: str
    gesture_stability: float
    gesture_command: str
    hands: tuple[HandTelemetry, ...]
    interception_active: bool
    fps: float
    uptime_seconds: float
    source_label: str
    source_running: bool
    system_state: str


class DashboardControls:
    """Map rendered dashboard hit boxes to queued actions for the main loop."""

    BUTTONS: tuple[tuple[str, DashboardAction], ...] = (
        ("CAMERA", DashboardAction.CAMERA),
        ("VIDEO FILE", DashboardAction.VIDEO),
        ("DETECT", DashboardAction.DETECTION),
        ("GESTURES", DashboardAction.GESTURES),
        ("PAUSE", DashboardAction.TRACKING),
        ("LOCK", DashboardAction.LOCK),
        ("NEXT TARGET", DashboardAction.NEXT_TARGET),
        ("INTERCEPT", DashboardAction.INTERCEPTION),
        ("RESET TRACK", DashboardAction.RESET_TRACKING),
        ("RESET SIM", DashboardAction.RESET_SIMULATION),
    )

    def __init__(self) -> None:
        from queue import SimpleQueue

        self._actions: SimpleQueue[DashboardAction] = SimpleQueue()
        self._bounds: list[tuple[int, int, int, int, DashboardAction]] = []
        self._display_size = (0, 0)

    def enqueue(self, action: DashboardAction) -> None:
        self._actions.put(action)

    def next_action(self) -> DashboardAction | None:
        if self._actions.empty():
            return None
        return self._actions.get()

    def mouse_callback(self, event: int, x: int, y: int, flags: int, data: object) -> None:
        del flags, data
        if event != cv2.EVENT_LBUTTONUP:
            return
        display_width, display_height = self._display_size
        if display_width <= 0 or display_height <= 0:
            return
        try:
            _, _, window_width, window_height = cv2.getWindowImageRect(
                config.WINDOW_NAME
            )
        except cv2.error:
            window_width, window_height = display_width, display_height
        if window_width > 0 and window_height > 0:
            x = round(x * display_width / window_width)
            y = round(y * display_height / window_height)
        for left, top, right, bottom, action in self._bounds:
            if left <= x < right and top <= y < bottom:
                self.enqueue(action)
                return

    def draw(self, frame: np.ndarray, telemetry: DashboardTelemetry) -> np.ndarray:
        """Append a two-row, state-aware click toolbar below the dashboard."""
        bar_height = config.DASHBOARD_TOOLBAR_HEIGHT
        height, width = frame.shape[:2]
        canvas = np.zeros((height + bar_height, width, 3), dtype=frame.dtype)
        canvas[:height] = frame
        canvas[height:] = (10, 18, 22)
        cv2.line(canvas, (0, height), (width, height), (65, 155, 175), 1)

        labels = {
            DashboardAction.CAMERA: (
                "STOP CAMERA"
                if telemetry.source_label.startswith("CAM") and telemetry.source_running
                else "START CAMERA"
            ),
            DashboardAction.VIDEO: "OPEN VIDEO",
            DashboardAction.DETECTION: "DETECT ON" if telemetry.detection_enabled else "DETECT OFF",
            DashboardAction.GESTURES: "GESTURES ON" if telemetry.gestures_enabled else "GESTURES OFF",
            DashboardAction.TRACKING: "RESUME" if telemetry.tracking_paused else "PAUSE",
            DashboardAction.LOCK: "UNLOCK" if telemetry.target_locked else "LOCK",
            DashboardAction.NEXT_TARGET: "NEXT TARGET",
            DashboardAction.INTERCEPTION: "STOP INTERCEPT" if telemetry.interception_active else "INTERCEPT",
            DashboardAction.RESET_TRACKING: "RESET TRACK",
            DashboardAction.RESET_SIMULATION: "RESET SIM",
        }
        self._bounds.clear()
        self._display_size = (width, height + bar_height)
        columns = 5
        gap = config.DASHBOARD_BUTTON_GAP
        button_height = (bar_height - 3 * gap) // 2
        button_width = (width - (columns + 1) * gap) // columns
        for index, (_, action) in enumerate(self.BUTTONS):
            row, column = divmod(index, columns)
            left = gap + column * (button_width + gap)
            top = height + gap + row * (button_height + gap)
            right = min(width - gap, left + button_width)
            bottom = min(height + bar_height - gap, top + button_height)
            selected = (
                action == DashboardAction.DETECTION and telemetry.detection_enabled
                or action == DashboardAction.GESTURES and telemetry.gestures_enabled
                or action == DashboardAction.INTERCEPTION and telemetry.interception_active
                or action == DashboardAction.LOCK and telemetry.target_locked
            )
            color = (0, 135, 165) if selected else (29, 49, 56)
            cv2.rectangle(canvas, (left, top), (right, bottom), color, cv2.FILLED)
            cv2.rectangle(canvas, (left, top), (right, bottom), (55, 165, 190), 1)
            label = labels[action]
            scale = 0.39 if len(label) > 11 else 0.44
            (text_width, text_height), _ = cv2.getTextSize(
                label, config.FONT, scale, 1
            )
            cv2.putText(
                canvas,
                label,
                (left + max(4, (right - left - text_width) // 2), top + (bottom - top + text_height) // 2),
                config.FONT,
                scale,
                (235, 245, 245),
                1,
                cv2.LINE_AA,
            )
            self._bounds.append((left, top, right, bottom, action))
        return canvas

    def hit_test(self, x: int, y: int) -> DashboardAction | None:
        """Return the action under a dashboard coordinate (also used by tests)."""
        for left, top, right, bottom, action in self._bounds:
            if left <= x < right and top <= y < bottom:
                return action
        return None
