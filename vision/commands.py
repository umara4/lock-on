"""Debounced gesture commands and application-level control state."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import hypot, radians, tan

import config
from vision.gestures import GestureType, HandGesture
from vision.tracker import ByteTrackTracker, TrackedObject


class CommandType(str, Enum):
    POINT_SELECT = "point_select"
    TOGGLE_TRACKING = "toggle_tracking"
    TOGGLE_TARGET_LOCK = "toggle_target_lock"
    TOGGLE_INTERCEPTION = "toggle_interception"
    CYCLE_TARGET = "cycle_target"
    RESET = "reset"


@dataclass(frozen=True)
class CommandEvent:
    command: CommandType


@dataclass(frozen=True)
class GestureUpdate:
    raw_gesture: GestureType
    stable_gesture: GestureType | None
    stable_frames: int
    stability: float
    events: tuple[CommandEvent, ...]


class GestureCommandProcessor:
    """Smooth gesture classifications and emit one-shot toggle commands."""

    _TOGGLE_COMMANDS = {
        GestureType.OPEN_PALM: CommandType.TOGGLE_TRACKING,
        GestureType.PINCH: CommandType.TOGGLE_TARGET_LOCK,
        GestureType.FIST: CommandType.TOGGLE_INTERCEPTION,
    }

    def __init__(
        self,
        stable_frames: int = config.GESTURE_STABLE_FRAMES,
        cooldown_seconds: float = config.GESTURE_COOLDOWN_SECONDS,
        history_length: int = config.GESTURE_HISTORY_LENGTH,
    ) -> None:
        if stable_frames < 1:
            raise ValueError("stable_frames must be at least 1")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must not be negative")
        if history_length < stable_frames:
            raise ValueError("history_length must be at least stable_frames")
        self.required_stable_frames = stable_frames
        self.cooldown_seconds = cooldown_seconds
        self.history: deque[GestureType] = deque(maxlen=history_length)
        self._candidate = GestureType.NONE
        self._candidate_frames = 0
        self._stable_gesture: GestureType | None = None
        self._latched_toggle: GestureType | None = None
        self._last_command_time = float("-inf")

    def update(self, hand: HandGesture, timestamp: float) -> GestureUpdate:
        """Return a stable gesture and any newly activated command events."""
        gesture = hand.gesture if hand.detected else GestureType.NONE
        self.history.append(gesture)
        if gesture == self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate = gesture
            self._candidate_frames = 1

        stable_gesture = (
            gesture
            if self._candidate_frames >= self.required_stable_frames
            and all(
                item == gesture
                for item in list(self.history)[-self.required_stable_frames :]
            )
            else None
        )
        stability = sum(item == gesture for item in self.history) / len(self.history)
        events: list[CommandEvent] = []
        if stable_gesture is not None and stable_gesture != self._stable_gesture:
            self._stable_gesture = stable_gesture
            if stable_gesture == GestureType.NONE:
                self._latched_toggle = None
            elif stable_gesture not in self._TOGGLE_COMMANDS:
                self._latched_toggle = None

        if stable_gesture == GestureType.POINTING:
            events.append(CommandEvent(CommandType.POINT_SELECT))
        elif stable_gesture in self._TOGGLE_COMMANDS:
            if stable_gesture != self._latched_toggle:
                self._latched_toggle = stable_gesture
                if timestamp - self._last_command_time >= self.cooldown_seconds:
                    events.append(
                        CommandEvent(self._TOGGLE_COMMANDS[stable_gesture])
                    )
                    self._last_command_time = timestamp

        return GestureUpdate(
            raw_gesture=gesture,
            stable_gesture=stable_gesture,
            stable_frames=self._candidate_frames,
            stability=stability,
            events=tuple(events),
        )

    def reset(self) -> None:
        self.history.clear()
        self._candidate = GestureType.NONE
        self._candidate_frames = 0
        self._stable_gesture = None
        self._latched_toggle = None
        self._last_command_time = float("-inf")


@dataclass
class ControlState:
    tracking_paused: bool = False
    target_locked: bool = False
    locked_target_id: int | None = None
    interception_active: bool = False
    feedback: str = "SYSTEM READY"


class SystemCommandRouter:
    """Apply command events to the tracker and expose displayable system state."""

    def __init__(self) -> None:
        self.state = ControlState()
        self.pointing_ray: tuple[tuple[int, int], tuple[int, int]] | None = None
        self._active_hand = HandGesture(GestureType.NONE)

    def handle(
        self,
        event: CommandEvent,
        tracker: ByteTrackTracker,
        tracks: list[TrackedObject],
        frame_shape: tuple[int, ...],
    ) -> None:
        command = event.command
        if command == CommandType.POINT_SELECT:
            self._point_select(tracker, tracks, frame_shape)
        elif command == CommandType.TOGGLE_TRACKING:
            self.state.tracking_paused = not self.state.tracking_paused
            self.state.feedback = (
                "TRACKING PAUSED"
                if self.state.tracking_paused
                else "TRACKING RESUMED"
            )
        elif command == CommandType.TOGGLE_TARGET_LOCK:
            self._toggle_target_lock(tracker)
        elif command == CommandType.TOGGLE_INTERCEPTION:
            self.state.interception_active = not self.state.interception_active
            self.state.feedback = (
                "INTERCEPTION MODE ACTIVE"
                if self.state.interception_active
                else "INTERCEPTION MODE INACTIVE"
            )
        elif command == CommandType.CYCLE_TARGET:
            if self.state.target_locked:
                self.state.feedback = "UNLOCK TARGET BEFORE CYCLING"
                return
            selected = tracker.cycle_selection(
                [track.track_id for track in tracks]
            )
            self.state.feedback = (
                f"TARGET ID {selected} SELECTED"
                if selected is not None
                else "NO TRACKED TARGETS"
            )
        elif command == CommandType.RESET:
            tracker.reset()
            self.state.tracking_paused = False
            self.state.target_locked = False
            self.state.locked_target_id = None
            self.state.interception_active = False
            self.pointing_ray = None
            self.state.feedback = "TRACKING RESET"

    def handle_key(
        self,
        key: int,
        tracker: ByteTrackTracker,
        tracks: list[TrackedObject],
        frame_shape: tuple[int, ...],
    ) -> bool:
        """Apply keyboard fallbacks; return true when the key was handled."""
        commands = {
            ord(" "): CommandType.TOGGLE_TRACKING,
            ord("t"): CommandType.TOGGLE_TRACKING,
            ord("l"): CommandType.TOGGLE_TARGET_LOCK,
            ord("i"): CommandType.TOGGLE_INTERCEPTION,
            ord("n"): CommandType.CYCLE_TARGET,
            ord("c"): CommandType.CYCLE_TARGET,
            ord("r"): CommandType.RESET,
        }
        command = commands.get(key)
        if command is None:
            return False
        self.handle(CommandEvent(command), tracker, tracks, frame_shape)
        return True

    def validate_lock(self, tracker: ByteTrackTracker) -> None:
        """Release lock when ByteTrack expires the selected target."""
        if self.state.target_locked and tracker.selected_id is None:
            self.state.target_locked = False
            self.state.locked_target_id = None
            self.state.feedback = "LOCK RELEASED // TARGET LOST"

    def _point_select(
        self,
        tracker: ByteTrackTracker,
        tracks: list[TrackedObject],
        frame_shape: tuple[int, ...],
    ) -> None:
        selected_id, ray = select_pointed_track(
            tracks,
            self._active_hand,
            frame_shape,
        )
        self.pointing_ray = ray
        if self.state.target_locked:
            return
        if selected_id != tracker.selected_id:
            tracker.select(selected_id)
            self.state.feedback = (
                f"TARGET ID {selected_id} SELECTED"
                if selected_id is not None
                else "NO TARGET ALIGNED"
            )

    def set_hand(self, hand: HandGesture) -> None:
        self._active_hand = hand
        if hand.gesture != GestureType.POINTING:
            self.pointing_ray = None

    def _toggle_target_lock(self, tracker: ByteTrackTracker) -> None:
        if self.state.target_locked:
            self.state.target_locked = False
            self.state.locked_target_id = None
            self.state.feedback = "TARGET UNLOCKED"
        elif tracker.selected_id is None:
            self.state.feedback = "PINCH IGNORED // NO SELECTED TARGET"
        else:
            self.state.target_locked = True
            self.state.locked_target_id = tracker.selected_id
            self.state.feedback = f"TARGET ID {tracker.selected_id} LOCKED"


def select_pointed_track(
    tracks: list[TrackedObject],
    hand: HandGesture,
    frame_shape: tuple[int, ...],
    angle_tolerance_degrees: float = config.POINTING_ANGLE_TOLERANCE_DEGREES,
) -> tuple[int | None, tuple[tuple[int, int], tuple[int, int]] | None]:
    """Select the nearest track center aligned with the index-finger pointing ray."""
    if not hand.detected or hand.gesture != GestureType.POINTING:
        return None, None

    height, width = frame_shape[:2]
    landmarks = hand.landmarks
    tip = landmarks[8]
    pip = landmarks[6]
    origin = (int(tip[0] * width), int(tip[1] * height))
    dx = (tip[0] - pip[0]) * width
    dy = (tip[1] - pip[1]) * height
    magnitude = hypot(dx, dy)
    if magnitude <= 1e-6:
        return None, None
    direction = (dx / magnitude, dy / magnitude)
    ray_length = int(hypot(width, height) * config.POINTING_RAY_LENGTH_RATIO)
    endpoint = (
        int(origin[0] + direction[0] * ray_length),
        int(origin[1] + direction[1] * ray_length),
    )
    ray = (origin, endpoint)

    min_tolerance = hypot(width, height) * config.POINTING_MIN_TOLERANCE_RATIO
    tangent = tan(radians(angle_tolerance_degrees))
    candidates: list[tuple[float, int]] = []
    for track in tracks:
        offset_x = track.center[0] - origin[0]
        offset_y = track.center[1] - origin[1]
        along_ray = offset_x * direction[0] + offset_y * direction[1]
        if along_ray <= 0:
            continue
        perpendicular = abs(offset_x * direction[1] - offset_y * direction[0])
        if perpendicular <= max(min_tolerance, along_ray * tangent):
            candidates.append((along_ray, track.track_id))

    if not candidates:
        return None, ray
    _, selected_id = min(candidates)
    return selected_id, ray
