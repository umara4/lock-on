"""Run JARVIS video input, YOLO detection, and persistent object tracking."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from tkinter import TclError, Tk, filedialog

import cv2
import numpy as np

import config
from interface.dashboard import (
    DashboardAction,
    DashboardControls,
    DashboardTelemetry,
    EventLog,
    EventSeverity,
    HandTelemetry,
)
from interface.hud import draw_simulation_panel, draw_status, draw_tracking
from prediction.motion import MotionEstimate, MotionEstimator, PredictionStatus
from robotics.interceptor import InterceptionSimulator, InterceptionStatus
from utils.video import VideoSourceError, open_video_source
from vision.commands import (
    GestureCommandProcessor,
    GestureUpdate,
    SystemCommandRouter,
)
from vision.detector import YOLODetector
from vision.gestures import (
    HandGesture,
    HandGestureRecognizer,
    draw_hand_landmarks,
)
from vision.tracker import ByteTrackTracker, TrackedObject


def parse_source(value: str) -> int | str:
    """Treat a numeric source as a camera index and all other values as paths."""
    if value.isdecimal():
        return int(value)
    return value


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run JARVIS object tracking on a webcam or prerecorded video."
    )
    parser.add_argument(
        "--source",
        default=str(config.DEFAULT_CAMERA_INDEX),
        help="Camera index (default: 0) or path to a video file.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Detector weights (defaults to the model for --detector).",
    )
    parser.add_argument(
        "--detector",
        choices=("coco", "world"),
        default=config.DETECTOR_TYPE,
        help="Closed COCO detector or optional YOLO-World open vocabulary detector.",
    )
    parser.add_argument(
        "--classes",
        help="Comma-separated text vocabulary for --detector world.",
    )
    parser.add_argument(
        "--no-classification",
        action="store_true",
        help="Disable asynchronous crop-classification verification.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=config.CONFIDENCE_THRESHOLD,
        help="Minimum confidence for displaying a tracked detection, from 0.0 to 1.0.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example 'cpu' or '0' for the first CUDA GPU.",
    )
    parser.add_argument(
        "--tracker",
        default=config.BYTE_TRACKER_CONFIG,
        help="Ultralytics tracker configuration YAML (default: bytetrack.yaml).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=config.TRACKING_INFERENCE_SIZE,
        help="YOLO inference image size (default: 640).",
    )
    parser.add_argument(
        "--legacy-opencv",
        action="store_true",
        help="Use the original OpenCV HUD instead of the PySide6 desktop interface.",
    )
    return parser


def choose_video_file() -> str | None:
    """Open the platform video picker without introducing a GUI dependency."""
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askopenfilename(
            title="Select JARVIS video input",
            filetypes=(
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*"),
            ),
        )
    finally:
        root.destroy()
    return selected or None


def _source_label(source: int | str) -> str:
    if isinstance(source, int):
        return f"CAM {source}"
    return Path(source).name[:16]


def _system_state(
    input_available: bool,
    detection_enabled: bool,
    tracked_count: int,
    tracking_paused: bool,
    selected_id: int | None,
    target_locked: bool,
    estimate: MotionEstimate | None,
    interception_active: bool,
    interception_status: InterceptionStatus,
) -> str:
    if not input_available:
        return "ERROR"
    if tracking_paused:
        return "PAUSED"
    if interception_status == InterceptionStatus.INTERCEPTED:
        return "INTERCEPTED"
    if interception_active:
        return "INTERCEPTION ACTIVE"
    if target_locked:
        return "TARGET LOCKED"
    if selected_id is not None:
        return "TARGET SELECTED"
    if estimate is not None and estimate.status in (
        PredictionStatus.PREDICTING,
        PredictionStatus.TRACKING,
    ):
        return "PREDICTING"
    if detection_enabled:
        return "TRACKING" if tracked_count > 0 else "DETECTING"
    return "READY"


def run(
    source: int | str,
    model_path: str,
    confidence: float,
    device: str | None,
    tracker_config: str = config.BYTE_TRACKER_CONFIG,
    inference_size: int = config.TRACKING_INFERENCE_SIZE,
    detector_type: str = config.DETECTOR_TYPE,
    vocabulary: tuple[str, ...] = config.OPEN_VOCABULARY_CLASSES,
) -> int:
    if not 0.0 <= confidence <= 1.0:
        print("Error: --confidence must be between 0.0 and 1.0.", file=sys.stderr)
        return 2
    if inference_size < 32:
        print("Error: --imgsz must be at least 32 pixels.", file=sys.stderr)
        return 2

    initial_source_error: str | None = None
    try:
        capture: cv2.VideoCapture | None = open_video_source(source)
    except VideoSourceError as error:
        capture = None
        initial_source_error = str(error)
        print(f"Warning: {error}", file=sys.stderr)

    gesture_recognizer: HandGestureRecognizer | None = None
    try:
        try:
            detector = YOLODetector(
                model_path,
                confidence,
                device,
                detector_type=detector_type,
                vocabulary=vocabulary,
            )
            tracker = ByteTrackTracker(
                detector.model,
                tracker_config=tracker_config,
                inference_confidence=config.TRACKING_INFERENCE_CONFIDENCE,
                history_length=config.POSITION_HISTORY_LENGTH,
                max_missed_frames=config.TRACKING_MAX_MISSED_FRAMES,
                device=device,
                inference_size=inference_size,
            )
            gesture_recognizer = HandGestureRecognizer()
            hand_recognizer = gesture_recognizer
        except RuntimeError as error:
            print(f"Error initializing vision modules: {error}", file=sys.stderr)
            return 1

        gesture_commands = GestureCommandProcessor()
        command_router = SystemCommandRouter()
        motion_estimator = MotionEstimator()
        interceptor = InterceptionSimulator()
        dashboard_controls = DashboardControls()
        event_log = EventLog()
        if initial_source_error:
            event_log.add(initial_source_error, EventSeverity.ERROR)
        detection_enabled = True
        gestures_enabled = True
        application_started = time.perf_counter()
        last_selected_id = tracker.selected_id
        last_stable_gesture: str | None = None
        last_interception_status = None
        previous_paused = command_router.state.tracking_paused
        previous_locked = command_router.state.target_locked
        previous_interception = command_router.state.interception_active
        event_log.add("SYSTEM READY // PHASE 6", EventSeverity.SUCCESS)
        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(config.WINDOW_NAME, dashboard_controls.mouse_callback)
        last_frame: np.ndarray | None = None
        last_tracks: list[TrackedObject] = []
        last_hand = HandGesture()
        gesture_update = GestureUpdate(
            raw_gesture=last_hand.gesture,
            stable_gesture=None,
            stable_frames=0,
            stability=0.0,
            events=(),
        )
        motion_estimates: dict[int, MotionEstimate] = {}
        fps = 0.0
        last_frame_time = time.perf_counter()

        def reset_vision_state(reset_interceptor: bool = True) -> None:
            nonlocal last_tracks, motion_estimates
            tracker.reset()
            motion_estimator.reset()
            gesture_commands.reset()
            last_tracks = []
            motion_estimates = {}
            command_router.state.target_locked = False
            command_router.state.locked_target_id = None
            command_router.state.interception_active = False
            if reset_interceptor:
                interceptor.reset()

        def switch_source(selected_source: int | str) -> bool:
            nonlocal capture, source
            try:
                new_capture = open_video_source(selected_source)
            except VideoSourceError as error:
                command_router.state.feedback = str(error)[:34]
                event_log.add(str(error), EventSeverity.ERROR)
                return False
            if capture is not None:
                capture.release()
            capture = new_capture
            source = selected_source
            reset_vision_state()
            command_router.state.tracking_paused = False
            command_router.state.feedback = "INPUT SOURCE CHANGED"
            event_log.add(f"INPUT SOURCE: {_source_label(source)}", EventSeverity.SUCCESS)
            return True

        def toggle_detection() -> None:
            nonlocal detection_enabled
            detection_enabled = not detection_enabled
            if not detection_enabled:
                reset_vision_state()
            command_router.state.feedback = (
                "OBJECT DETECTION ENABLED"
                if detection_enabled
                else "OBJECT DETECTION DISABLED"
            )
            event_log.add(
                command_router.state.feedback,
                EventSeverity.SUCCESS if detection_enabled else EventSeverity.WARNING,
            )

        def apply_dashboard_action(action: DashboardAction) -> None:
            nonlocal capture, gestures_enabled
            if action == DashboardAction.CAMERA:
                if isinstance(source, int) and capture is not None:
                    capture.release()
                    capture = None
                    reset_vision_state()
                    command_router.state.feedback = "CAMERA STOPPED"
                    event_log.add("CAMERA STOPPED", EventSeverity.WARNING)
                else:
                    switch_source(config.DEFAULT_CAMERA_INDEX)
            elif action == DashboardAction.VIDEO:
                try:
                    selected_path = choose_video_file()
                except (OSError, RuntimeError, TclError) as error:
                    command_router.state.feedback = "VIDEO PICKER UNAVAILABLE"
                    event_log.add(f"VIDEO PICKER ERROR: {error}", EventSeverity.ERROR)
                    return
                if selected_path:
                    switch_source(selected_path)
            elif action == DashboardAction.DETECTION:
                toggle_detection()
            elif action == DashboardAction.GESTURES:
                gestures_enabled = not gestures_enabled
                gesture_commands.reset()
                command_router.state.feedback = (
                    "GESTURE RECOGNITION ENABLED"
                    if gestures_enabled
                    else "GESTURE RECOGNITION DISABLED"
                )
                event_log.add(
                    command_router.state.feedback,
                    EventSeverity.SUCCESS if gestures_enabled else EventSeverity.WARNING,
                )
            elif action == DashboardAction.TRACKING:
                was_paused = command_router.state.tracking_paused
                command_router.handle_key(ord(" "), tracker, last_tracks, frame.shape)
                event_log.add(
                    command_router.state.feedback,
                    EventSeverity.WARNING if not was_paused else EventSeverity.SUCCESS,
                )
            elif action == DashboardAction.LOCK:
                command_router.handle_key(ord("l"), tracker, last_tracks, frame.shape)
                event_log.add(command_router.state.feedback, EventSeverity.SUCCESS)
            elif action == DashboardAction.NEXT_TARGET:
                command_router.handle_key(ord("n"), tracker, last_tracks, frame.shape)
                event_log.add(command_router.state.feedback, EventSeverity.INFO)
            elif action == DashboardAction.INTERCEPTION:
                was_active = command_router.state.interception_active
                command_router.handle_key(ord("i"), tracker, last_tracks, frame.shape)
                validate_interception_activation(was_active)
                event_log.add(
                    command_router.state.feedback,
                    EventSeverity.SUCCESS
                    if command_router.state.interception_active
                    else EventSeverity.WARNING,
                )
            elif action == DashboardAction.RESET_TRACKING:
                reset_vision_state()
                command_router.state.tracking_paused = False
                command_router.state.feedback = "TRACKING RESET"
                event_log.add("TRACKING RESET", EventSeverity.WARNING)
            elif action == DashboardAction.RESET_SIMULATION:
                interceptor.reset()
                command_router.state.feedback = "SIMULATION RESET"
                event_log.add("INTERCEPTION SIMULATION RESET", EventSeverity.INFO)

        def validate_interception_activation(was_active: bool) -> None:
            if was_active or not command_router.state.interception_active:
                return
            target_id = (
                command_router.state.locked_target_id
                if command_router.state.target_locked
                else tracker.selected_id
            )
            target_visible = any(track.track_id == target_id for track in last_tracks)
            if target_id is None or not target_visible or target_id not in motion_estimates:
                command_router.state.interception_active = False
                command_router.state.feedback = "NO VALID TARGET FOR INTERCEPTION"

        def log_control_state_changes() -> None:
            nonlocal previous_paused, previous_locked, previous_interception
            if command_router.state.tracking_paused != previous_paused:
                event_log.add(
                    "TRACKING PAUSED"
                    if command_router.state.tracking_paused
                    else "TRACKING RESUMED",
                    EventSeverity.WARNING
                    if command_router.state.tracking_paused
                    else EventSeverity.SUCCESS,
                )
                previous_paused = command_router.state.tracking_paused
            if command_router.state.target_locked != previous_locked:
                event_log.add(
                    "TARGET LOCKED"
                    if command_router.state.target_locked
                    else "TARGET UNLOCKED",
                    EventSeverity.SUCCESS,
                )
                previous_locked = command_router.state.target_locked
            if command_router.state.interception_active != previous_interception:
                event_log.add(
                    "INTERCEPTION ACTIVATED"
                    if command_router.state.interception_active
                    else "INTERCEPTION DEACTIVATED",
                    EventSeverity.SUCCESS
                    if command_router.state.interception_active
                    else EventSeverity.WARNING,
                )
                previous_interception = command_router.state.interception_active

        while True:
            frame_available = False
            if capture is not None:
                ok, incoming_frame = capture.read()
                if ok:
                    frame = incoming_frame
                    frame_available = True
                else:
                    capture.release()
                    capture = None
                    last_tracks = []
                    command_router.state.feedback = "INPUT ENDED // SELECT SOURCE"
                    event_log.add(command_router.state.feedback, EventSeverity.WARNING)
                    frame = (
                        last_frame.copy()
                        if last_frame is not None
                        else np.zeros((480, 640, 3), dtype=np.uint8)
                    )
            else:
                frame = (
                    last_frame.copy()
                    if last_frame is not None
                    else np.zeros((480, 640, 3), dtype=np.uint8)
                )

            now = time.perf_counter()
            elapsed = now - last_frame_time
            if elapsed > 0:
                instantaneous_fps = 1.0 / elapsed
                fps = instantaneous_fps if fps == 0 else 0.9 * fps + 0.1 * instantaneous_fps
            last_frame_time = now

            if frame_available and detection_enabled and not command_router.state.tracking_paused:
                prior_ids = {track.track_id for track in last_tracks}
                last_tracks = tracker.update(frame, now)
                current_ids = {track.track_id for track in last_tracks}
                for track_id in sorted(current_ids - prior_ids):
                    event_log.add(f"OBJECT DETECTED: ID {track_id}", EventSeverity.SUCCESS)
                command_router.validate_lock(tracker)
                log_control_state_changes()
                motion_estimates = motion_estimator.update(last_tracks, now)
            elif frame_available and not command_router.state.tracking_paused and not detection_enabled:
                last_tracks = []
                motion_estimates = {}

            if frame_available and gestures_enabled:
                last_hand = hand_recognizer.process(frame)
            elif not gestures_enabled or not frame_available:
                last_hand = HandGesture()
            command_router.set_hand(last_hand)
            gesture_update = gesture_commands.update(last_hand, now)
            for event in gesture_update.events:
                was_active = command_router.state.interception_active
                command_router.handle(event, tracker, last_tracks, frame.shape)
                validate_interception_activation(was_active)
                event_log.add(
                    f"GESTURE COMMAND: {event.command.value.upper()}",
                    EventSeverity.INFO,
                )
            command_router.validate_lock(tracker)
            if tracker.selected_id != last_selected_id:
                if tracker.selected_id is not None:
                    event_log.add(
                        f"TARGET SELECTED: ID {tracker.selected_id}",
                        EventSeverity.SUCCESS,
                    )
                last_selected_id = tracker.selected_id
            stable_name = (
                gesture_update.stable_gesture.value
                if gesture_update.stable_gesture is not None
                else None
            )
            if stable_name and stable_name != last_stable_gesture:
                event_log.add(f"GESTURE RECOGNIZED: {stable_name.upper()}")
            last_stable_gesture = stable_name
            target_id = (
                command_router.state.locked_target_id
                if command_router.state.target_locked
                else tracker.selected_id
            )
            target_is_visible = any(
                track.track_id == target_id
                for track in last_tracks
            )
            target_estimate = (
                motion_estimates.get(target_id)
                if frame_available and target_id is not None and target_is_visible
                else None
            )
            simulation = interceptor.step(
                target_estimate,
                now,
                enabled=command_router.state.interception_active,
                width=frame.shape[1],
                height=frame.shape[0],
                paused=command_router.state.tracking_paused,
            )
            if frame_available:
                last_frame = frame

            render_frame = last_frame if last_frame is not None else frame
            if render_frame.size > 0:
                hand_overlay = render_frame.copy()
                draw_hand_landmarks(hand_overlay, last_hand)
                display = draw_tracking(
                    hand_overlay,
                    last_tracks,
                    tracker.position_histories,
                    tracker.selected_id,
                    confidence_threshold=confidence,
                    motion_estimate=motion_estimates.get(
                        command_router.state.locked_target_id
                        if command_router.state.target_locked
                        else tracker.selected_id
                    ),
                )
                draw_status(
                    display,
                    fps=fps,
                    paused=command_router.state.tracking_paused,
                    tracked_count=sum(
                        track.confidence >= confidence
                        or track.track_id == tracker.selected_id
                        for track in last_tracks
                    ),
                    selected_id=tracker.selected_id,
                    hand_detected=last_hand.detected,
                    gesture=last_hand.gesture.value,
                    stable_gesture=(
                        gesture_update.stable_gesture.value
                        if gesture_update.stable_gesture is not None
                        else "stabilizing"
                    ),
                    stable_frames=gesture_update.stable_frames,
                    gesture_stability=gesture_update.stability,
                    required_stable_frames=gesture_commands.required_stable_frames,
                    target_locked=command_router.state.target_locked,
                    interception_active=command_router.state.interception_active,
                    feedback=command_router.state.feedback,
                    pointing_ray=command_router.pointing_ray,
                    motion_estimate=motion_estimates.get(
                        command_router.state.locked_target_id
                        if command_router.state.target_locked
                        else tracker.selected_id
                    ),
                )
                if simulation.status != last_interception_status:
                    if simulation.status != InterceptionStatus.IDLE:
                        severity = (
                            EventSeverity.SUCCESS
                            if simulation.status == InterceptionStatus.INTERCEPTED
                            else EventSeverity.WARNING
                            if simulation.status
                            in (
                                InterceptionStatus.IMPOSSIBLE,
                                InterceptionStatus.TARGET_LOST,
                                InterceptionStatus.LOW_CONFIDENCE,
                            )
                            else EventSeverity.INFO
                        )
                        event_log.add(
                            f"INTERCEPTION: {simulation.status.value}",
                            severity,
                        )
                    last_interception_status = simulation.status
                selected_track = next(
                    (
                        track
                        for track in last_tracks
                        if track.track_id == tracker.selected_id
                    ),
                    None,
                )
                system_state = _system_state(
                    frame_available,
                    detection_enabled,
                    len(last_tracks),
                    command_router.state.tracking_paused,
                    tracker.selected_id,
                    command_router.state.target_locked,
                    motion_estimates.get(target_id),
                    command_router.state.interception_active,
                    simulation.status,
                )
                telemetry = DashboardTelemetry(
                    detected_count=len(last_tracks) if detection_enabled else 0,
                    tracked_count=sum(
                        track.confidence >= confidence
                        or track.track_id == tracker.selected_id
                        for track in last_tracks
                    ),
                    selected_id=tracker.selected_id,
                    selected_label=selected_track.label if selected_track else None,
                    selected_confidence=selected_track.confidence if selected_track else None,
                    target_locked=command_router.state.target_locked,
                    tracking_paused=command_router.state.tracking_paused,
                    detection_enabled=detection_enabled,
                    gestures_enabled=gestures_enabled,
                    hand_detected=last_hand.detected,
                    gesture=last_hand.gesture.value,
                    stable_gesture=stable_name or "stabilizing",
                    gesture_stability=gesture_update.stability,
                    gesture_command=(
                        gesture_update.events[-1].command.value
                        if gesture_update.events
                        else ""
                    ),
                    hands=(
                        (
                            HandTelemetry(
                                handedness=last_hand.handedness,
                                gesture=last_hand.gesture.value,
                                stable_gesture=stable_name or "stabilizing",
                                stability=gesture_update.stability,
                                command=(
                                    gesture_update.events[-1].command.value
                                    if gesture_update.events
                                    else ""
                                ),
                            ),
                        )
                        if last_hand.detected
                        else ()
                    ),
                    interception_active=command_router.state.interception_active,
                    fps=fps,
                    uptime_seconds=now - application_started,
                    source_label=_source_label(source),
                    source_running=capture is not None,
                    system_state=system_state,
                )
                display = draw_simulation_panel(
                    display,
                    simulation,
                    telemetry=telemetry,
                    events=event_log.events,
                    motion_estimate=motion_estimates.get(target_id),
                )
                display = dashboard_controls.draw(display, telemetry)
                cv2.imshow(config.WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if ord("A") <= key <= ord("Z"):
                key = ord(chr(key).lower())
            if key == ord("q"):
                break
            if key == ord("d"):
                toggle_detection()
            elif key == ord("h"):
                apply_dashboard_action(DashboardAction.GESTURES)
            elif key == ord("m"):
                apply_dashboard_action(DashboardAction.CAMERA)
            elif key == ord("v"):
                apply_dashboard_action(DashboardAction.VIDEO)
            was_active = command_router.state.interception_active
            if key not in (ord("d"), ord("h"), ord("m"), ord("v")) and command_router.handle_key(
                key, tracker, last_tracks, frame.shape
            ):
                validate_interception_activation(was_active)
                if key == ord("r"):
                    last_tracks = []
                    gesture_commands.reset()
                    motion_estimator.reset()
                    interceptor.reset()
                    event_log.add("SYSTEM STATE RESET", EventSeverity.WARNING)

            while True:
                action = dashboard_controls.next_action()
                if action is None:
                    break
                apply_dashboard_action(action)
            log_control_state_changes()

        return 0
    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()
        if gesture_recognizer is not None:
            gesture_recognizer.close()


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0.0 and 1.0")
    source = parse_source(args.source)

    if isinstance(source, int) and source < 0:
        parser.error("camera index must not be negative")
    if isinstance(source, str) and not Path(source).is_file():
        print(f"Error: video file does not exist: {source}", file=sys.stderr)
        return 1

    if args.imgsz < 32:
        parser.error("--imgsz must be at least 32 pixels")
    model_path = args.model or (
        config.OPEN_VOCABULARY_MODEL_PATH
        if args.detector == "world"
        else config.MODEL_PATH
    )
    vocabulary = (
        tuple(value.strip() for value in args.classes.split(",") if value.strip())
        if args.classes
        else config.OPEN_VOCABULARY_CLASSES
    )
    if args.detector == "world" and not vocabulary:
        parser.error("--classes must contain at least one class for --detector world")
    if args.legacy_opencv:
        return run(
            source,
            model_path,
            args.confidence,
            args.device,
            args.tracker,
            args.imgsz,
            args.detector,
            vocabulary,
        )

    application_launch_started = time.perf_counter()
    from interface.desktop import run_desktop_app

    return run_desktop_app(
        source,
        model_path,
        args.confidence,
        args.device,
        args.tracker,
        args.imgsz,
        application_launch_started,
        args.detector,
        vocabulary,
        not args.no_classification,
    )


if __name__ == "__main__":
    raise SystemExit(main())
