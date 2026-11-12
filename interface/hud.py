"""OpenCV drawing helpers for the JARVIS detection and tracking display."""

from __future__ import annotations

import cv2
import numpy as np

import config
from vision.detector import Detection
from prediction.motion import MotionEstimate, PredictionStatus
from interface.dashboard import DashboardEvent, DashboardTelemetry, EventSeverity
from robotics.interceptor import InterceptionStatus, SimulationSnapshot
from vision.tracker import TrackedObject

CYAN = (255, 220, 40)
BLUE = (210, 110, 30)
WHITE = (235, 245, 245)
PANEL = (18, 30, 35)
SELECTED_COLOR = (0, 210, 255)
MUTED = (130, 165, 175)


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw detection boxes, class names, and confidence values."""
    output = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy
        cv2.rectangle(output, (x1, y1), (x2, y2), CYAN, config.LINE_THICKNESS)
        label = f"{detection.label} {detection.confidence:.0%}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, config.FONT, config.FONT_SCALE, 1
        )
        label_top = max(0, y1 - text_height - baseline - 8)
        cv2.rectangle(
            output,
            (x1, label_top),
            (x1 + text_width + 10, y1),
            PANEL,
            cv2.FILLED,
        )
        cv2.putText(
            output,
            label,
            (x1 + 5, max(text_height, y1 - baseline - 4)),
            config.FONT,
            config.FONT_SCALE,
            WHITE,
            1,
            cv2.LINE_AA,
        )
    return output


def draw_status(
    frame: np.ndarray,
    fps: float,
    paused: bool,
    tracked_count: int = 0,
    selected_id: int | None = None,
    hand_detected: bool = False,
    gesture: str = "none",
    stable_gesture: str = "none",
    stable_frames: int = 0,
    gesture_stability: float = 0.0,
    required_stable_frames: int = config.GESTURE_STABLE_FRAMES,
    target_locked: bool = False,
    interception_active: bool = False,
    feedback: str = "",
    pointing_ray: tuple[tuple[int, int], tuple[int, int]] | None = None,
    motion_estimate: MotionEstimate | None = None,
) -> None:
    """Draw system state, gesture state, and actionable command feedback."""
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width, 34), PANEL, cv2.FILLED)
    status = "PAUSED" if paused else "VISION ONLINE"
    text = f"JARVIS  //  {status}  //  {fps:4.1f} FPS  //  TRACKS {tracked_count}"
    cv2.putText(
        frame,
        text,
        (12, 23),
        config.FONT,
        config.FONT_SCALE,
        CYAN if not paused else BLUE,
        1,
        cv2.LINE_AA,
    )
    camera_width = width - config.TRACKING_PANEL_WIDTH
    if pointing_ray is not None:
        cv2.arrowedLine(
            frame,
            pointing_ray[0],
            pointing_ray[1],
            SELECTED_COLOR,
            2,
            cv2.LINE_AA,
            tipLength=0.04,
        )

    panel_x = camera_width + 14
    if motion_estimate is not None:
        _draw_motion_readout(frame, panel_x, height - 266, motion_estimate)

    status_y = height - 132
    rows = (
        (f"HAND: {'DETECTED' if hand_detected else 'NOT DETECTED'}", CYAN),
        (f"GESTURE: {gesture.upper()}", WHITE),
        (
            f"STABILITY: {stable_gesture.upper()} "
            f"{min(stable_frames, required_stable_frames)}/{required_stable_frames} "
            f"{gesture_stability:.0%}",
            WHITE,
        ),
        (
            f"TARGET: {selected_id if selected_id is not None else 'NONE'}  "
            f"{'LOCKED' if target_locked else 'UNLOCKED'}",
            SELECTED_COLOR if target_locked else MUTED,
        ),
        (
            f"INTERCEPT: {'ACTIVE' if interception_active else 'OFF'}",
            SELECTED_COLOR if interception_active else MUTED,
        ),
        (feedback[:34].upper(), CYAN),
    )
    for row_index, (row_text, color) in enumerate(rows):
        cv2.putText(
            frame,
            row_text,
            (panel_x, status_y + row_index * 18),
            config.FONT,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        "C/N CYCLE  R RESET  Q QUIT",
        (panel_x, height - 30),
        config.FONT,
        0.36,
        MUTED,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "SPACE/T TRACK  L LOCK  I MODE",
        (panel_x, height - 12),
        config.FONT,
        0.36,
        MUTED,
        1,
        cv2.LINE_AA,
    )


def draw_tracking(
    frame: np.ndarray,
    tracks: list[TrackedObject],
    position_histories: dict[int, tuple[tuple[int, int], ...]],
    selected_id: int | None,
    confidence_threshold: float,
    motion_estimate: MotionEstimate | None = None,
) -> np.ndarray:
    """Draw consistent ID trails, detection boxes, and a tracking side panel."""
    height, width = frame.shape[:2]
    panel_width = config.TRACKING_PANEL_WIDTH
    output = np.zeros((height, width + panel_width, 3), dtype=frame.dtype)
    output[:, :width] = frame
    output[:, width:] = PANEL
    visible_tracks = [
        track
        for track in tracks
        if track.confidence >= confidence_threshold or track.track_id == selected_id
    ]

    for track_id, points in position_histories.items():
        if len(points) < 2:
            continue
        polyline = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
        color = SELECTED_COLOR if track_id == selected_id else _track_color(track_id)
        thickness = (
            config.SELECTED_TRAIL_THICKNESS
            if track_id == selected_id
            else config.TRAIL_THICKNESS
        )
        cv2.polylines(output, [polyline], False, color, thickness, cv2.LINE_AA)

    for track in visible_tracks:
        x1, y1, x2, y2 = track.bbox
        color = SELECTED_COLOR if track.track_id == selected_id else _track_color(track.track_id)
        thickness = (
            config.SELECTED_BOX_THICKNESS
            if track.track_id == selected_id
            else config.LINE_THICKNESS
        )
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        marker = "  [SELECTED]" if track.track_id == selected_id else ""
        label = f"ID {track.track_id} {track.label} {track.confidence:.0%}{marker}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, config.FONT, config.FONT_SCALE, 1
        )
        label_top = max(34, y1 - text_height - baseline - 8)
        cv2.rectangle(
            output,
            (x1, label_top),
            (min(width - 1, x1 + text_width + 10), y1),
            PANEL,
            cv2.FILLED,
        )
        cv2.putText(
            output,
            label,
            (x1 + 5, max(text_height + 34, y1 - baseline - 4)),
            config.FONT,
            config.FONT_SCALE,
            color,
            1,
            cv2.LINE_AA,
        )

    if motion_estimate is not None:
        _draw_prediction_overlay(output[:, :width], motion_estimate)

    _draw_tracking_panel(output, width, visible_tracks, position_histories, selected_id)
    return output


def draw_simulation_panel(
    frame: np.ndarray,
    snapshot: SimulationSnapshot,
    telemetry: DashboardTelemetry | None = None,
    events: tuple[DashboardEvent, ...] = (),
    motion_estimate: MotionEstimate | None = None,
) -> np.ndarray:
    """Append a compact, live 2D robot/target simulation beside the tracking HUD."""
    height, width = frame.shape[:2]
    panel_width = config.ROBOTICS_PANEL_WIDTH
    output = np.zeros((height, width + panel_width, 3), dtype=frame.dtype)
    output[:, :width] = frame
    output[:, width:] = PANEL
    panel_x = width
    cv2.line(output, (panel_x, 34), (panel_x, height), CYAN, 1)
    text_x = panel_x + 14
    cv2.putText(
        output,
        "ROBOTICS // INTERCEPT SIM",
        (text_x, 62),
        config.FONT,
        0.43,
        CYAN,
        1,
        cv2.LINE_AA,
    )
    status_color = _simulation_status_color(snapshot.status)
    cv2.putText(
        output,
        snapshot.status.value,
        (text_x, 86),
        config.FONT,
        0.43,
        status_color,
        1,
        cv2.LINE_AA,
    )

    view_x, view_y = panel_x + 12, 112
    view_width = panel_width - 24
    view_height = min(
        max(55, int(height * 0.26)),
        max(35, height - view_y - 205),
    )
    cv2.rectangle(
        output,
        (view_x, view_y),
        (view_x + view_width, view_y + view_height),
        (35, 54, 60),
        cv2.FILLED,
    )
    cv2.rectangle(
        output,
        (view_x, view_y),
        (view_x + view_width, view_y + view_height),
        MUTED,
        1,
        cv2.LINE_AA,
    )

    def project(point: tuple[float, float]) -> tuple[int, int]:
        scale = min(
            (view_width - 12) / max(1, snapshot.width),
            (view_height - 12) / max(1, snapshot.height),
        )
        return (
            max(view_x + 3, min(view_x + view_width - 3, int(view_x + 6 + point[0] * scale))),
            max(view_y + 3, min(view_y + view_height - 3, int(view_y + 6 + point[1] * scale))),
        )

    history = [project(sample.position) for sample in snapshot.target_history]
    if len(history) > 1:
        cv2.polylines(
            output,
            [np.asarray(history, dtype=np.int32).reshape((-1, 1, 2))],
            False,
            CYAN,
            1,
            cv2.LINE_AA,
        )
    if len(snapshot.predicted_trajectory) > 0 and snapshot.target_position is not None:
        predicted = [project(snapshot.target_position)] + [
            project(point) for point in snapshot.predicted_trajectory
        ]
        for start, end in zip(predicted, predicted[1:]):
            _draw_dashed_line(output, start, end, SELECTED_COLOR)
    if len(snapshot.robot_path) > 1:
        robot_path = np.asarray(
            [project(point) for point in snapshot.robot_path], dtype=np.int32
        ).reshape((-1, 1, 2))
        cv2.polylines(output, [robot_path], False, BLUE, 2, cv2.LINE_AA)

    start = project(snapshot.robot_start)
    robot = project(snapshot.robot_position)
    cv2.rectangle(
        output,
        (start[0] - 4, start[1] - 4),
        (start[0] + 4, start[1] + 4),
        MUTED,
        1,
        cv2.LINE_AA,
    )
    cv2.circle(output, robot, 6, BLUE, cv2.FILLED, cv2.LINE_AA)
    if snapshot.target_position is not None:
        target = project(snapshot.target_position)
        cv2.circle(output, target, 6, CYAN, 2, cv2.LINE_AA)
    if snapshot.interception_point is not None:
        point = project(snapshot.interception_point)
        cv2.drawMarker(
            output,
            point,
            SELECTED_COLOR,
            cv2.MARKER_CROSS,
            13,
            2,
            cv2.LINE_AA,
        )
    if snapshot.success_position is not None:
        cv2.drawMarker(
            output,
            project(snapshot.success_position),
            (80, 235, 100),
            cv2.MARKER_DIAMOND,
            16,
            2,
            cv2.LINE_AA,
        )

    info_y = view_y + view_height + 18
    target_text = (
        f"TARGET ID {snapshot.target_id}"
        if snapshot.target_id is not None
        else "TARGET ID --"
    )
    if telemetry is not None:
        state_rows = (
            (
                f"SYS {telemetry.system_state}  {telemetry.fps:.1f} FPS  "
                f"{telemetry.uptime_seconds:.0f}s",
                CYAN,
            ),
            (
                f"OBJ {telemetry.detected_count} DET / {telemetry.tracked_count} TRACK  "
                f"SRC {telemetry.source_label[:10]}",
                WHITE,
            ),
            (
                f"HANDS {len(telemetry.hands)}  "
                f"{'HAND' if telemetry.hand_detected else 'NO HAND'} "
                f"{telemetry.gesture.upper()[:8]} "
                f"{telemetry.stable_gesture.upper()[:8]} "
                f"{telemetry.gesture_stability:.0%}",
                WHITE,
            ),
            (
                f"LOCK {'ON' if telemetry.target_locked else 'OFF'}  "
                f"INTERCEPT {'ON' if telemetry.interception_active else 'OFF'}  "
                f"DETECT {'ON' if telemetry.detection_enabled else 'OFF'}",
                WHITE,
            ),
            (
                f"CMD {telemetry.gesture_command.upper()[:22] or '--'}",
                MUTED,
            ),
        )
        for index, (text, color) in enumerate(state_rows):
            cv2.putText(
                output,
                text,
                (text_x, info_y + index * 15),
                config.FONT,
                0.30,
                color,
                1,
                cv2.LINE_AA,
            )
        info_y += len(state_rows) * 15 + 2
    eta_text = (
        f"ETA {snapshot.interception_time:.2f}s"
        if snapshot.interception_time is not None
        else "ETA --"
    )
    distance_text = (
        f"DIST {snapshot.distance_pixels:.0f}px"
        if snapshot.distance_pixels is not None
        else "DIST --"
    )
    selected_text = (
        f"{target_text} {telemetry.selected_label[:9] if telemetry and telemetry.selected_label else ''} "
        f"{telemetry.selected_confidence:.0%}"
        if telemetry is not None and telemetry.selected_confidence is not None
        else target_text
    )
    cv2.putText(
        output,
        f"{selected_text}  {eta_text} {distance_text}",
        (text_x, info_y),
        config.FONT,
        0.31,
        WHITE,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"ROBOT {snapshot.robot_position[0]:.0f},{snapshot.robot_position[1]:.0f}",
        (text_x, info_y + 15),
        config.FONT,
        0.31,
        MUTED,
        1,
        cv2.LINE_AA,
    )
    if motion_estimate is not None:
        direction = (
            f"{motion_estimate.direction_degrees:.0f}deg"
            if motion_estimate.direction_degrees is not None
            else "STILL"
        )
        motion_rows = (
            f"POS {motion_estimate.measured_position[0]:.0f},"
            f"{motion_estimate.measured_position[1]:.0f}  "
            f"V {motion_estimate.velocity[0]:.0f},{motion_estimate.velocity[1]:.0f}px/s",
            f"SPEED {motion_estimate.speed:.0f}px/s  DIR {direction}  "
            f"{motion_estimate.status.value}",
        )
        for index, text in enumerate(motion_rows):
            cv2.putText(
                output,
                text,
                (text_x, info_y + 30 + index * 15),
                config.FONT,
                0.29,
                CYAN if index else WHITE,
                1,
                cv2.LINE_AA,
            )
        predictions = sorted(motion_estimate.predictions.items())
        prediction_text = "PRED " + " ".join(
            f"{horizon:g}s:{position[0]:.0f},{position[1]:.0f}"
            for horizon, position in predictions
        )
        cv2.putText(
            output,
            prediction_text[:43],
            (text_x, info_y + 60),
            config.FONT,
            0.29,
            MUTED,
            1,
            cv2.LINE_AA,
        )
    log_y = max(info_y + 78, height - 58)
    cv2.line(output, (text_x, log_y - 12), (panel_x + panel_width - 12, log_y - 12), (65, 155, 175), 1)
    cv2.putText(
        output,
        "EVENT LOG",
        (text_x, log_y),
        config.FONT,
        0.37,
        CYAN,
        1,
        cv2.LINE_AA,
    )
    for index, event_record in enumerate(events[-2:], start=1):
        cv2.putText(
            output,
            f"{event_record.timestamp} {event_record.message[:27]}",
            (text_x, log_y + index * 17),
            config.FONT,
            0.31,
            _event_color(event_record.severity),
            1,
            cv2.LINE_AA,
        )
    return output


def _draw_tracking_panel(
    frame: np.ndarray,
    camera_width: int,
    tracks: list[TrackedObject],
    histories: dict[int, tuple[tuple[int, int], ...]],
    selected_id: int | None,
) -> None:
    panel_x = camera_width
    cv2.line(frame, (panel_x, 34), (panel_x, frame.shape[0]), CYAN, 1)
    cv2.putText(
        frame,
        "TRACKING // BYTETRACK",
        (panel_x + 14, 62),
        config.FONT,
        config.FONT_SCALE,
        CYAN,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"ACTIVE OBJECTS: {len(tracks)}",
        (panel_x + 14, 88),
        config.FONT,
        config.FONT_SCALE,
        WHITE,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "ID  CLASS       POSITION  FRAMES",
        (panel_x + 14, 119),
        config.FONT,
        0.38,
        MUTED,
        1,
        cv2.LINE_AA,
    )
    row_y = 145
    for track in tracks:
        if row_y >= frame.shape[0] - 276:
            break
        selected = track.track_id == selected_id
        color = SELECTED_COLOR if selected else _track_color(track.track_id)
        class_name = track.label[:10]
        text = (
            f"{track.track_id:<3} {class_name:<10} "
            f"{track.center[0]:>4},{track.center[1]:<4} {track.frames_tracked:>4}"
        )
        cv2.putText(
            frame,
            text,
            (panel_x + 14, row_y),
            config.FONT,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
        row_y += 23

    if selected_id is not None and selected_id not in {track.track_id for track in tracks}:
        points = histories.get(selected_id, ())
        if row_y < frame.shape[0] - 25:
            cv2.putText(
                frame,
                f"SELECTED ID {selected_id} // TEMPORARILY LOST",
                (panel_x + 14, row_y + 8),
                config.FONT,
                0.38,
                SELECTED_COLOR,
                1,
                cv2.LINE_AA,
            )
        if points and row_y + 31 < frame.shape[0] - 8:
            x, y = points[-1]
            cv2.putText(
                frame,
                f"LAST POSITION {x},{y}",
                (panel_x + 14, row_y + 29),
                config.FONT,
                0.38,
                MUTED,
                1,
                cv2.LINE_AA,
            )


def _track_color(track_id: int) -> tuple[int, int, int]:
    hue = (track_id * 37) % 180
    hsv = np.uint8([[[hue, 190, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _simulation_status_color(status: InterceptionStatus) -> tuple[int, int, int]:
    if status == InterceptionStatus.INTERCEPTED:
        return 80, 235, 100
    if status in (InterceptionStatus.IMPOSSIBLE, InterceptionStatus.TARGET_LOST):
        return 80, 80, 245
    if status == InterceptionStatus.LOW_CONFIDENCE:
        return SELECTED_COLOR
    if status in (InterceptionStatus.APPROACHING, InterceptionStatus.POSSIBLE):
        return CYAN
    return MUTED


def _event_color(severity: EventSeverity) -> tuple[int, int, int]:
    if severity == EventSeverity.SUCCESS:
        return 80, 235, 100
    if severity in (EventSeverity.WARNING,):
        return 0, 175, 255
    if severity == EventSeverity.ERROR:
        return 80, 80, 245
    return MUTED


def _draw_motion_readout(
    frame: np.ndarray,
    x: int,
    y: int,
    estimate: MotionEstimate,
) -> None:
    cv2.putText(
        frame,
        f"MOTION // TARGET ID {estimate.track_id}",
        (x, y),
        config.FONT,
        0.4,
        SELECTED_COLOR,
        1,
        cv2.LINE_AA,
    )
    direction = (
        f"{estimate.direction_degrees:.0f} deg"
        if estimate.direction_degrees is not None
        else "STATIONARY"
    )
    rows = (
        (
            f"{estimate.status.value}  N={estimate.measurement_count}  "
            f"RESID={estimate.residual_pixels:.1f}px",
            CYAN if estimate.status != PredictionStatus.LOW_CONFIDENCE else SELECTED_COLOR,
        ),
        (
            f"MEAS {estimate.measured_position[0]:.0f},{estimate.measured_position[1]:.0f}  "
            f"FILTER {estimate.filtered_position[0]:.0f},{estimate.filtered_position[1]:.0f}",
            WHITE,
        ),
        (
            f"V=({estimate.velocity[0]:.0f},{estimate.velocity[1]:.0f})px/s  "
            f"SPEED={estimate.speed:.0f}px/s",
            WHITE,
        ),
        (
            f"DIR={direction}  ACC=({estimate.acceleration[0]:.0f},"
            f"{estimate.acceleration[1]:.0f})px/s2",
            WHITE,
        ),
    )
    for row_index, (text, color) in enumerate(rows, start=1):
        cv2.putText(
            frame,
            text,
            (x, y + row_index * 16),
            config.FONT,
            0.34,
            color,
            1,
            cv2.LINE_AA,
        )
    prediction_rows = []
    for horizon, position in sorted(estimate.predictions.items()):
        prediction_rows.append(
            f"{horizon:.1f}s=({position[0]:.0f},{position[1]:.0f})"
        )
    for row_index in range(0, len(prediction_rows), 2):
        text = "PRED " + "  ".join(prediction_rows[row_index : row_index + 2])
        cv2.putText(
            frame,
            text,
            (x, y + (5 + row_index // 2) * 16),
            config.FONT,
            0.34,
            CYAN,
            1,
            cv2.LINE_AA,
        )
    if estimate.seconds_since_measurement > 0.0:
        cv2.putText(
            frame,
            f"LAST MEASUREMENT {estimate.seconds_since_measurement:.1f}s AGO",
            (x, y + 7 * 16),
            config.FONT,
            0.34,
            MUTED,
            1,
            cv2.LINE_AA,
        )
    if estimate.velocity_clamped:
        cv2.putText(
            frame,
            "VELOCITY CLAMPED",
            (x, y + 8 * 16),
            config.FONT,
            0.34,
            SELECTED_COLOR,
            1,
            cv2.LINE_AA,
        )


def _draw_prediction_overlay(
    camera_frame: np.ndarray,
    estimate: MotionEstimate,
) -> None:
    filtered = tuple(int(round(value)) for value in estimate.filtered_position)
    measured = tuple(int(round(value)) for value in estimate.measured_position)
    cv2.line(camera_frame, measured, filtered, BLUE, 1, cv2.LINE_AA)
    cv2.circle(camera_frame, filtered, 6, CYAN, 2, cv2.LINE_AA)
    cv2.circle(camera_frame, measured, 3, WHITE, cv2.FILLED, cv2.LINE_AA)

    ordered_predictions = sorted(estimate.predictions.items())
    points = [filtered] + [
        tuple(int(round(value)) for value in position)
        for _, position in ordered_predictions
    ]
    for start, end in zip(points, points[1:]):
        _draw_dashed_line(camera_frame, start, end, SELECTED_COLOR)
    for (horizon, _), point in zip(ordered_predictions, points[1:]):
        color = _horizon_color(horizon)
        cv2.circle(camera_frame, point, 6, color, 2, cv2.LINE_AA)
        label_x = min(max(point[0] + 8, 2), max(2, camera_frame.shape[1] - 52))
        label_y = min(max(point[1] - 7, 42), camera_frame.shape[0] - 4)
        cv2.putText(
            camera_frame,
            f"+{horizon:g}s",
            (label_x, label_y),
            config.FONT,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )


def _draw_dashed_line(
    frame: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = float(np.hypot(dx, dy))
    if length <= 0.0:
        return
    dash_length = float(config.MOTION_TRAJECTORY_SEGMENT_PIXELS)
    segment_start = 0.0
    while segment_start < length:
        segment_end = min(segment_start + dash_length / 2.0, length)
        first = (
            int(start[0] + dx * segment_start / length),
            int(start[1] + dy * segment_start / length),
        )
        second = (
            int(start[0] + dx * segment_end / length),
            int(start[1] + dy * segment_end / length),
        )
        cv2.line(frame, first, second, color, 2, cv2.LINE_AA)
        segment_start += dash_length


def _horizon_color(horizon: float) -> tuple[int, int, int]:
    if horizon <= 0.5:
        return (70, 235, 120)
    if horizon <= 1.0:
        return (40, 190, 255)
    return SELECTED_COLOR
