"""PySide6 desktop presentation for the JARVIS application pipeline."""

from __future__ import annotations

from collections import deque
import sys
import time
from typing import Callable

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QShortcut,
    QTextCharFormat,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import config
from application.capture import CapturedFrame, LatestFrameCapture, source_label
from application.pipeline import PipelineCommand, PipelineSnapshot
from application.worker import LatestFrameVisionWorker, VisionWorkerState
from prediction.motion import MotionEstimate
from vision.recognition import ClassificationStatus, TrackRecognition
from robotics.interceptor import SimulationSnapshot
from vision.gestures import DetectedHand, GestureType
from vision.tracker import TrackedObject
from interface.visualization import (
    FINGER_CONNECTIONS,
    HAND_CONNECTIONS,
    HandLandmarkSmoother,
    hand_outline,
)
from interface.effect_renderer import EffectRenderer
from interface.ability_effect_renderer import AbilityEffectRenderer
from interface.ability_effects import AbilityEffectSnapshot
from vision.ability_library import load_ability_library
from vision.anime_gestures import AnimeRecognitionSnapshot
from interface.effects import EffectSnapshot


_STYLE = """
QMainWindow, QWidget#root { background: #040a10; color: #d9f4f8; }
QWidget { color: #d9f4f8; font-family: "Segoe UI"; font-size: 9pt; }
QFrame#topOverlay, QFrame#contextCard, QFrame#commandPalette,
QFrame#simulationCard, QFrame#toast {
    background: rgba(5, 15, 23, 218);
    border: 1px solid rgba(63, 180, 214, 125);
    border-radius: 10px;
}
QFrame#topOverlay { background: rgba(4, 12, 19, 192); border: none; }
QLabel#brand { color: #dffbff; font-size: 16pt; font-weight: 500; letter-spacing: 3px; }
QLabel#subtitle, QLabel#muted { color: #77a6b1; font-size: 8pt; }
QLabel#section { color: #64d7f0; font-size: 8pt; font-weight: 600; letter-spacing: 1px; }
QLabel#value { color: #d9f4f8; font-size: 9pt; }
QLabel#accent { color: #71e6fb; font-size: 9pt; font-weight: 600; }
QLabel#error { color: #ffd2cb; background: rgba(105, 35, 40, 225); border-radius: 6px; padding: 8px; }
QPushButton, QToolButton {
    background: rgba(12, 37, 51, 230); border: 1px solid rgba(64, 170, 204, 150);
    border-radius: 6px; padding: 7px 10px; color: #c9f4fb;
}
QPushButton:hover, QToolButton:hover { background: rgba(17, 62, 79, 245); border-color: #58d5f3; }
QPushButton:checked { background: rgba(22, 107, 129, 230); border-color: #78ebff; color: white; }
QPlainTextEdit {
    background: rgba(4, 12, 18, 245); color: #b6dce3; border: 1px solid rgba(63, 180, 214, 110);
    border-radius: 6px; font-family: Consolas, monospace; font-size: 8pt;
}
"""


class VideoCanvas(QWidget):
    target_selected = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._image = QImage()
        self._tracks: tuple[TrackedObject, ...] = ()
        self._histories: dict[int, tuple[tuple[int, int], ...]] = {}
        self._selected_id: int | None = None
        self._estimate: MotionEstimate | None = None
        self._recognition: dict[int, TrackRecognition] = {}
        self._pointing_ray: tuple[tuple[int, int], tuple[int, int]] | None = None
        self._hands: tuple[DetectedHand, ...] = ()
        self._anime = AnimeRecognitionSnapshot()
        self._effects: tuple[EffectSnapshot, ...] = ()
        self._ability_effects: tuple[AbilityEffectSnapshot, ...] = ()
        self._clone_images: dict[tuple[int, int], QImage] = {}
        self._hand_smoother = HandLandmarkSmoother()
        self._hand_timestamp: float | None = None
        self._show_history = True
        self._show_predictions = True
        self._status = "Waiting for camera"
        self._frame_size = (0, 0)
        self._render_times: deque[float] = deque(maxlen=30)
        self._image_rect = QRectF()

    @property
    def render_fps(self) -> float:
        if len(self._render_times) < 2:
            return 0.0
        elapsed = self._render_times[-1] - self._render_times[0]
        return (len(self._render_times) - 1) / elapsed if elapsed > 0 else 0.0

    def set_image(self, image: QImage, frame_size: tuple[int, int]) -> None:
        self._image = image
        self._frame_size = frame_size
        self.update()

    def set_snapshot(
        self,
        snapshot: PipelineSnapshot | None,
        overlay_age: float | None,
        show_history: bool,
        show_predictions: bool,
    ) -> None:
        self._show_history = show_history
        self._show_predictions = show_predictions
        if snapshot is None:
            self._tracks = ()
            self._histories = {}
            self._selected_id = None
            self._estimate = None
            self._recognition = {}
            self._pointing_ray = None
            self._hands = ()
            self._anime = AnimeRecognitionSnapshot()
            self._effects = ()
            self._ability_effects = ()
            self._clone_images.clear()
            self._hand_smoother.update((), time.perf_counter())
            self._hand_timestamp = None
            self._status = "Vision system initializing"
        else:
            overlay_current = overlay_age is None or overlay_age <= 0.5
            self._tracks = snapshot.tracks if overlay_current else ()
            self._histories = snapshot.position_histories
            self._selected_id = snapshot.selected_id
            self._estimate = snapshot.selected_estimate if overlay_current else None
            self._anime = snapshot.anime if overlay_current else AnimeRecognitionSnapshot()
            self._effects = snapshot.effects if overlay_current and snapshot.anime_effects_enabled else ()
            self._ability_effects = (
                getattr(snapshot, "ability_effects", ())
                if overlay_current and snapshot.anime_effects_enabled
                else ()
            )
            sprites = tuple(
                sprite
                for effect in self._ability_effects
                for sprite in effect.clone_sprites
            )
            current_clone_keys = {
                (sprite.effect_id, sprite.clone_index)
                for sprite in sprites
            }
            self._clone_images = {
                key: image
                for key, image in self._clone_images.items()
                if key in current_clone_keys
            }
            for sprite in sprites:
                key = (sprite.effect_id, sprite.clone_index)
                if key not in self._clone_images:
                    self._clone_images[key] = QImage(
                        sprite.rgba,
                        sprite.width,
                        sprite.height,
                        sprite.width * 4,
                        QImage.Format.Format_RGBA8888,
                    ).copy()
            self._recognition = snapshot.recognition if overlay_current else {}
            self._pointing_ray = snapshot.pointing_ray if overlay_current else None
            detected_hands = snapshot.hand.hands
            if not detected_hands and snapshot.hand.detected:
                detected_hands = (
                    DetectedHand(
                        snapshot.hand.landmarks,
                        snapshot.hand.handedness,
                        snapshot.hand.gesture,
                        snapshot.hand.confidence,
                    ),
                )
            if overlay_current and snapshot.timestamp != self._hand_timestamp:
                self._hands = self._hand_smoother.update(
                    detected_hands if snapshot.gestures_enabled else (),
                    snapshot.timestamp,
                )
                self._hand_timestamp = snapshot.timestamp
            elif not overlay_current:
                self._hands = ()
                self._hand_smoother.update((), time.perf_counter())
                self._hand_timestamp = None
            self._status = (
                f"Vision update {overlay_age:.1f}s ago"
                if overlay_age is not None and overlay_age > 0.15
                else f"{snapshot.system_state}  ·  {snapshot.inference_fps:.1f} inference FPS"
            )
        self.update()

    def set_overlay_options(self, show_history: bool, show_predictions: bool) -> None:
        self._show_history = show_history
        self._show_predictions = show_predictions
        self.update()

    def paintEvent(self, _event: object) -> None:
        del _event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#090c0e"))
        self._render_times.append(time.perf_counter())
        if self._image.isNull():
            center = self.rect().center()
            painter.setPen(QPen(QColor(48, 177, 213, 90), 1))
            painter.drawEllipse(center, 90, 90)
            painter.drawEllipse(center, 72, 72)
            painter.drawLine(center.x() - 112, center.y(), center.x() + 112, center.y())
            painter.drawLine(center.x(), center.y() - 112, center.x(), center.y() + 112)
            painter.setPen(QColor("#829099"))
            painter.setFont(QFont("Segoe UI", 13))
            painter.drawText(
                self.rect().adjusted(0, 100, 0, 0),
                Qt.AlignmentFlag.AlignCenter,
                self._status,
            )
            return

        scale = min(
            self.width() / self._image.width(),
            self.height() / self._image.height(),
        )
        destination = QRectF(
            0.0,
            0.0,
            self._image.width() * scale,
            self._image.height() * scale,
        )
        destination.moveCenter(QPointF(self.width() / 2, self.height() / 2))
        self._image_rect = destination
        painter.drawImage(destination, self._image)
        AbilityEffectRenderer.draw_clones(
            painter,
            destination,
            self._ability_effects,
            time.perf_counter(),
            self._clone_images,
        )
        scale_x = destination.width() / max(1, self._frame_size[0])
        scale_y = destination.height() / max(1, self._frame_size[1])

        def point(position: tuple[float, float]) -> QPointF:
            return QPointF(
                destination.left() + position[0] * scale_x,
                destination.top() + position[1] * scale_y,
            )

        if self._show_history:
            for track_id, positions in self._histories.items():
                if len(positions) < 2:
                    continue
                color = QColor(80, 222, 243, 145) if track_id == self._selected_id else QColor(145, 190, 198, 70)
                painter.setPen(QPen(color, 1.7 if track_id == self._selected_id else 0.9))
                for start, end in zip(positions, positions[1:]):
                    painter.drawLine(point(start), point(end))

        for track in self._tracks:
            x1, y1, x2, y2 = track.bbox
            box = QRectF(point((x1, y1)), point((x2, y2))).normalized()
            selected = track.track_id == self._selected_id
            color = QColor("#75e8fa" if selected else "#a5c2c9")
            self._draw_corner_brackets(painter, box, color, 13 if selected else 8)
            if track.track_id == self._selected_id:
                painter.setPen(color)
                painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
                recognition = self._recognition.get(track.track_id)
                display_label = (
                    f"{recognition.detector_label} → {recognition.refined_label}"
                    if recognition
                    and recognition.status == ClassificationStatus.REFINED
                    and recognition.refined_label
                    else track.label
                )
                painter.drawText(
                    QPointF(box.left(), max(destination.top() + 15, box.top() - 8)),
                    f"{display_label.upper()}   /   {track.track_id}",
                )
            else:
                painter.setPen(QColor(218, 237, 241, 190))
                painter.setFont(QFont("Segoe UI", 7))
                painter.drawText(
                    QPointF(box.left(), max(destination.top() + 12, box.top() - 5)),
                    f"{track.label.upper()}  {track.track_id}",
                )
            if selected:
                center = point(track.center)
                painter.setPen(QPen(QColor("#75e8fa"), 1))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(center, 11, 11)
                painter.drawLine(center + QPointF(-17, 0), center + QPointF(-12, 0))
                painter.drawLine(center + QPointF(12, 0), center + QPointF(17, 0))
                painter.drawLine(center + QPointF(0, -17), center + QPointF(0, -12))
                painter.drawLine(center + QPointF(0, 12), center + QPointF(0, 17))

        if self._estimate is not None:
            estimate = self._estimate
            measured = point(estimate.measured_position)
            filtered = point(estimate.filtered_position)
            painter.setPen(QPen(QColor("#a8b8bf"), 1, Qt.PenStyle.DotLine))
            painter.drawLine(measured, filtered)
            painter.setPen(QPen(QColor("#f3f5f6"), 1))
            painter.setBrush(QColor("#56d4e6"))
            painter.drawEllipse(filtered, 4.5, 4.5)
            if self._show_predictions:
                predicted = [filtered] + [
                    point(position)
                    for _, position in sorted(estimate.predictions.items())
                ]
                painter.setPen(QPen(QColor("#52c8d8"), 1.6, Qt.PenStyle.DashLine))
                for start, end in zip(predicted, predicted[1:]):
                    painter.drawLine(start, end)
                for (horizon, _), marker in zip(
                    sorted(estimate.predictions.items()), predicted[1:]
                ):
                    painter.setBrush(QColor("#101316"))
                    painter.drawEllipse(marker, 4, 4)
                    painter.setPen(QColor("#d7e4e8"))
                    painter.drawText(marker + QPointF(7, -5), f"+{horizon:g}s")

        if self._pointing_ray is not None:
            painter.setPen(QPen(QColor(86, 222, 255, 170), 1.2, Qt.PenStyle.DashLine))
            painter.drawLine(
                point(self._pointing_ray[0]),
                point(self._pointing_ray[1]),
            )
        EffectRenderer.draw(
            painter,
            self._image_rect,
            self._effects,
            time.perf_counter(),
        )
        AbilityEffectRenderer.draw(
            painter,
            self._image_rect,
            self._ability_effects,
            time.perf_counter(),
        )
        self._draw_hands(painter, point)

    @staticmethod
    def _draw_corner_brackets(
        painter: QPainter,
        box: QRectF,
        color: QColor,
        length: float,
    ) -> None:
        painter.setPen(QPen(color, 1.3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap))
        corners = (
            (box.topLeft(), 1, 1),
            (box.topRight(), -1, 1),
            (box.bottomLeft(), 1, -1),
            (box.bottomRight(), -1, -1),
        )
        for corner, horizontal, vertical in corners:
            painter.drawLine(corner, corner + QPointF(horizontal * length, 0))
            painter.drawLine(corner, corner + QPointF(0, vertical * length))

    def _draw_hands(
        self,
        painter: QPainter,
        point: Callable[[tuple[float, float]], QPointF],
    ) -> None:
        painter.save()
        painter.setClipRect(self._image_rect)
        for index, hand in enumerate(self._hands):
            if len(hand.landmarks) != 21:
                continue
            gesture_color = {
                GestureType.POINTING: QColor("#6cecff"),
                GestureType.OPEN_PALM: QColor("#8bffe0"),
                GestureType.PINCH: QColor("#c1f8ff"),
                GestureType.FIST: QColor("#67c9ff"),
            }.get(hand.gesture, QColor("#68dff5"))

            def hand_point(position: tuple[float, float]) -> QPointF:
                return point(
                    (
                        position[0] * self._frame_size[0],
                        position[1] * self._frame_size[1],
                    )
                )

            points = [hand_point((x, y)) for x, y, _ in hand.landmarks]
            hull = hand_outline(hand.landmarks)
            if len(hull) >= 3:
                polygon = [hand_point(position) for position in hull]
                path = QPainterPath(polygon[0])
                for current, following in zip(polygon, polygon[1:] + polygon[:1]):
                    midpoint = (current + following) / 2
                    path.quadTo(current, midpoint)
                path.closeSubpath()
                fill_alpha = 30 if hand.gesture == GestureType.OPEN_PALM else 18
                painter.setBrush(QColor(20, 196, 233, fill_alpha))
                contour_color = QColor(
                    gesture_color.red(),
                    gesture_color.green(),
                    gesture_color.blue(),
                    48,
                )
                painter.setPen(QPen(contour_color, 5))
                painter.drawPath(path)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(gesture_color, 1.15))
                painter.drawPath(path)

            active_color = gesture_color
            for start, end in HAND_CONNECTIONS:
                if hand.gesture == GestureType.POINTING and (start, end) in FINGER_CONNECTIONS["index"]:
                    painter.setPen(QPen(QColor("#d5fcff"), 1.8))
                else:
                    painter.setPen(QPen(QColor(95, 220, 244, 180), 0.85))
                painter.drawLine(points[start], points[end])

            important = {0, 4, 8, 12, 16, 20}
            if hand.gesture == GestureType.FIST:
                important.update((5, 9, 13, 17))
            if hand.gesture == GestureType.PINCH:
                painter.setPen(QPen(QColor("#f0fdff"), 1.4))
                painter.drawLine(points[4], points[8])
                important.update((4, 8))
            if hand.gesture == GestureType.POINTING:
                dx = points[8].x() - points[6].x()
                dy = points[8].y() - points[6].y()
                magnitude = max(float(np.hypot(dx, dy)), 1.0)
                ray = min(self.width(), self.height()) * 0.22
                endpoint = QPointF(
                    points[8].x() + dx / magnitude * ray,
                    points[8].y() + dy / magnitude * ray,
                )
                painter.setPen(QPen(QColor(101, 236, 255, 175), 1.0, Qt.PenStyle.DashLine))
                painter.drawLine(points[8], endpoint)
                important.add(8)
            for landmark_index in important:
                radius = 2.8 if landmark_index in (4, 8, 12, 16, 20) else 2.1
                painter.setPen(QPen(QColor(205, 250, 255, 220), 0.8))
                painter.setBrush(active_color if landmark_index in (4, 8) and hand.gesture == GestureType.PINCH else QColor(9, 51, 66, 235))
                painter.drawEllipse(points[landmark_index], radius, radius)

            xs = [position.x() for position in points]
            ys = [position.y() for position in points]
            label = hand.handedness or f"HAND {index + 1}"
            if hand.gesture != GestureType.NONE:
                label += f"  /  {hand.gesture.value.upper()}"
            painter.setPen(QColor("#aaf5ff"))
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.DemiBold))
            painter.drawText(QPointF(min(xs), max(self._image_rect.top() + 13, min(ys) - 7)), label)
        if self._anime.pose is not None and self._anime.pose.status in (
            "RECOGNIZING",
            "RECOGNIZED",
            "HELD",
        ):
            pose = self._anime.pose
            anchor = QPointF(
                self._image_rect.left() + pose.center[0] * self._image_rect.width(),
                self._image_rect.top() + pose.center[1] * self._image_rect.height(),
            )
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
            painter.setPen(QColor("#dbfbff"))
            painter.drawText(anchor + QPointF(10, -14), f"{pose.display_name}  /  {pose.status}")
        if self._anime.sequence_progress:
            progress = self._anime.sequence_progress[0]
            painter.setFont(QFont("Consolas", 8, QFont.Weight.DemiBold))
            painter.setPen(QColor("#e9c879"))
            painter.drawText(
                QPointF(self._image_rect.left() + 18, self._image_rect.top() + 54),
                f"SEQUENCE  {progress.text}",
            )
        if self._effects:
            active_type = self._effects[-1].effect_type.value.replace("_", " ").upper()
            painter.setFont(QFont("Consolas", 8, QFont.Weight.DemiBold))
            painter.setPen(QColor("#bdeeff"))
            painter.drawText(
                QPointF(self._image_rect.right() - 230, self._image_rect.bottom() - 16),
                f"ABILITY  /  {active_type}",
            )
        painter.restore()

    def mousePressEvent(self, event: object) -> None:
        if not hasattr(event, "position") or not self._frame_size[0]:
            return
        position = event.position()
        if not self._image_rect.contains(position):
            return
        image_x = (position.x() - self._image_rect.left()) * self._frame_size[0] / self._image_rect.width()
        image_y = (position.y() - self._image_rect.top()) * self._frame_size[1] / self._image_rect.height()
        for track in reversed(self._tracks):
            x1, y1, x2, y2 = track.bbox
            if x1 <= image_x <= x2 and y1 <= image_y <= y2:
                self.target_selected.emit(track.track_id)
                return
        super().mousePressEvent(event)


class SimulationCanvas(QWidget):
    """Render robot motion and target interception state in image-space pixels."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("simulationCanvas")
        self.setMinimumHeight(112)
        self.setMaximumHeight(170)
        self._snapshot: SimulationSnapshot | None = None

    def set_snapshot(self, snapshot: SimulationSnapshot) -> None:
        self._snapshot = snapshot
        self.update()

    def paintEvent(self, _event: object) -> None:
        del _event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(3, 12, 19, 170))
        snapshot = self._snapshot
        if snapshot is None or snapshot.width < 2 or snapshot.height < 2:
            painter.setPen(QColor("#829099"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Simulation awaits camera dimensions")
            return

        footer_height = 22
        bounds = QRectF(8, 8, self.width() - 16, self.height() - footer_height - 12)
        scale = min(bounds.width() / snapshot.width, bounds.height() / snapshot.height)
        image_rect = QRectF(
            bounds.left(),
            bounds.top(),
            snapshot.width * scale,
            snapshot.height * scale,
        )
        image_rect.moveCenter(bounds.center())

        def point(position: tuple[float, float]) -> QPointF:
            return QPointF(
                image_rect.left() + position[0] * scale,
                image_rect.top() + position[1] * scale,
            )

        painter.save()
        painter.setClipRect(bounds)
        for start, end in zip(snapshot.target_history, snapshot.target_history[1:]):
            painter.setPen(QPen(QColor("#8d7044"), 1))
            painter.drawLine(point(start.position), point(end.position))
        for start, end in zip(snapshot.robot_path, snapshot.robot_path[1:]):
            painter.setPen(QPen(QColor("#4fa8d1"), 1.5))
            painter.drawLine(point(start), point(end))
        for start, end in zip(snapshot.predicted_trajectory, snapshot.predicted_trajectory[1:]):
            painter.setPen(QPen(QColor("#e7ad54"), 1.5, Qt.PenStyle.DashLine))
            painter.drawLine(point(start), point(end))

        if snapshot.target_position is not None:
            target = point(snapshot.target_position)
            painter.setBrush(QColor("#e7ad54"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(target, 4, 4)
        robot = point(snapshot.robot_position)
        painter.setBrush(QColor("#56c4e3"))
        painter.setPen(QPen(QColor("#c4eff8"), 1))
        painter.drawEllipse(robot, 4, 4)
        if snapshot.interception_point is not None:
            intercept = point(snapshot.interception_point)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#85d69a"), 1.7))
            painter.drawEllipse(intercept, 6, 6)
            painter.drawLine(intercept + QPointF(-4, -4), intercept + QPointF(4, 4))
            painter.drawLine(intercept + QPointF(-4, 4), intercept + QPointF(4, -4))
        painter.restore()

        painter.setPen(QColor("#a8b6bc"))
        painter.setFont(QFont("Segoe UI", 8))
        eta = (
            f"ETA {snapshot.interception_time:.2f}s"
            if snapshot.interception_time is not None
            else "ETA --"
        )
        target_id = (
            f"TARGET {snapshot.target_id}"
            if snapshot.target_id is not None
            else "NO TARGET"
        )
        status = snapshot.status.value.replace("INTERCEPTION ", "")
        robot_position = (
            f"ROBOT {snapshot.robot_position[0]:.0f},{snapshot.robot_position[1]:.0f}"
        )
        painter.drawText(
            8,
            self.height() - 5,
            f"{status}  ·  {target_id}  ·  {eta}  ·  {robot_position}",
        )


class JarvisWindow(QMainWindow):
    """Responsive desktop shell; only worker snapshots cross into Qt widgets."""

    def __init__(
        self,
        source: int | str,
        model_path: str,
        confidence: float,
        device: str | None,
        tracker_config: str,
        inference_size: int,
        detector_type: str = config.DETECTOR_TYPE,
        vocabulary: tuple[str, ...] = config.OPEN_VOCABULARY_CLASSES,
        classification_enabled: bool = config.CLASSIFICATION_ENABLED,
    ) -> None:
        super().__init__()
        self.setWindowTitle("JARVIS | Perception Interface")
        self.setMinimumSize(900, 600)
        self.resize(1440, 900)
        self.setStyleSheet(_STYLE)
        self._capture = LatestFrameCapture(source)
        self._vision = LatestFrameVisionWorker(
            self._capture,
            model_path,
            confidence,
            device,
            tracker_config,
            inference_size,
            detector_type,
            vocabulary,
            classification_enabled,
        )
        self._last_frame_sequence = -1
        self.time_to_visible_ms: float | None = None
        self._last_events: set[tuple[str, str]] = set()
        self._snapshot: PipelineSnapshot | None = None
        self._show_history = True
        self._show_predictions = True
        self._build_ui()
        self._connect_controls()
        self._install_shortcuts()
        self._timer = QTimer(self)
        self._timer.setInterval(config.GUI_REFRESH_INTERVAL_MS)
        self._timer.timeout.connect(self._refresh)

    def start_workers(self) -> None:
        self._capture.start()
        self._vision.start()
        self._timer.start()

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.root = root
        self.canvas = VideoCanvas(root)
        self.top_overlay = QFrame(root)
        self.top_overlay.setObjectName("topOverlay")
        header_layout = QHBoxLayout(self.top_overlay)
        header_layout.setContentsMargins(16, 2, 14, 2)
        header_layout.setSpacing(12)
        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title = QLabel("J A R V I S")
        title.setObjectName("brand")
        subtitle = QLabel("PERCEPTION INTERFACE")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header_layout.addLayout(title_box)
        header_layout.addStretch(1)
        self.system_indicator = QLabel("●")
        self.system_indicator.setObjectName("accent")
        header_layout.addWidget(self.system_indicator)
        self.initialization_label = QLabel("INITIALIZING")
        self.initialization_label.setObjectName("accent")
        header_layout.addWidget(self.initialization_label)
        self.camera_status = QLabel("NO INPUT")
        self.camera_status.setObjectName("muted")
        header_layout.addWidget(self.camera_status)
        self.menu_button = QPushButton("COMMANDS   +")
        header_layout.addWidget(self.menu_button)

        self.error_label = QLabel(root)
        self.error_label.setObjectName("error")
        self.error_label.setWordWrap(True)
        self.error_label.hide()

        self.target_card = QFrame(root)
        self.target_card.setObjectName("contextCard")
        target_layout = QVBoxLayout(self.target_card)
        target_layout.setContentsMargins(14, 12, 14, 12)
        target_layout.setSpacing(5)
        target_heading = QLabel("TARGET ACQUIRED")
        target_heading.setObjectName("section")
        target_layout.addWidget(target_heading)
        self.target_name = QLabel("No target selected")
        self.target_name.setObjectName("accent")
        self.target_name.setWordWrap(True)
        target_layout.addWidget(self.target_name)
        self.target_summary = QLabel("Select a tracked object to inspect.")
        self.target_summary.setObjectName("value")
        self.target_summary.setWordWrap(True)
        target_layout.addWidget(self.target_summary)
        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText("MOTION DETAILS  +")
        self.advanced_toggle.setCheckable(True)
        target_layout.addWidget(self.advanced_toggle)
        self.advanced = QFrame()
        advanced_layout = QVBoxLayout(self.advanced)
        advanced_layout.setContentsMargins(0, 3, 0, 0)
        self.prediction_values = QLabel("No prediction available")
        self.prediction_values.setObjectName("muted")
        self.prediction_values.setWordWrap(True)
        advanced_layout.addWidget(self.prediction_values)
        self.advanced.hide()
        target_layout.addWidget(self.advanced)
        self.target_card.hide()

        self.gesture_feedback = QLabel(root)
        self.gesture_feedback.setObjectName("section")
        self.gesture_feedback.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gesture_feedback.hide()

        self.command_button = QPushButton("OPEN COMMANDS   /", root)
        self.command_palette = QFrame(root)
        self.command_palette.setObjectName("commandPalette")
        palette_layout = QVBoxLayout(self.command_palette)
        palette_layout.setContentsMargins(12, 10, 12, 12)
        palette_heading = QLabel("COMMAND INTERFACE")
        palette_heading.setObjectName("section")
        palette_layout.addWidget(palette_heading)
        control_layout = QGridLayout()
        control_layout.setHorizontalSpacing(6)
        control_layout.setVerticalSpacing(6)
        palette_layout.addLayout(control_layout)
        self.buttons: dict[str, QPushButton] = {}
        button_specs = (
            ("camera", "Start camera"),
            ("video", "Open video"),
            ("pause", "Pause"),
            ("detection", "Detection on"),
            ("gestures", "Gestures on"),
            ("lock", "Lock target"),
            ("interception", "Start intercept"),
            ("cycle", "Next target"),
            ("reset_tracking", "Reset tracking"),
            ("reset_simulation", "Reset simulation"),
            ("classification", "Classification on"),
            ("simulation", "Simulation view"),
            ("anime", "Anime gestures on"),
            ("effects", "Anime effects on"),
            ("abilities", "Abilities on"),
        )
        for index, (key, text) in enumerate(button_specs):
            button = QPushButton(text, self.command_palette)
            button.setObjectName(key)
            row, column = divmod(index, 2)
            control_layout.addWidget(button, row, column)
            self.buttons[key] = button
        options = QHBoxLayout()
        self.trail_toggle = QToolButton()
        self.trail_toggle.setText("TRAILS")
        self.trail_toggle.setCheckable(True)
        self.trail_toggle.setChecked(True)
        self.prediction_toggle = QToolButton()
        self.prediction_toggle.setText("PREDICTIONS")
        self.prediction_toggle.setCheckable(True)
        self.prediction_toggle.setChecked(True)
        self.log_toggle = QToolButton()
        self.log_toggle.setText("EVENTS")
        self.log_toggle.setCheckable(True)
        options.addWidget(self.trail_toggle)
        options.addWidget(self.prediction_toggle)
        options.addWidget(self.log_toggle)
        palette_layout.addLayout(options)
        ability_heading = QLabel("ABILITY LIBRARY  /  CALIBRATION")
        ability_heading.setObjectName("section")
        palette_layout.addWidget(ability_heading)
        self.ability_selector = QComboBox(self.command_palette)
        self.ability_definitions = load_ability_library(config.ABILITY_LIBRARY_PATH)
        for definition in self.ability_definitions:
            self.ability_selector.addItem(
                definition.display_name,
                definition.ability_id,
            )
        palette_layout.addWidget(self.ability_selector)
        self.ability_description = QLabel(
            self.ability_definitions[0].description if self.ability_definitions else ""
        )
        self.ability_description.setObjectName("muted")
        self.ability_description.setWordWrap(True)
        palette_layout.addWidget(self.ability_description)
        calibration_controls = QHBoxLayout()
        self.ability_sample_status = QLabel("0 pose samples")
        self.ability_sample_status.setObjectName("muted")
        calibration_controls.addWidget(self.ability_sample_status, 1)
        self.record_ability_sample = QPushButton("Record pose")
        calibration_controls.addWidget(self.record_ability_sample)
        self.record_ability_movement = QPushButton("Record movement")
        calibration_controls.addWidget(self.record_ability_movement)
        palette_layout.addLayout(calibration_controls)
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("POSE SIMILARITY")
        threshold_label.setObjectName("muted")
        threshold_layout.addWidget(threshold_label)
        self.ability_similarity = QDoubleSpinBox(self.command_palette)
        self.ability_similarity.setRange(0.5, 0.98)
        self.ability_similarity.setSingleStep(0.02)
        self.ability_similarity.setDecimals(2)
        self.ability_similarity.setKeyboardTracking(False)
        self.ability_similarity.setValue(config.ABILITY_MIN_CALIBRATION_SIMILARITY)
        threshold_layout.addWidget(self.ability_similarity)
        palette_layout.addLayout(threshold_layout)
        self.event_log = QPlainTextEdit(self.command_palette)
        self.event_log.setReadOnly(True)
        self.event_log.setMaximumBlockCount(config.GUI_EVENT_LOG_LIMIT)
        self.event_log.setMaximumHeight(132)
        self.event_log.hide()
        palette_layout.addWidget(self.event_log)
        self.command_palette.hide()

        self.simulation_card = QFrame(root)
        self.simulation_card.setObjectName("simulationCard")
        simulation_layout = QVBoxLayout(self.simulation_card)
        simulation_layout.setContentsMargins(10, 8, 10, 8)
        simulation_title = QLabel("INTERCEPTION  /  IMAGE SPACE")
        simulation_title.setObjectName("section")
        simulation_layout.addWidget(simulation_title)
        self.simulation_canvas = SimulationCanvas()
        self.simulation_canvas.setMinimumHeight(130)
        self.simulation_canvas.setMaximumHeight(175)
        simulation_layout.addWidget(self.simulation_canvas)
        self.simulation_card.hide()
        self._simulation_pinned = False
        self._events_visible = False

        self.setCentralWidget(root)
        self.statusBar().hide()
        self._position_overlays()

    def _connect_controls(self) -> None:
        self.menu_button.clicked.connect(self._toggle_commands)
        self.command_button.clicked.connect(self._toggle_commands)
        self.buttons["camera"].clicked.connect(self._toggle_camera)
        self.buttons["video"].clicked.connect(self._choose_video)
        self.buttons["pause"].clicked.connect(
            lambda: self._submit(PipelineCommand.TOGGLE_PAUSE)
        )
        self.buttons["detection"].clicked.connect(
            lambda: self._submit(PipelineCommand.TOGGLE_DETECTION)
        )
        self.buttons["gestures"].clicked.connect(
            lambda: self._submit(PipelineCommand.TOGGLE_GESTURES)
        )
        self.buttons["classification"].clicked.connect(
            lambda: self._submit(PipelineCommand.TOGGLE_CLASSIFICATION)
        )
        self.buttons["lock"].clicked.connect(
            lambda: self._submit(PipelineCommand.TOGGLE_LOCK)
        )
        self.buttons["interception"].clicked.connect(
            lambda: self._submit(PipelineCommand.TOGGLE_INTERCEPTION)
        )
        self.buttons["cycle"].clicked.connect(
            lambda: self._submit(PipelineCommand.CYCLE_TARGET)
        )
        self.buttons["reset_tracking"].clicked.connect(
            lambda: self._submit(PipelineCommand.RESET_TRACKING)
        )
        self.buttons["reset_simulation"].clicked.connect(
            lambda: self._submit(PipelineCommand.RESET_SIMULATION)
        )
        self.buttons["simulation"].clicked.connect(self._toggle_simulation_view)
        self.buttons["anime"].clicked.connect(
            lambda: self._submit(PipelineCommand.TOGGLE_ANIME_GESTURES)
        )
        self.buttons["effects"].clicked.connect(
            lambda: self._submit(PipelineCommand.TOGGLE_ANIME_EFFECTS)
        )
        self.buttons["abilities"].clicked.connect(
            lambda: self._submit(PipelineCommand.TOGGLE_ABILITIES)
        )
        self.ability_selector.currentIndexChanged.connect(
            self._select_ability
        )
        self.record_ability_sample.clicked.connect(
            lambda: self._submit(
                PipelineCommand.RECORD_ABILITY_SAMPLE,
                ability_id=self._selected_ability_id(),
            )
        )
        self.record_ability_movement.clicked.connect(
            lambda: self._submit(
                PipelineCommand.TOGGLE_ABILITY_MOVEMENT_RECORDING,
                ability_id=self._selected_ability_id(),
            )
        )
        self.ability_similarity.valueChanged.connect(
            self._set_ability_similarity
        )
        self.canvas.target_selected.connect(
            lambda track_id: self._submit(PipelineCommand.SELECT_TARGET, track_id)
        )
        self.trail_toggle.toggled.connect(self._set_overlay_options)
        self.prediction_toggle.toggled.connect(self._set_overlay_options)
        self.log_toggle.toggled.connect(self._toggle_events)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)

    def _install_shortcuts(self) -> None:
        shortcuts = (
            ("Space", PipelineCommand.TOGGLE_PAUSE),
            ("T", PipelineCommand.TOGGLE_PAUSE),
            ("L", PipelineCommand.TOGGLE_LOCK),
            ("I", PipelineCommand.TOGGLE_INTERCEPTION),
            ("N", PipelineCommand.CYCLE_TARGET),
            ("C", PipelineCommand.CYCLE_TARGET),
            ("R", PipelineCommand.RESET_TRACKING),
            ("D", PipelineCommand.TOGGLE_DETECTION),
            ("H", PipelineCommand.TOGGLE_GESTURES),
        )
        self._shortcuts: list[QShortcut] = []
        for key, command in shortcuts:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(lambda selected=command: self._submit(selected))
            self._shortcuts.append(shortcut)
        quit_shortcut = QShortcut(QKeySequence("Q"), self)
        quit_shortcut.activated.connect(self.close)
        self._shortcuts.append(quit_shortcut)
        camera_shortcut = QShortcut(QKeySequence("M"), self)
        camera_shortcut.activated.connect(self._toggle_camera)
        self._shortcuts.append(camera_shortcut)
        video_shortcut = QShortcut(QKeySequence("V"), self)
        video_shortcut.activated.connect(self._choose_video)
        self._shortcuts.append(video_shortcut)
        menu_shortcut = QShortcut(QKeySequence("Tab"), self)
        menu_shortcut.activated.connect(self._toggle_commands)
        self._shortcuts.append(menu_shortcut)
        escape_shortcut = QShortcut(QKeySequence("Escape"), self)
        escape_shortcut.activated.connect(self.command_palette.hide)
        self._shortcuts.append(escape_shortcut)

    def _submit(
        self,
        command: PipelineCommand,
        target_id: int | None = None,
        *,
        ability_id: str | None = None,
        ability_threshold: float | None = None,
    ) -> None:
        self._vision.submit(command, target_id, ability_id, ability_threshold)

    def _selected_ability_id(self) -> str:
        selected = self.ability_selector.currentData()
        return str(selected) if selected is not None else ""

    def _select_ability(self, index: int) -> None:
        selected_id = str(self.ability_selector.itemData(index) or "")
        definition = next(
            (
                item for item in self.ability_definitions
                if item.ability_id == selected_id
            ),
            None,
        )
        self.ability_description.setText(
            definition.description if definition is not None else "No ability selected."
        )
        self._submit(
            PipelineCommand.SELECT_ABILITY,
            ability_id=selected_id,
        )

    def _set_ability_similarity(self, value: float) -> None:
        self._submit(
            PipelineCommand.SET_ABILITY_THRESHOLD,
            ability_threshold=value,
        )

    def _toggle_camera(self) -> None:
        current = self._capture.latest
        if isinstance(current.source, int) and current.running:
            self._capture.request_source(None)
            self._submit(PipelineCommand.RESET_TRACKING)
        else:
            self._capture.request_source(config.DEFAULT_CAMERA_INDEX)
            self._submit(PipelineCommand.RESET_TRACKING)

    def _choose_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open video",
            "",
            "Video files (*.mp4 *.avi *.mov *.mkv *.wmv);;All files (*)",
        )
        if path:
            self._capture.request_source(path)
            self._submit(PipelineCommand.RESET_TRACKING)

    def _set_overlay_options(self, _checked: bool) -> None:
        del _checked
        self._show_history = self.trail_toggle.isChecked()
        self._show_predictions = self.prediction_toggle.isChecked()
        self.canvas.set_overlay_options(self._show_history, self._show_predictions)

    def _toggle_commands(self) -> None:
        visible = not self.command_palette.isVisible()
        self.command_palette.setVisible(visible)
        self.menu_button.setText("COMMANDS   −" if visible else "COMMANDS   +")
        self._position_overlays()

    def _toggle_events(self, visible: bool) -> None:
        self._events_visible = visible
        self.event_log.setVisible(visible)
        self._position_overlays()

    def _toggle_simulation_view(self) -> None:
        self._simulation_pinned = not self._simulation_pinned
        self.buttons["simulation"].setText(
            "Hide simulation" if self._simulation_pinned else "Simulation view"
        )
        self._position_overlays()

    def _toggle_advanced(self, visible: bool) -> None:
        self.advanced.setVisible(visible)
        self.advanced_toggle.setText(
            "MOTION DETAILS  −" if visible else "MOTION DETAILS  +"
        )
        self._position_overlays()

    def _position_overlays(self) -> None:
        if not hasattr(self, "root"):
            return
        width = self.root.width()
        height = self.root.height()
        self.canvas.setGeometry(self.root.rect())
        self.top_overlay.setGeometry(14, 12, max(100, width - 28), 52)
        if not self.error_label.isHidden():
            toast_width = min(520, width - 48)
            self.error_label.setGeometry(
                (width - toast_width) // 2,
                68,
                toast_width,
                max(40, self.error_label.sizeHint().height()),
            )
        target_width = min(292, max(230, width // 4))
        target_height = min(
            320 if self.advanced_toggle.isChecked() else 154,
            max(130, self.target_card.sizeHint().height()),
        )
        self.target_card.setGeometry(
            max(16, width - target_width - 22),
            78,
            target_width,
            target_height,
        )
        self.command_button.setGeometry(22, max(16, height - 52), 164, 34)
        if self.command_palette.isVisible():
            palette_width = 320
            palette_height = min(self.command_palette.sizeHint().height(), height - 112)
            self.command_palette.setGeometry(
                22,
                max(64, height - palette_height - 62),
                palette_width,
                palette_height,
            )
        self.gesture_feedback.setGeometry(
            max(198, width // 2 - 210),
            max(16, height - 48),
            min(420, width - 220),
            28,
        )
        simulation_visible = self._simulation_pinned or (
            self._snapshot is not None
            and (
                self._snapshot.interception_active
                or self._snapshot.simulation.status.value != "IDLE"
            )
        )
        self.simulation_card.setVisible(simulation_visible)
        if simulation_visible:
            simulation_width = min(376, max(300, width // 3))
            self.simulation_card.setGeometry(
                max(16, width - simulation_width - 22),
                max(86, height - 246),
                simulation_width,
                220,
            )

    def _refresh(self) -> None:
        captured = self._capture.latest
        if captured.sequence != self._last_frame_sequence and captured.frame is not None:
            self.canvas.set_image(
                self._to_qimage(captured.frame),
                (captured.frame.shape[1], captured.frame.shape[0]),
            )
            self._last_frame_sequence = captured.sequence

        worker_state = self._vision.latest
        snapshot = worker_state.snapshot
        overlay_age = (
            max(0.0, time.perf_counter() - snapshot.timestamp)
            if snapshot is not None
            else None
        )
        self.canvas.set_snapshot(
            snapshot,
            overlay_age,
            self._show_history,
            self._show_predictions,
        )
        self._update_status(captured, worker_state, snapshot)
        if snapshot is not None:
            self._update_target(snapshot)
            self.simulation_canvas.set_snapshot(snapshot.simulation)
            self._append_events(snapshot)
        self._snapshot = snapshot
        self._position_overlays()

    @staticmethod
    def _to_qimage(frame: np.ndarray) -> QImage:
        height, width = frame.shape[:2]
        return QImage(
            frame.data,
            width,
            height,
            int(frame.strides[0]),
            QImage.Format.Format_BGR888,
        ).copy()

    def _update_status(
        self,
        captured: CapturedFrame,
        worker: VisionWorkerState,
        snapshot: PipelineSnapshot | None,
    ) -> None:
        self.camera_status.setText(
            f"{source_label(captured.source)}  ·  {captured.status.upper()}"
        )
        self.initialization_label.setText(worker.initialization_state)
        self.error_label.setText(worker.error or captured.error or "")
        self.error_label.setVisible(bool(worker.error or captured.error))
        state = snapshot.system_state if snapshot else worker.initialization_state.upper()
        self.system_indicator.setStyleSheet(
            "color: #ff887d;" if worker.error or captured.error
            else "color: #7deeb3;" if state in ("READY", "TRACKING", "PREDICTING")
            else "color: #73e5fb;"
        )
        self.buttons["camera"].setText(
            "Stop camera"
            if isinstance(captured.source, int) and captured.running
            else "Start camera"
        )
        if snapshot is not None:
            self.initialization_label.setText(state)
            self.buttons["pause"].setText("Resume" if snapshot.tracking_paused else "Pause")
            self.buttons["detection"].setText(
                "Detection on" if snapshot.detection_enabled else "Detection off"
            )
            self.buttons["gestures"].setText(
                "Gestures on" if snapshot.gestures_enabled else "Gestures off"
            )
            self.buttons["classification"].setText(
                "Classification on"
                if snapshot.classification_enabled
                else "Classification off"
            )
            self.buttons["lock"].setText(
                "Unlock target" if snapshot.target_locked else "Lock target"
            )
            self.buttons["interception"].setText(
                "Stop intercept" if snapshot.interception_active else "Start intercept"
            )
            self.buttons["anime"].setText(
                "Anime gestures on"
                if snapshot.anime_gestures_enabled
                else "Anime gestures off"
            )
            self.buttons["effects"].setText(
                "Anime effects on"
                if snapshot.anime_effects_enabled
                else "Anime effects off"
            )
            self.buttons["abilities"].setText(
                "Abilities on" if snapshot.abilities_enabled else "Abilities off"
            )
            ability_samples = snapshot.ability_samples or {}
            sample_count = ability_samples.get(self._selected_ability_id(), 0)
            movement_samples = snapshot.ability_movement_samples or {}
            movement_count = movement_samples.get(self._selected_ability_id(), 0)
            self.ability_sample_status.setText(
                f"{sample_count} pose / {movement_count} movement samples"
                + (
                    f"  ·  RECORDING {snapshot.ability_calibration_frames}"
                    if snapshot.ability_calibration_recording
                    else ""
                )
            )
            self.record_ability_movement.setText(
                "Stop movement" if snapshot.ability_calibration_recording else "Record movement"
            )
            if abs(self.ability_similarity.value() - snapshot.ability_similarity_threshold) > 0.005:
                self.ability_similarity.blockSignals(True)
                self.ability_similarity.setValue(snapshot.ability_similarity_threshold)
                self.ability_similarity.blockSignals(False)

    def _update_target(self, snapshot: PipelineSnapshot) -> None:
        track = snapshot.selected_track
        estimate = snapshot.selected_estimate
        if track is None:
            self.target_card.hide()
        else:
            self.target_card.show()
            self.target_name.setText(
                f"{track.label.upper()}  /  ID {track.track_id}"
            )
            if snapshot.target_locked:
                self.target_name.setText(
                    f"LOCKED  /  {track.label.upper()}  /  ID {track.track_id}"
                )
            if estimate is None:
                self.target_summary.setText(
                    f"{track.center[0]}, {track.center[1]} px  ·  {track.confidence:.0%}\n"
                    "Motion estimate initializing"
                )
                motion_details = "Collecting motion measurements."
            else:
                self.target_summary.setText(
                    f"{estimate.speed:.0f} px/s  ·  {estimate.status.value}\n"
                    f"{estimate.measured_position[0]:.0f}, "
                    f"{estimate.measured_position[1]:.0f} px  ·  "
                    f"{track.confidence:.0%} confidence"
                )
                motion_details = (
                    f"Filtered: {estimate.filtered_position[0]:.0f}, "
                    f"{estimate.filtered_position[1]:.0f} px\n"
                    f"Velocity: {estimate.velocity[0]:.0f}, "
                    f"{estimate.velocity[1]:.0f} px/s\n"
                    f"Direction: {estimate.direction_degrees:.0f}°"
                    if estimate.direction_degrees is not None
                    else f"Filtered: {estimate.filtered_position[0]:.0f}, "
                    f"{estimate.filtered_position[1]:.0f} px\nStationary"
                )
                motion_details += "\nPredictions: " + "  ·  ".join(
                    f"+{horizon:g}s {position[0]:.0f},{position[1]:.0f}"
                    for horizon, position in sorted(estimate.predictions.items())
                )
            recognition = snapshot.recognition.get(track.track_id)
            recognition_details = (
                f"\nRecognition: {recognition.status.value} "
                f"· {recognition.refined_label or recognition.detector_label}"
                if recognition is not None
                else "\nRecognition: detected"
            )
            self.prediction_values.setText(motion_details + recognition_details)

        visual_hands = snapshot.hand.hands
        if not visual_hands and snapshot.hand.detected:
            visual_hands = (
                DetectedHand(
                    snapshot.hand.landmarks,
                    snapshot.hand.handedness,
                    snapshot.hand.gesture,
                    snapshot.hand.confidence,
                ),
            )
        command = (
            snapshot.gesture_update.events[-1].command.value.upper()
            if snapshot.gesture_update.events
            else ""
        )
        anime_pose = snapshot.anime.pose
        ability = snapshot.ability_recognition
        if ability.selected_ability_id is not None:
            definition = next(
                (
                    item for item in self.ability_definitions
                    if item.ability_id == ability.selected_ability_id
                ),
                None,
            )
            message = (
                f"{definition.display_name if definition else ability.selected_ability_id.upper()}"
                f"  ·  {ability.phase}"
            )
            if ability.sequence_progress:
                progress = ability.sequence_progress[0]
                message += f"  ·  {progress.text}"
            self.gesture_feedback.setText(message)
            self.gesture_feedback.show()
        elif snapshot.ability_effects:
            active_ability = snapshot.ability_effects[-1].ability_id.replace("_", " ").upper()
            self.gesture_feedback.setText(f"ABILITY EFFECT  ·  {active_ability}")
            self.gesture_feedback.show()
        elif anime_pose is not None and anime_pose.status in (
            "RECOGNIZING",
            "RECOGNIZED",
            "HELD",
        ):
            message = (
                f"{anime_pose.display_name}  ·  {anime_pose.status}"
                f"  ·  {anime_pose.held_seconds:.1f}s"
            )
            if snapshot.anime.sequence_progress:
                message += f"  ·  {snapshot.anime.sequence_progress[0].text}"
            self.gesture_feedback.setText(message)
            self.gesture_feedback.show()
        elif snapshot.effects:
            effect_name = snapshot.effects[-1].effect_type.value.replace("_", " ").upper()
            self.gesture_feedback.setText(f"ABILITY ACTIVE  ·  {effect_name}")
            self.gesture_feedback.show()
        elif visual_hands and snapshot.gestures_enabled:
            hand_names = " + ".join(
                (hand.handedness or f"HAND {index + 1}").upper()
                for index, hand in enumerate(visual_hands)
            )
            self.gesture_feedback.setText(
                f"{hand_names}  ·  {snapshot.hand.gesture.value.upper()}"
                f"  ·  {snapshot.gesture_update.stability:.0%} STABLE"
                + (f"  ·  {command}" if command else "")
            )
            self.gesture_feedback.show()
        else:
            self.gesture_feedback.hide()

    def _append_events(self, snapshot: PipelineSnapshot) -> None:
        for event in snapshot.events:
            key = (event.timestamp, event.message)
            if key not in self._last_events:
                self._last_events.add(key)
                colors = {
                    "ERROR": "#ff8a80",
                    "WARNING": "#e7b45b",
                    "SUCCESS": "#83d69a",
                    "INFO": "#9caeb7",
                }
                formatting = QTextCharFormat()
                formatting.setForeground(QColor(colors.get(event.severity.value, "#9caeb7")))
                cursor = self.event_log.textCursor()
                cursor.movePosition(QTextCursor.MoveOperation.End)
                cursor.insertText(
                    f"{event.timestamp}  {event.severity.value:<7}  {event.message}\n",
                    formatting,
                )
                self.event_log.setTextCursor(cursor)
        if len(self._last_events) > config.GUI_EVENT_LOG_LIMIT * 2:
            self._last_events = {
                (event.timestamp, event.message) for event in snapshot.events
            }

    def closeEvent(self, event: object) -> None:
        self._timer.stop()
        capture_stopped = self._capture.stop()
        vision_stopped = self._vision.stop()
        if not capture_stopped or not vision_stopped:
            self.statusBar().showMessage(
                "A worker did not stop before the shutdown timeout; closing safely."
            )
        if hasattr(event, "accept"):
            event.accept()

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)
        self._position_overlays()


def run_desktop_app(
    source: int | str,
    model_path: str,
    confidence: float,
    device: str | None,
    tracker_config: str,
    inference_size: int,
    launch_started: float | None = None,
    detector_type: str = config.DETECTOR_TYPE,
    vocabulary: tuple[str, ...] = config.OPEN_VOCABULARY_CLASSES,
    classification_enabled: bool = config.CLASSIFICATION_ENABLED,
) -> int:
    launch_started = launch_started or time.perf_counter()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("JARVIS Vision Workspace")
    app.setStyle("Fusion")
    window = JarvisWindow(
        source,
        model_path,
        confidence,
        device,
        tracker_config,
        inference_size,
        detector_type,
        vocabulary,
        classification_enabled,
    )
    window.show()
    app.processEvents()
    window.time_to_visible_ms = (time.perf_counter() - launch_started) * 1_000.0
    window.start_workers()
    return app.exec()
