"""Small procedural QPainter effects for the live camera canvas."""

from __future__ import annotations

from math import cos, pi, sin

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QRadialGradient, QPen

import config
from interface.effects import EffectSnapshot, EffectType


class EffectRenderer:
    """Draw deterministic, bounded vector effects without per-frame image buffers."""

    @staticmethod
    def draw(
        painter: QPainter,
        image_rect: QRectF,
        effects: tuple[EffectSnapshot, ...],
        timestamp: float,
    ) -> None:
        if image_rect.isEmpty():
            return
        painter.save()
        painter.setClipRect(image_rect)
        for effect in effects:
            x = image_rect.left() + effect.position[0] * image_rect.width()
            y = image_rect.top() + effect.position[1] * image_rect.height()
            age_ratio = min(1.0, effect.age_seconds / max(effect.duration_seconds, 1e-6))
            fade = max(0.0, 1.0 - age_ratio)
            pulse = 0.88 + 0.12 * sin(timestamp * 9.0 + effect.seed)
            radius = min(image_rect.width(), image_rect.height()) * (
                0.025 + effect.intensity * 0.035
            )
            center = QPointF(x, y)
            painter.save()
            painter.setOpacity(fade * effect.intensity)
            if effect.effect_type == EffectType.ENERGY_CHARGE:
                EffectRenderer._charge(painter, center, radius, pulse, age_ratio)
            elif effect.effect_type == EffectType.AURA:
                EffectRenderer._aura(painter, center, radius, pulse, timestamp, effect.seed)
            elif effect.effect_type == EffectType.LIGHTNING:
                EffectRenderer._lightning(painter, center, radius, timestamp, effect.seed)
            elif effect.effect_type == EffectType.FIREBALL:
                EffectRenderer._fireball(painter, center, radius, pulse, effect.seed)
            elif effect.effect_type == EffectType.ENERGY_BURST:
                EffectRenderer._burst(painter, center, radius, age_ratio, effect.seed)
            painter.restore()
        painter.restore()

    @staticmethod
    def _charge(
        painter: QPainter,
        center: QPointF,
        radius: float,
        pulse: float,
        age_ratio: float,
    ) -> None:
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for ring in range(3):
            phase = (age_ratio * 2.0 + ring / 3.0) % 1.0
            ring_radius = radius * (0.65 + phase * 0.85)
            painter.setPen(QPen(QColor(102, 233, 255, int(145 * (1.0 - phase))), 1.3))
            painter.drawEllipse(center, ring_radius, ring_radius)
        EffectRenderer._glow(painter, center, radius * pulse, QColor("#5cecff"))
        EffectRenderer._sparks(
            painter,
            center,
            radius * 1.35,
            config.ANIME_EFFECT_PARTICLE_COUNT,
            0.72,
        )

    @staticmethod
    def _aura(
        painter: QPainter,
        center: QPointF,
        radius: float,
        pulse: float,
        timestamp: float,
        seed: int,
    ) -> None:
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for arc in range(3):
            start = (timestamp * 45.0 + seed + arc * 120.0) % 360.0
            painter.setPen(QPen(QColor(89, 221, 255, 95), 1.6))
            painter.drawArc(
                QRectF(center.x() - radius * pulse, center.y() - radius * pulse,
                       radius * 2 * pulse, radius * 2 * pulse),
                int(start * 16),
                100 * 16,
            )
        EffectRenderer._glow(painter, center, radius * 0.7, QColor("#57b8ff"))

    @staticmethod
    def _lightning(
        painter: QPainter,
        center: QPointF,
        radius: float,
        timestamp: float,
        seed: int,
    ) -> None:
        phase = int(timestamp * 24.0) % 3
        painter.setPen(QPen(QColor(116, 239, 255, 230), 2.0))
        for arc in range(4):
            angle = (arc * pi / 2.0) + phase * 0.08
            direction = QPointF(cos(angle), sin(angle))
            path = QPainterPath(center)
            previous = center
            segments = 5
            for segment in range(1, segments + 1):
                length = radius * segment / segments
                jitter = sin(seed + arc * 13 + segment * 7 + phase) * radius * 0.18
                current = center + QPointF(
                    direction.x() * length - direction.y() * jitter,
                    direction.y() * length + direction.x() * jitter,
                )
                path.lineTo(current)
                previous = current
            del previous
            painter.drawPath(path)
        EffectRenderer._glow(painter, center, radius * 0.42, QColor("#d8fcff"))

    @staticmethod
    def _fireball(
        painter: QPainter,
        center: QPointF,
        radius: float,
        pulse: float,
        seed: int,
    ) -> None:
        EffectRenderer._sparks(
            painter,
            center,
            radius * 1.85,
            config.ANIME_EFFECT_PARTICLE_COUNT,
            0.8,
            seed,
        )
        EffectRenderer._glow(painter, center, radius * pulse, QColor("#ff9d48"))
        painter.setPen(QPen(QColor("#ffe6a0"), 1.2))
        painter.setBrush(QColor(255, 177, 80, 175))
        painter.drawEllipse(center, radius * 0.42, radius * 0.42)

    @staticmethod
    def _burst(
        painter: QPainter,
        center: QPointF,
        radius: float,
        age_ratio: float,
        seed: int,
    ) -> None:
        expansion = 0.25 + age_ratio * 2.4
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for ring, alpha in ((1.0, 180), (0.68, 105)):
            ring_radius = radius * expansion * ring
            painter.setPen(QPen(QColor(114, 235, 255, int(alpha * (1.0 - age_ratio))), 1.8))
            painter.drawEllipse(center, ring_radius, ring_radius)
        EffectRenderer._sparks(
            painter,
            center,
            radius * expansion * 1.15,
            config.ANIME_EFFECT_PARTICLE_COUNT,
            1.0 - age_ratio,
            seed,
        )
        EffectRenderer._glow(
            painter,
            center,
            radius * max(0.1, 0.55 * (1.0 - age_ratio)),
            QColor("#c8faff"),
        )

    @staticmethod
    def _glow(painter: QPainter, center: QPointF, radius: float, color: QColor) -> None:
        gradient = QRadialGradient(center, max(radius, 1.0))
        inner = QColor(color)
        inner.setAlpha(115)
        outer = QColor(color)
        outer.setAlpha(0)
        gradient.setColorAt(0.0, inner)
        gradient.setColorAt(1.0, outer)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawEllipse(center, radius, radius)

    @staticmethod
    def _sparks(
        painter: QPainter,
        center: QPointF,
        radius: float,
        count: int,
        intensity: float,
        seed: int = 0,
    ) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(168, 244, 255, max(0, min(220, int(180 * intensity)))))
        for index in range(count):
            angle = (2 * pi * index / count) + (seed % 31) * 0.01
            distance = radius * (0.62 + 0.28 * sin(index * 17 + seed))
            point = center + QPointF(cos(angle) * distance, sin(angle) * distance)
            painter.drawEllipse(point, 1.2 + (index % 2) * 0.5, 1.2 + (index % 2) * 0.5)
