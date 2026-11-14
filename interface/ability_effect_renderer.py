"""Procedural QPainter effects for abilities, anchored in camera coordinates."""

from __future__ import annotations

from math import cos, hypot, pi, sin

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
    QImage,
)

from interface.ability_effects import AbilityEffectSnapshot
import config


class AbilityEffectRenderer:
    """Draw bounded original effects without per-frame image allocations."""

    @staticmethod
    def draw(
        painter: QPainter,
        image_rect: QRectF,
        effects: tuple[AbilityEffectSnapshot, ...],
        timestamp: float,
    ) -> None:
        if image_rect.isEmpty():
            return
        painter.save()
        painter.setClipRect(image_rect)
        for effect in effects:
            center = QPointF(
                image_rect.left() + effect.position[0] * image_rect.width(),
                image_rect.top() + effect.position[1] * image_rect.height(),
            )
            progress = min(1.0, effect.age_seconds / max(effect.duration_seconds, 1e-6))
            fade = max(0.0, 1.0 - progress)
            painter.save()
            opacity = effect.intensity * (
                1.0 if effect.phase == "charging" or effect.effect_type == "domain"
                else max(0.0, 0.3 + 0.7 * fade)
            )
            painter.setOpacity(max(0.0, min(1.0, opacity)))
            if effect.phase == "charging":
                if effect.effect_type == "beam":
                    AbilityEffectRenderer._kamehameha_charge(
                        painter,
                        center,
                        image_rect,
                        effect,
                        timestamp,
                    )
                else:
                    AbilityEffectRenderer._charge(painter, center, image_rect, effect, timestamp)
            elif effect.effect_type == "beam":
                AbilityEffectRenderer._beam(painter, image_rect, effect)
            elif effect.effect_type == "sphere":
                AbilityEffectRenderer._sphere(painter, center, image_rect, progress, timestamp)
            elif effect.effect_type == "lightning":
                AbilityEffectRenderer._lightning(painter, center, image_rect, effect, timestamp)
            elif effect.effect_type == "domain":
                AbilityEffectRenderer._domain(
                    painter,
                    center,
                    image_rect,
                    effect,
                    timestamp,
                )
            elif effect.effect_type == "afterimage":
                AbilityEffectRenderer._afterimages(painter, center, image_rect, effect, progress)
            painter.restore()
        painter.restore()

    @staticmethod
    def draw_clones(
        painter: QPainter,
        image_rect: QRectF,
        effects: tuple[AbilityEffectSnapshot, ...],
        timestamp: float,
        images: dict[tuple[int, int], QImage],
    ) -> None:
        for effect in effects:
            if effect.effect_type != "shadow_clones":
                continue
            age = effect.age_seconds
            duration = effect.duration_seconds
            fade_in = min(1.0, age / config.ABILITY_SHADOW_CLONE_FADE_IN_SECONDS)
            fade_out = min(
                1.0,
                max(0.0, duration - age)
                / config.ABILITY_SHADOW_CLONE_FADE_OUT_SECONDS,
            )
            for sprite in effect.clone_sprites:
                image = images.get((sprite.effect_id, sprite.clone_index))
                if image is None:
                    continue
                left, top, width, height = sprite.bounds
                sway = sin(
                    timestamp * config.ABILITY_SHADOW_CLONE_SWAY_HZ
                    + sprite.clone_index * 1.9
                ) * config.ABILITY_SHADOW_CLONE_SWAY_RATIO
                scale = 1.0 + sin(
                    timestamp * config.ABILITY_SHADOW_CLONE_PULSE_HZ
                    + sprite.clone_index
                ) * config.ABILITY_SHADOW_CLONE_SCALE_VARIATION
                center_x = left + width / 2.0 + sprite.spacing_ratio + sway
                center_y = top + height / 2.0
                draw_width = width * scale
                draw_height = height * scale
                target = QRectF(
                    image_rect.left() + (center_x - draw_width / 2.0) * image_rect.width(),
                    image_rect.top() + (center_y - draw_height / 2.0) * image_rect.height(),
                    draw_width * image_rect.width(),
                    draw_height * image_rect.height(),
                )
                painter.save()
                painter.setOpacity(sprite.opacity * fade_in * fade_out)
                painter.drawImage(target, image)
                painter.restore()
            AbilityEffectRenderer._clone_smoke(
                painter,
                image_rect,
                effect,
                timestamp,
                fade_in,
                fade_out,
            )

    @staticmethod
    def _clone_smoke(
        painter: QPainter,
        rect: QRectF,
        effect: AbilityEffectSnapshot,
        timestamp: float,
        fade_in: float,
        fade_out: float,
    ) -> None:
        age = effect.age_seconds
        closing = effect.duration_seconds - age
        strength = max(
            max(0.0, 1.0 - age / config.ABILITY_SHADOW_CLONE_FADE_IN_SECONDS),
            max(
                0.0,
                1.0 - closing / config.ABILITY_SHADOW_CLONE_FADE_OUT_SECONDS,
            ),
        )
        if strength <= 0.0:
            return
        center = QPointF(
            rect.left() + effect.position[0] * rect.width(),
            rect.top() + effect.position[1] * rect.height(),
        )
        painter.save()
        painter.setOpacity(min(0.75, strength * 0.72) * max(fade_in, fade_out))
        for index in range(7):
            phase = timestamp * 0.8 + index * 2.4 + effect.seed
            offset = QPointF(
                sin(phase) * rect.width() * (0.025 + index * 0.004),
                cos(phase * 0.73) * rect.height() * 0.035,
            )
            cloud = center + offset
            radius = min(rect.width(), rect.height()) * (0.07 + (index % 3) * 0.035)
            color = QColor("#c6d9e8")
            color.setAlpha(115)
            AbilityEffectRenderer._glow(painter, cloud, radius, color)
        painter.restore()

    @staticmethod
    def _glow(painter: QPainter, center: QPointF, radius: float, color: QColor) -> None:
        gradient = QRadialGradient(center, max(1.0, radius))
        inner = QColor(color)
        inner.setAlpha(150)
        outer = QColor(color)
        outer.setAlpha(0)
        gradient.setColorAt(0.0, inner)
        gradient.setColorAt(1.0, outer)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawEllipse(center, radius, radius)

    @staticmethod
    def _charge(
        painter: QPainter,
        center: QPointF,
        rect: QRectF,
        effect: AbilityEffectSnapshot,
        timestamp: float,
    ) -> None:
        radius = min(rect.width(), rect.height()) * (
            0.018 + min(0.04, effect.age_seconds * 0.006)
        )
        color = QColor("#a474ff" if effect.effect_type == "domain" else "#52dfff")
        AbilityEffectRenderer._glow(painter, center, radius, color)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        phase = timestamp * 2.8 + effect.seed
        for ring in range(2):
            ring_radius = radius * (1.15 + 0.2 * sin(phase + ring))
            painter.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 150), 1.5))
            painter.drawEllipse(center, ring_radius, ring_radius)
        for index in range(config.ANIME_EFFECT_PARTICLE_COUNT):
            angle = phase + 2.0 * pi * index / config.ANIME_EFFECT_PARTICLE_COUNT
            point = center + QPointF(cos(angle) * radius * 1.4, sin(angle) * radius * 1.4)
            painter.setBrush(QColor("#eafcff"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(point, 1.5, 1.5)

    @staticmethod
    def _kamehameha_charge(
        painter: QPainter,
        center: QPointF,
        rect: QRectF,
        effect: AbilityEffectSnapshot,
        timestamp: float,
    ) -> None:
        charge = min(1.0, effect.age_seconds / config.ABILITY_KAMEHAMEHA_CHARGE_SECONDS)
        scale = min(rect.width(), rect.height())
        radius = scale * (0.025 + 0.055 * charge)
        phase = timestamp * (2.8 + charge * 4.0) + effect.seed
        AbilityEffectRenderer._glow(painter, center, radius * 2.4, QColor("#269cff"))
        AbilityEffectRenderer._glow(painter, center, radius * 1.25, QColor("#b9f4ff"))
        sphere = QRadialGradient(center, radius)
        sphere.setColorAt(0.0, QColor(255, 255, 255, 250))
        sphere.setColorAt(0.35, QColor(108, 222, 255, 230))
        sphere.setColorAt(1.0, QColor(26, 119, 255, 60))
        painter.setBrush(sphere)
        painter.setPen(QPen(QColor(220, 250, 255, 210), max(1.2, radius * 0.035)))
        painter.drawEllipse(center, radius, radius)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for ring in range(3):
            rotation = phase * (1.0 if ring % 2 == 0 else -0.72) + ring * 2.1
            orbit = radius * (0.53 + ring * 0.12)
            path = QPainterPath()
            for step in range(30):
                angle = rotation + step * 0.11
                point = center + QPointF(cos(angle) * orbit, sin(angle) * orbit * 0.72)
                if step == 0:
                    path.moveTo(point)
                else:
                    path.lineTo(point)
            painter.setPen(QPen(QColor(222, 251, 255, 210 - ring * 38), 1.3 + charge))
            painter.drawPath(path)
        for arc in range(5):
            angle = phase * 0.7 + arc * 2.0 * pi / 5.0
            path = QPainterPath(
                center + QPointF(cos(angle) * radius * 1.1, sin(angle) * radius * 1.1)
            )
            for segment in range(1, 6):
                distance = radius * (1.1 + segment * 0.24)
                jitter = sin(phase + arc * 13 + segment * 8) * radius * 0.16
                path.lineTo(
                    center
                    + QPointF(
                        cos(angle) * distance - sin(angle) * jitter,
                        sin(angle) * distance + cos(angle) * jitter,
                    )
                )
            painter.setPen(QPen(QColor(170, 236, 255, 190), 1.2))
            painter.drawPath(path)
        for index in range(config.ANIME_EFFECT_PARTICLE_COUNT):
            angle = phase + 2.0 * pi * index / config.ANIME_EFFECT_PARTICLE_COUNT
            inward = (timestamp * 0.72 + index * 0.17) % 1.0
            travel = radius * (2.1 + ((index * 7) % 5) * 0.38) * (
                1.0 - 0.52 * inward
            )
            point = center + QPointF(cos(angle) * travel, sin(angle) * travel)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(220, 248, 255, 170))
            painter.drawEllipse(point, 1.1 + charge, 1.1 + charge)
            painter.setPen(QPen(QColor(158, 229, 255, 92), 0.8))
            painter.drawLine(
                point,
                center + QPointF(cos(angle) * radius * 0.9, sin(angle) * radius * 0.9),
            )

    @staticmethod
    def _beam(
        painter: QPainter,
        rect: QRectF,
        effect: AbilityEffectSnapshot,
    ) -> None:
        origin_ratio = effect.origin or effect.position
        origin = QPointF(
            rect.left() + origin_ratio[0] * rect.width(),
            rect.top() + origin_ratio[1] * rect.height(),
        )
        dx = effect.direction[0] * rect.width()
        dy = effect.direction[1] * rect.height()
        magnitude = max(1e-8, hypot(dx, dy))
        dx, dy = dx / magnitude, dy / magnitude
        edge_distance = min(
            rect.width() * (1.0 - origin_ratio[0]) / dx if dx > 0 else rect.width() * origin_ratio[0] / -dx if dx < 0 else float("inf"),
            rect.height() * (1.0 - origin_ratio[1]) / dy if dy > 0 else rect.height() * origin_ratio[1] / -dy if dy < 0 else float("inf"),
        )
        distance = min(edge_distance, magnitude * 1.1)
        travel_progress = min(
            1.0,
            effect.age_seconds * config.ABILITY_KAMEHAMEHA_PROJECTILE_SPEED_RATIO,
        )
        end = origin + QPointF(dx * distance * travel_progress, dy * distance * travel_progress)
        half_width = min(rect.width(), rect.height()) * (0.018 + 0.042 * travel_progress)
        perpendicular = QPointF(-dy * half_width, dx * half_width)
        for width_scale, color_start, color_end in (
            (1.0, QColor(49, 142, 255, 24), QColor(67, 151, 255, 0)),
            (0.43, QColor(77, 196, 255, 100), QColor(93, 185, 255, 0)),
            (0.13, QColor(242, 255, 255, 240), QColor(163, 233, 255, 0)),
        ):
            width = perpendicular * width_scale
            path = QPainterPath(origin + width)
            path.lineTo(origin - width)
            path.lineTo(end - width * 0.12)
            path.lineTo(end + width * 0.12)
            path.closeSubpath()
            gradient = QLinearGradient(origin, end)
            gradient.setColorAt(0.0, color_start)
            gradient.setColorAt(1.0, color_end)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(gradient)
            painter.drawPath(path)
        for streak in range(5):
            drift = sin(effect.age_seconds * 24.0 + effect.seed + streak * 2.3) * half_width * 0.32
            start = origin + perpendicular * (drift / max(half_width, 1e-6))
            painter.setPen(
                QPen(
                    QColor(185, 240, 255, 120 - streak * 12),
                    max(0.8, half_width * 0.025),
                )
            )
            painter.drawLine(start, end)
        painter.setPen(QPen(QColor(244, 255, 255, 245), max(2.0, half_width * 0.12)))
        painter.drawLine(origin, end)
        AbilityEffectRenderer._glow(painter, origin, half_width * 1.8, QColor("#4bcfff"))
        if effect.age_seconds > 1.0 / config.ABILITY_KAMEHAMEHA_PROJECTILE_SPEED_RATIO:
            pulse = 0.65 + 0.35 * sin(effect.age_seconds * 18.0 + effect.seed)
            AbilityEffectRenderer._glow(
                painter,
                origin + QPointF(dx * distance, dy * distance),
                min(rect.width(), rect.height()) * 0.07 * pulse,
                QColor("#a8edff"),
            )

    @staticmethod
    def _sphere(
        painter: QPainter,
        center: QPointF,
        rect: QRectF,
        progress: float,
        timestamp: float,
    ) -> None:
        radius = min(rect.width(), rect.height()) * (0.035 + 0.10 * progress)
        AbilityEffectRenderer._glow(painter, center, radius * 1.7, QColor("#53aaff"))
        painter.setBrush(QColor(121, 206, 255, 135))
        painter.setPen(QPen(QColor("#dffaff"), 2.0))
        painter.drawEllipse(center, radius, radius)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(220, 248, 255, 190), 1.3))
        for ray in range(12):
            angle = timestamp * 0.8 + 2 * pi * ray / 12
            inner = center + QPointF(cos(angle) * radius * 1.15, sin(angle) * radius * 1.15)
            outer = center + QPointF(cos(angle) * radius * 1.45, sin(angle) * radius * 1.45)
            painter.drawLine(inner, outer)

    @staticmethod
    def _lightning(
        painter: QPainter,
        center: QPointF,
        rect: QRectF,
        effect: AbilityEffectSnapshot,
        timestamp: float,
    ) -> None:
        radius = min(rect.width(), rect.height()) * 0.075
        AbilityEffectRenderer._glow(painter, center, radius, QColor("#53dcff"))
        phase = int(timestamp * 22.0) + effect.seed
        painter.setPen(QPen(QColor("#e3fbff"), 2.2))
        for arc in range(8):
            angle = arc * pi / 4.0
            path = QPainterPath(center)
            for segment in range(1, 5):
                length = radius * segment / 4.0
                jitter = sin(phase + arc * 11 + segment * 7) * radius * 0.22
                point = center + QPointF(
                    cos(angle) * length - sin(angle) * jitter,
                    sin(angle) * length + cos(angle) * jitter,
                )
                path.lineTo(point)
            painter.drawPath(path)
        AbilityEffectRenderer._beam(painter, rect, effect)

    @staticmethod
    def _domain(
        painter: QPainter,
        center: QPointF,
        rect: QRectF,
        effect: AbilityEffectSnapshot,
        timestamp: float,
    ) -> None:
        expansion = min(
            1.0,
            effect.age_seconds / max(config.ABILITY_DOMAIN_EXPANSION_SECONDS, 0.1),
        )
        closing = min(
            1.0,
            max(0.0, effect.duration_seconds - effect.age_seconds)
            / max(config.ABILITY_DOMAIN_COLLAPSE_SECONDS, 0.1),
        )
        radius_progress = min(expansion, closing)
        hold = max(0.0, min(1.0, expansion * 5.0, closing * 5.0))
        painter.fillRect(rect, QColor(8, 8, 34, int(150 * hold)))
        gradient = QRadialGradient(center, max(rect.width(), rect.height()) * 0.82)
        gradient.setColorAt(0.0, QColor(47, 42, 105, int(88 * hold)))
        gradient.setColorAt(0.48, QColor(19, 27, 74, int(94 * hold)))
        gradient.setColorAt(1.0, QColor(4, 9, 31, int(170 * hold)))
        painter.fillRect(rect, gradient)

        boundary = max(rect.width(), rect.height()) * 0.92 * radius_progress
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for layer in range(4):
            shrink = 1.0 - layer * 0.17
            alpha = int((176 - layer * 34) * max(0.0, 1.0 - abs(radius_progress - 0.75) * 0.16))
            path = QPainterPath()
            horizon_y = rect.top() + rect.height() * (0.18 + layer * 0.12)
            path.moveTo(center.x() - boundary * shrink, horizon_y)
            path.cubicTo(
                center.x() - boundary * 0.55 * shrink,
                center.y() - boundary * 0.88 * shrink,
                center.x() + boundary * 0.55 * shrink,
                center.y() - boundary * 0.88 * shrink,
                center.x() + boundary * shrink,
                horizon_y,
            )
            painter.setPen(QPen(QColor(116 + layer * 22, 126 + layer * 14, 255, alpha), 1.2 + layer * 0.55))
            painter.drawPath(path)
        # A vanishing-point lattice and orbiting void sigils give the field depth.
        vanish = QPointF(center.x(), rect.top() + rect.height() * 0.36)
        painter.setPen(QPen(QColor(155, 175, 255, int(68 * hold)), 0.8))
        for index in range(25):
            x = rect.left() + rect.width() * index / 24.0
            painter.drawLine(vanish, QPointF(x, rect.bottom()))
        for row in range(1, 9):
            t = row / 9.0
            y = vanish.y() + (rect.bottom() - vanish.y()) * t * t
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        for index in range(9):
            angle = timestamp * (0.12 + index * 0.003) + index * 2.4 + effect.seed
            orbit = 0.12 + (index % 4) * 0.075
            point = center + QPointF(
                cos(angle) * rect.width() * orbit,
                sin(angle * 0.83) * rect.height() * orbit,
            )
            painter.setPen(QPen(QColor(189, 205, 255, int(122 * hold)), 1.0))
            painter.setBrush(QColor(63, 85, 180, int(70 * hold)))
            painter.drawEllipse(point, 5 + index % 4, 8 + index % 5)
        for index in range(44):
            angle = index * 2.399963 + effect.seed
            radius = min(rect.width(), rect.height()) * (
                0.08 + ((index * 17) % 89) / 100.0
            )
            drift = timestamp * (0.018 + (index % 7) * 0.003)
            point = center + QPointF(
                cos(angle + drift) * radius,
                sin(angle * 1.21 - drift) * radius * 0.72,
            )
            alpha = int((60 + index % 6 * 24) * hold)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(170, 202, 255, alpha))
            painter.drawEllipse(point, 1.2 + index % 3 * 0.4, 1.2 + index % 3 * 0.4)
        flash = max(0.0, 1.0 - effect.age_seconds / 0.22)
        if flash > 0.0:
            painter.fillRect(rect, QColor(225, 233, 255, int(210 * flash)))
        AbilityEffectRenderer._glow(
            painter,
            center,
            min(rect.width(), rect.height()) * (0.10 + 0.28 * expansion),
            QColor("#8b9cff"),
        )

    @staticmethod
    def _afterimages(
        painter: QPainter,
        center: QPointF,
        rect: QRectF,
        effect: AbilityEffectSnapshot,
        progress: float,
    ) -> None:
        scale = min(rect.width(), rect.height())
        points = effect.hand_trail[-8:]
        for index, position in enumerate(points[:-1]):
            point = QPointF(rect.left() + position[0] * rect.width(), rect.top() + position[1] * rect.height())
            alpha = int(125 * (index + 1) / max(1, len(points)) * (1.0 - progress))
            painter.setPen(QPen(QColor(74, 222, 255, alpha), max(1.0, 3.5 - index * 0.25)))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(point, scale * 0.028, scale * 0.045)
        AbilityEffectRenderer._glow(painter, center, scale * 0.12, QColor("#45dbff"))
