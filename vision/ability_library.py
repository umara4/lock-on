"""Validated, data-driven descriptions for camera-space abilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AbilityDefinition:
    ability_id: str
    display_name: str
    franchise: str
    description: str
    hand_configuration: str
    movement: str
    pose: str
    movement_gate: str
    minimum_hold_seconds: float
    movement_threshold_scales: float
    movement_window_seconds: float
    cooldown_seconds: float
    effect: str
    effect_duration_seconds: float
    sequence_id: str | None = None
    calibration_hand_count: int | None = None


def load_ability_library(
    path: str | Path | None = None,
) -> tuple[AbilityDefinition, ...]:
    """Load a stable, validated ability catalogue from JSON."""
    source = Path(path) if path is not None else Path(__file__).with_name("ability_library.json")
    if not source.is_absolute() and not source.is_file():
        source = Path(__file__).with_name(source.name)
    payload: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != "jarvis.abilities.v1":
        raise ValueError(f"unsupported ability-library schema in {source}")
    raw_abilities = payload.get("abilities")
    if not isinstance(raw_abilities, list):
        raise ValueError("ability library must contain an abilities list")
    abilities: list[AbilityDefinition] = []
    for raw in raw_abilities:
        if not isinstance(raw, dict):
            raise ValueError("each ability must be a JSON object")
        definition = AbilityDefinition(
            ability_id=str(raw["id"]),
            display_name=str(raw["display_name"]),
            franchise=str(raw["franchise"]),
            description=str(raw["description"]),
            hand_configuration=str(raw["hand_configuration"]),
            movement=str(raw["movement"]),
            pose=str(raw["pose"]),
            movement_gate=str(raw["movement_gate"]),
            minimum_hold_seconds=float(raw["minimum_hold_seconds"]),
            movement_threshold_scales=float(raw["movement_threshold_scales"]),
            movement_window_seconds=float(raw["movement_window_seconds"]),
            cooldown_seconds=float(raw["cooldown_seconds"]),
            effect=str(raw["effect"]),
            effect_duration_seconds=float(raw["effect_duration_seconds"]),
            sequence_id=(
                str(raw["sequence_id"])
                if raw.get("sequence_id") is not None
                else None
            ),
            calibration_hand_count=(
                int(raw["calibration_hand_count"])
                if raw.get("calibration_hand_count") is not None
                else None
            ),
        )
        if (
            not definition.ability_id
            or definition.minimum_hold_seconds < 0.0
            or definition.movement_threshold_scales < 0.0
            or definition.movement_window_seconds < 0.0
            or definition.cooldown_seconds < 0.0
            or definition.effect_duration_seconds <= 0.0
            or definition.calibration_hand_count not in (None, 1, 2)
        ):
            raise ValueError(f"invalid ability definition: {definition.ability_id}")
        abilities.append(definition)
    if len({item.ability_id for item in abilities}) != len(abilities):
        raise ValueError("ability ids must be unique")
    return tuple(abilities)
