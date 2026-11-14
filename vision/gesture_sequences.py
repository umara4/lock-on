"""Time-bounded gesture sequences with explicit release and replay gating."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class GestureSequence:
    sequence_id: str
    display_name: str
    steps: tuple[str, ...]
    max_gap_seconds: float
    completion_effect: str

    def __post_init__(self) -> None:
        if not self.steps or self.max_gap_seconds <= 0.0:
            raise ValueError("a gesture sequence needs steps and a positive time limit")


@dataclass(frozen=True)
class SequenceProgress:
    sequence_id: str
    display_name: str
    completed_steps: int
    total_steps: int
    waiting_for_release: bool

    @property
    def text(self) -> str:
        return f"{self.display_name}  {self.completed_steps}/{self.total_steps}"


@dataclass(frozen=True)
class SequenceEvent:
    sequence_id: str
    display_name: str
    completion_effect: str


@dataclass
class _SequenceState:
    completed_steps: int = 0
    waiting_for_release: bool = False
    last_step_time: float | None = None
    latched: bool = False


class GestureSequenceRecognizer:
    """Advance configured sequences only across recognized poses and releases."""

    def __init__(self, sequences: Iterable[GestureSequence]) -> None:
        self.sequences = tuple(sequences)
        if len({sequence.sequence_id for sequence in self.sequences}) != len(self.sequences):
            raise ValueError("gesture sequence ids must be unique")
        self._states = {
            sequence.sequence_id: _SequenceState()
            for sequence in self.sequences
        }

    def reset(self) -> None:
        for state in self._states.values():
            state.completed_steps = 0
            state.waiting_for_release = False
            state.last_step_time = None
            state.latched = False

    def update(
        self,
        events: Iterable[tuple[str, str]],
        timestamp: float,
    ) -> tuple[tuple[SequenceProgress, ...], tuple[SequenceEvent, ...]]:
        """Consume ``(kind, pose_id)`` transitions and return progress/completions.

        A pose must be released between consecutive signs. Completion latches
        until the final sign is released, preventing repeated activations.
        """
        transitions = tuple(events)
        completed: list[SequenceEvent] = []
        for sequence in self.sequences:
            state = self._states[sequence.sequence_id]
            if (
                state.last_step_time is not None
                and timestamp - state.last_step_time > sequence.max_gap_seconds
            ):
                state.completed_steps = 0
                state.waiting_for_release = False
                state.last_step_time = None
                state.latched = False

            for kind, pose_id in transitions:
                if kind == "released":
                    state.waiting_for_release = False
                    if state.latched:
                        state.latched = False
                        state.completed_steps = 0
                        state.last_step_time = None
                    continue
                if kind != "recognized" or state.waiting_for_release or state.latched:
                    continue

                expected = sequence.steps[state.completed_steps]
                if pose_id == expected:
                    state.completed_steps += 1
                    state.last_step_time = timestamp
                    state.waiting_for_release = True
                    if state.completed_steps == len(sequence.steps):
                        completed.append(
                            SequenceEvent(
                                sequence.sequence_id,
                                sequence.display_name,
                                sequence.completion_effect,
                            )
                        )
                        state.latched = True
                elif pose_id == sequence.steps[0]:
                    state.completed_steps = 1
                    state.last_step_time = timestamp
                    state.waiting_for_release = True
                else:
                    state.completed_steps = 0
                    state.waiting_for_release = False
                    state.last_step_time = None

        progress = tuple(
            SequenceProgress(
                sequence.sequence_id,
                sequence.display_name,
                self._states[sequence.sequence_id].completed_steps,
                len(sequence.steps),
                self._states[sequence.sequence_id].waiting_for_release,
            )
            for sequence in self.sequences
            if self._states[sequence.sequence_id].completed_steps
        )
        return progress, tuple(completed)
