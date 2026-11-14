"""Thread-confined orchestration of the existing JARVIS vision modules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from time import perf_counter
from typing import Callable

import numpy as np

import config
from interface.dashboard import DashboardEvent, EventLog, EventSeverity
from prediction.motion import MotionEstimate, MotionEstimator
from robotics.interceptor import InterceptionSimulator, InterceptionStatus, SimulationSnapshot
from interface.effects import EffectEngine, EffectSnapshot
from interface.ability_effects import AbilityEffectEngine, AbilityEffectSnapshot
from vision.ability_calibration import AbilityCalibrationStore
from vision.ability_library import load_ability_library
from vision.ability_recognition import AbilityRecognitionSnapshot, AbilityRecognizer
from vision.anime_gestures import AnimeGestureRecognizer, AnimeRecognitionSnapshot
from vision.body_pose import BodyPoseTracker
from vision.commands import (
    CommandEvent,
    CommandType,
    GestureCommandProcessor,
    GestureUpdate,
    SystemCommandRouter,
)
from vision.detector import YOLODetector
from vision.gestures import GestureType, HandGesture, HandGestureRecognizer
from vision.tracker import ByteTrackTracker, TrackedObject
from vision.recognition import (
    ClassificationStatus,
    TrackRecognition,
    TrackRecognitionManager,
)

logger = logging.getLogger(__name__)


class PipelineCommand(str, Enum):
    TOGGLE_DETECTION = "toggle_detection"
    TOGGLE_GESTURES = "toggle_gestures"
    TOGGLE_PAUSE = "toggle_pause"
    TOGGLE_LOCK = "toggle_lock"
    TOGGLE_INTERCEPTION = "toggle_interception"
    CYCLE_TARGET = "cycle_target"
    RESET_TRACKING = "reset_tracking"
    RESET_SIMULATION = "reset_simulation"
    SELECT_TARGET = "select_target"
    RESET = "reset"
    TOGGLE_CLASSIFICATION = "toggle_classification"
    TOGGLE_ANIME_GESTURES = "toggle_anime_gestures"
    TOGGLE_ANIME_EFFECTS = "toggle_anime_effects"
    TOGGLE_ABILITIES = "toggle_abilities"
    SELECT_ABILITY = "select_ability"
    RECORD_ABILITY_SAMPLE = "record_ability_sample"
    TOGGLE_ABILITY_MOVEMENT_RECORDING = "toggle_ability_movement_recording"
    SET_ABILITY_THRESHOLD = "set_ability_threshold"


@dataclass(frozen=True)
class PipelineTiming:
    tracking_seconds: float = 0.0
    tracking_adapter_seconds: float = 0.0
    gesture_seconds: float = 0.0
    motion_seconds: float = 0.0
    end_to_end_seconds: float = 0.0


@dataclass(frozen=True)
class PipelineSnapshot:
    timestamp: float
    tracks: tuple[TrackedObject, ...]
    position_histories: dict[int, tuple[tuple[int, int], ...]]
    motion_estimates: dict[int, MotionEstimate]
    selected_id: int | None
    locked_target_id: int | None
    target_locked: bool
    detection_enabled: bool
    gestures_enabled: bool
    tracking_paused: bool
    interception_active: bool
    hand: HandGesture
    gesture_update: GestureUpdate
    simulation: SimulationSnapshot
    events: tuple[DashboardEvent, ...]
    inference_fps: float
    system_state: str
    feedback: str
    pointing_ray: tuple[tuple[int, int], tuple[int, int]] | None
    initialization_state: str
    error: str | None
    recognition: dict[int, TrackRecognition]
    classification_enabled: bool
    anime_gestures_enabled: bool = config.ANIME_GESTURES_ENABLED
    anime_effects_enabled: bool = config.ANIME_EFFECTS_ENABLED
    anime: AnimeRecognitionSnapshot = AnimeRecognitionSnapshot()
    effects: tuple[EffectSnapshot, ...] = ()
    abilities_enabled: bool = config.ABILITIES_ENABLED
    selected_ability_id: str = "kamehameha"
    ability_samples: dict[str, int] | None = None
    ability_recognition: AbilityRecognitionSnapshot = AbilityRecognitionSnapshot()
    ability_effects: tuple[AbilityEffectSnapshot, ...] = ()
    ability_movement_samples: dict[str, int] | None = None
    ability_calibration_recording: bool = False
    ability_calibration_frames: int = 0
    ability_similarity_threshold: float = config.ABILITY_MIN_CALIBRATION_SIMILARITY

    @property
    def selected_estimate(self) -> MotionEstimate | None:
        target_id = self.locked_target_id if self.target_locked else self.selected_id
        return self.motion_estimates.get(target_id) if target_id is not None else None

    @property
    def selected_track(self) -> TrackedObject | None:
        target_id = self.locked_target_id if self.target_locked else self.selected_id
        return next((track for track in self.tracks if track.track_id == target_id), None)


class VisionPipeline:
    """Own and update perception/control modules on a single worker thread."""

    def __init__(
        self,
        model_path: str,
        confidence: float,
        device: str | None,
        tracker_config: str,
        on_initialization: Callable[[str], None] | None = None,
        inference_size: int = config.TRACKING_INFERENCE_SIZE,
        detector_type: str = config.DETECTOR_TYPE,
        vocabulary: tuple[str, ...] = config.OPEN_VOCABULARY_CLASSES,
        classification_enabled: bool = config.CLASSIFICATION_ENABLED,
    ) -> None:
        self._on_initialization = on_initialization or (lambda _state: None)
        self.initialization_state = "Loading detection model"
        self._on_initialization(self.initialization_state)
        detector = YOLODetector(
            model_path,
            confidence,
            device,
            detector_type=detector_type,
            vocabulary=vocabulary,
        )
        self.tracker = ByteTrackTracker(
            detector.model,
            tracker_config=tracker_config,
            inference_confidence=config.TRACKING_INFERENCE_CONFIDENCE,
            history_length=config.POSITION_HISTORY_LENGTH,
            max_missed_frames=config.TRACKING_MAX_MISSED_FRAMES,
            device=device,
            inference_size=inference_size,
        )
        self.initialization_state = "Initializing gesture recognition"
        self._on_initialization(self.initialization_state)
        self.gesture_recognizer = HandGestureRecognizer()
        self.anime_gesture_recognizer = AnimeGestureRecognizer()
        self.effect_engine = EffectEngine()
        self.ability_library = load_ability_library(config.ABILITY_LIBRARY_PATH)
        self.ability_calibration = AbilityCalibrationStore(
            config.ABILITY_CALIBRATION_PATH,
            config.ABILITY_MAX_CALIBRATION_SAMPLES,
            config.ABILITY_MIN_CALIBRATION_SIMILARITY,
        )
        self.ability_recognizer = AbilityRecognizer(
            self.ability_calibration,
            self.ability_library,
        )
        self.ability_effect_engine = AbilityEffectEngine(
            config.ABILITY_MAX_ACTIVE_EFFECTS,
            self.ability_library,
        )
        self.body_pose_tracker: BodyPoseTracker | None = None
        self._frame_size = (0, 0)
        self.gesture_commands = GestureCommandProcessor()
        self.command_router = SystemCommandRouter()
        self.motion_estimator = MotionEstimator()
        self.recognition_manager = TrackRecognitionManager(enabled=classification_enabled)
        self.recognition: dict[int, TrackRecognition] = {}
        self.interceptor = InterceptionSimulator()
        self.event_log = EventLog(config.GUI_EVENT_LOG_LIMIT)
        self.event_log.add("SYSTEM READY", EventSeverity.SUCCESS)
        self.detection_enabled = True
        self.gestures_enabled = True
        self.anime_gestures_enabled = config.ANIME_GESTURES_ENABLED
        self.anime_effects_enabled = config.ANIME_EFFECTS_ENABLED
        self.abilities_enabled = config.ABILITIES_ENABLED
        self.selected_ability_id = (
            self.ability_library[0].ability_id if self.ability_library else ""
        )
        self.anime_snapshot = AnimeRecognitionSnapshot()
        self.ability_snapshot = AbilityRecognitionSnapshot()
        self.effects: tuple[EffectSnapshot, ...] = ()
        self.ability_effects: tuple[AbilityEffectSnapshot, ...] = ()
        self.last_tracks: list[TrackedObject] = []
        self.motion_estimates: dict[int, MotionEstimate] = {}
        self.last_hand = HandGesture()
        self.gesture_update = GestureUpdate(
            raw_gesture=GestureType.NONE,
            stable_gesture=None,
            stable_frames=0,
            stability=0.0,
            events=(),
        )
        self.last_interception_status = InterceptionStatus.IDLE
        self._last_gesture_timestamp = float("-inf")
        self._last_inference_timestamp: float | None = None
        self._inference_intervals: list[float] = []
        self.last_timing = PipelineTiming()
        self.initialization_state = "System ready"
        self._on_initialization(self.initialization_state)

    def close(self) -> None:
        try:
            if self.body_pose_tracker is not None:
                self.body_pose_tracker.close()
                self.body_pose_tracker = None
            self.ability_effect_engine.close()
            if not self.recognition_manager.close():
                logger.warning("Crop classifier did not stop before its shutdown timeout")
        finally:
            self.gesture_recognizer.close()

    def apply_command(
        self,
        command: PipelineCommand,
        timestamp: float,
        target_id: int | None = None,
        ability_id: str | None = None,
        ability_threshold: float | None = None,
    ) -> PipelineSnapshot:
        if command == PipelineCommand.TOGGLE_DETECTION:
            self.detection_enabled = not self.detection_enabled
            if not self.detection_enabled:
                self._reset_tracking_state()
            message = "DETECTION ENABLED" if self.detection_enabled else "DETECTION DISABLED"
            self.event_log.add(message, EventSeverity.SUCCESS)
        elif command == PipelineCommand.TOGGLE_GESTURES:
            self.gestures_enabled = not self.gestures_enabled
            self.gesture_commands.reset()
            self.last_hand = HandGesture()
            self.event_log.add(
                "GESTURE RECOGNITION ENABLED" if self.gestures_enabled else "GESTURE RECOGNITION DISABLED",
                EventSeverity.SUCCESS,
            )
        elif command == PipelineCommand.TOGGLE_ANIME_GESTURES:
            self.anime_gestures_enabled = not self.anime_gestures_enabled
            self.anime_gesture_recognizer.reset()
            self.anime_snapshot = AnimeRecognitionSnapshot()
            self.effect_engine.reset()
            self.effects = ()
            self.event_log.add(
                "ANIME GESTURES ENABLED"
                if self.anime_gestures_enabled
                else "ANIME GESTURES DISABLED",
                EventSeverity.SUCCESS if self.anime_gestures_enabled else EventSeverity.WARNING,
            )
        elif command == PipelineCommand.TOGGLE_ANIME_EFFECTS:
            self.anime_effects_enabled = not self.anime_effects_enabled
            self.effect_engine.set_enabled(self.anime_effects_enabled)
            self.ability_effect_engine.set_enabled(self.anime_effects_enabled)
            self.effects = ()
            self.ability_effects = ()
            self.event_log.add(
                "ANIME EFFECTS ENABLED"
                if self.anime_effects_enabled
                else "ANIME EFFECTS DISABLED",
                EventSeverity.SUCCESS if self.anime_effects_enabled else EventSeverity.WARNING,
            )
        elif command == PipelineCommand.TOGGLE_ABILITIES:
            self.abilities_enabled = not self.abilities_enabled
            self.ability_recognizer.reset()
            self.ability_snapshot = AbilityRecognitionSnapshot()
            self.ability_effect_engine.reset()
            self.ability_effects = ()
            self.event_log.add(
                "ABILITY RECOGNITION ENABLED"
                if self.abilities_enabled
                else "ABILITY RECOGNITION DISABLED",
                EventSeverity.SUCCESS if self.abilities_enabled else EventSeverity.WARNING,
            )
        elif command == PipelineCommand.SELECT_ABILITY:
            if ability_id not in {item.ability_id for item in self.ability_library}:
                self.event_log.add(
                    f"UNKNOWN ABILITY SELECTION: {ability_id}",
                    EventSeverity.ERROR,
                )
            else:
                self.selected_ability_id = str(ability_id)
                if ability_id == "shadow_clone":
                    self.ability_effect_engine.prepare_segmentation()
        elif command == PipelineCommand.RECORD_ABILITY_SAMPLE:
            if ability_id not in {item.ability_id for item in self.ability_library}:
                self.event_log.add(
                    f"UNKNOWN ABILITY CALIBRATION: {ability_id}",
                    EventSeverity.ERROR,
                )
            else:
                try:
                    count = self.ability_calibration.record(
                        str(ability_id),
                        self.anime_snapshot.hands,
                        next(
                            item.calibration_hand_count
                            for item in self.ability_library
                            if item.ability_id == ability_id
                        ),
                        (
                            self._frame_size[0] / self._frame_size[1]
                            if self._frame_size[1] > 0
                            else 1.0
                        ),
                    )
                except ValueError as error:
                    self.event_log.add(
                        f"CALIBRATION SAMPLE NOT SAVED: {error}",
                        EventSeverity.WARNING,
                    )
                else:
                    self.event_log.add(
                        f"CALIBRATION SAVED: {ability_id} SAMPLE {count}",
                        EventSeverity.SUCCESS,
                    )
        elif command == PipelineCommand.TOGGLE_ABILITY_MOVEMENT_RECORDING:
            if self.ability_calibration.recording_ability is None:
                selected = ability_id or self.selected_ability_id
                if selected not in {item.ability_id for item in self.ability_library}:
                    self.event_log.add(
                        f"UNKNOWN ABILITY SEQUENCE CALIBRATION: {selected}",
                        EventSeverity.ERROR,
                    )
                else:
                    self.ability_calibration.begin_movement_recording(selected, timestamp)
                    self.event_log.add(
                        f"MOVEMENT CALIBRATION STARTED: {selected}",
                        EventSeverity.INFO,
                    )
            else:
                try:
                    count = self.ability_calibration.finish_movement_recording()
                except ValueError as error:
                    self.event_log.add(
                        f"MOVEMENT SAMPLE NOT SAVED: {error}",
                        EventSeverity.WARNING,
                    )
                else:
                    self.event_log.add(
                        f"MOVEMENT SAMPLE SAVED: {ability_id or self.selected_ability_id} "
                        f"SEQUENCE {count}",
                        EventSeverity.SUCCESS,
                    )
        elif command == PipelineCommand.SET_ABILITY_THRESHOLD:
            if ability_threshold is None:
                self.event_log.add(
                    "ABILITY THRESHOLD NOT CHANGED: no value supplied",
                    EventSeverity.ERROR,
                )
            else:
                try:
                    self.ability_calibration.set_minimum_similarity(ability_threshold)
                except ValueError as error:
                    self.event_log.add(
                        f"ABILITY THRESHOLD NOT CHANGED: {error}",
                        EventSeverity.WARNING,
                    )
                else:
                    self.event_log.add(
                        f"ABILITY CALIBRATION SIMILARITY: {ability_threshold:.2f}",
                        EventSeverity.INFO,
                    )
        elif command == PipelineCommand.TOGGLE_CLASSIFICATION:
            enabled = not self.recognition_manager.enabled
            self.recognition_manager.set_enabled(enabled)
            self.recognition = {}
            self.event_log.add(
                "CLASSIFICATION ENABLED" if enabled else "CLASSIFICATION DISABLED",
                EventSeverity.SUCCESS,
            )
        elif command == PipelineCommand.TOGGLE_PAUSE:
            self.command_router.state.tracking_paused = not self.command_router.state.tracking_paused
            paused = self.command_router.state.tracking_paused
            self.event_log.add("TRACKING PAUSED" if paused else "TRACKING RESUMED", EventSeverity.WARNING if paused else EventSeverity.SUCCESS)
        elif command == PipelineCommand.TOGGLE_LOCK:
            self.command_router.handle(
                CommandEvent(CommandType.TOGGLE_TARGET_LOCK),
                self.tracker,
                self.last_tracks,
                (1, 1, 3),
            )
            self.event_log.add(self.command_router.state.feedback, EventSeverity.SUCCESS)
        elif command == PipelineCommand.TOGGLE_INTERCEPTION:
            was_active = self.command_router.state.interception_active
            self.command_router.state.interception_active = not was_active
            self.command_router.state.feedback = (
                "INTERCEPTION MODE ACTIVE"
                if self.command_router.state.interception_active
                else "INTERCEPTION MODE INACTIVE"
            )
            if self.command_router.state.interception_active:
                selected_id = (
                    self.command_router.state.locked_target_id
                    if self.command_router.state.target_locked
                    else self.tracker.selected_id
                )
                if (
                    selected_id is None
                    or selected_id not in self.motion_estimates
                    or not any(track.track_id == selected_id for track in self.last_tracks)
                ):
                    self.command_router.state.interception_active = False
                    self.command_router.state.feedback = "NO VALID TARGET FOR INTERCEPTION"
            self.event_log.add(
                self.command_router.state.feedback if was_active or not self.command_router.state.interception_active else "INTERCEPTION ACTIVATED",
                EventSeverity.WARNING if not self.command_router.state.interception_active else EventSeverity.SUCCESS,
            )
        elif command == PipelineCommand.CYCLE_TARGET:
            self.command_router.handle(
                CommandEvent(CommandType.CYCLE_TARGET),
                self.tracker,
                self.last_tracks,
                (1, 1, 3),
            )
            self.event_log.add(self.command_router.state.feedback)
        elif command == PipelineCommand.SELECT_TARGET and target_id is not None:
            if self.command_router.state.target_locked:
                self.command_router.state.feedback = "UNLOCK TARGET BEFORE SELECTING"
            elif self.tracker.select(target_id):
                self.command_router.state.feedback = f"TARGET ID {target_id} SELECTED"
                self.event_log.add(self.command_router.state.feedback, EventSeverity.SUCCESS)
        elif command in (PipelineCommand.RESET_TRACKING, PipelineCommand.RESET):
            self._reset_tracking_state()
            self.event_log.add("TRACKING RESET", EventSeverity.WARNING)
        elif command == PipelineCommand.RESET_SIMULATION:
            self.interceptor.reset()
            self.command_router.state.interception_active = False
            self.command_router.state.feedback = "SIMULATION RESET"
            self.event_log.add("SIMULATION RESET")
        return self._publish(timestamp)

    def process_frame(self, frame: np.ndarray, timestamp: float) -> PipelineSnapshot:
        frame_started = perf_counter()
        self._frame_size = (frame.shape[1], frame.shape[0])
        tracking_seconds = 0.0
        tracking_adapter_seconds = 0.0
        gesture_seconds = 0.0
        motion_seconds = 0.0
        previous_ids = {track.track_id for track in self.last_tracks}
        if self.detection_enabled and not self.command_router.state.tracking_paused:
            self.last_tracks = self.tracker.update(frame, timestamp)
            tracking_seconds = self.tracker.last_timing.ultralytics_track_seconds
            tracking_adapter_seconds = self.tracker.last_timing.adapter_seconds
            current_ids = {track.track_id for track in self.last_tracks}
            for track_id in sorted(current_ids - previous_ids):
                self.event_log.add(f"OBJECT DETECTED: ID {track_id}", EventSeverity.SUCCESS)
            self.command_router.validate_lock(self.tracker)
            motion_started = perf_counter()
            self.motion_estimates = self.motion_estimator.update(self.last_tracks, timestamp)
            motion_seconds = perf_counter() - motion_started
            selected_target_id = (
                self.command_router.state.locked_target_id
                if self.command_router.state.target_locked
                else self.tracker.selected_id
            )
            previous_recognition = self.recognition
            self.recognition = self.recognition_manager.update(
                frame,
                self.last_tracks,
                {track.track_id for track in self.tracker.retained_tracks},
                timestamp,
                selected_target_id,
            )
            for track_id, result in self.recognition.items():
                previous = previous_recognition.get(track_id)
                if result.status == ClassificationStatus.REFINED and (
                    previous is None
                    or previous.refined_label != result.refined_label
                ):
                    self.event_log.add(
                        f"CLASSIFICATION REFINED: ID {track_id} "
                        f"{result.detector_label} -> {result.refined_label}",
                        EventSeverity.INFO,
                    )
                elif result.error and (previous is None or previous.error != result.error):
                    self.event_log.add(
                        f"CLASSIFICATION ERROR: ID {track_id}: {result.error}",
                        EventSeverity.WARNING,
                    )

            if self._last_inference_timestamp is not None:
                self._inference_intervals.append(timestamp - self._last_inference_timestamp)
                self._inference_intervals = self._inference_intervals[-30:]
            self._last_inference_timestamp = timestamp
        elif not self.detection_enabled and not self.command_router.state.tracking_paused:
            self.last_tracks = []
            self.motion_estimates = {}

        if (
            self.gestures_enabled
            or self.anime_gestures_enabled
            or self.abilities_enabled
            or self.ability_calibration.recording_ability is not None
        ) and (
            timestamp - self._last_gesture_timestamp
            >= config.GESTURE_PROCESSING_INTERVAL_SECONDS
        ):
            gesture_started = perf_counter()
            self.last_hand = self.gesture_recognizer.process(frame)
            gesture_seconds = perf_counter() - gesture_started
            if self.anime_gestures_enabled or self.abilities_enabled:
                self.anime_snapshot = self.anime_gesture_recognizer.update(
                    self.last_hand.hands,
                    timestamp,
                    (frame.shape[1], frame.shape[0]),
                )
                if self.anime_gestures_enabled:
                    self.effect_engine.consume(self.anime_snapshot, timestamp)
                    for event in self.anime_snapshot.events:
                        if event.kind == "recognized":
                            self.event_log.add(
                                f"ANIME POSE: {event.display_name}",
                                EventSeverity.SUCCESS,
                            )
                    for event in self.anime_snapshot.sequence_events:
                        self.event_log.add(
                            f"GESTURE SEQUENCE: {event.display_name}",
                            EventSeverity.SUCCESS,
                        )
                if self.abilities_enabled:
                    pose_landmarks: tuple[
                        tuple[float, float, float, float], ...
                    ] = ()
                    visible_hand_count = sum(
                        hand.visible for hand in self.anime_snapshot.hands
                    )
                    if visible_hand_count >= 2:
                        if self.body_pose_tracker is None:
                            self.body_pose_tracker = BodyPoseTracker()
                        pose_landmarks = self.body_pose_tracker.process(frame)
                    self.ability_snapshot = self.ability_recognizer.update(
                        self.anime_snapshot,
                        timestamp,
                        pose_landmarks,
                        (frame.shape[1], frame.shape[0]),
                    )
                    for event in self.ability_snapshot.events:
                        severity = (
                            EventSeverity.SUCCESS
                            if event.phase == "activated"
                            else EventSeverity.INFO
                        )
                        self.event_log.add(
                            f"ABILITY {event.phase.upper()}: {event.display_name}",
                            severity,
                        )
            two_hand_ability_pose = (
                self.abilities_enabled
                and len(self.last_hand.hands) == 2
                and all(
                    hand.gesture in (GestureType.OPEN_PALM, GestureType.FIST)
                    for hand in self.last_hand.hands
                )
            )
            ability_pose_active = (
                self.abilities_enabled
                and self.ability_snapshot.selected_ability_id is not None
            )
            suppress_system_command = two_hand_ability_pose or ability_pose_active
            if self.gestures_enabled and not suppress_system_command:
                self.gesture_update = self.gesture_commands.update(self.last_hand, timestamp)
                self.command_router.set_hand(self.last_hand)
                for event in self.gesture_update.events:
                    was_active = self.command_router.state.interception_active
                    self.command_router.handle(
                        event,
                        self.tracker,
                        self.last_tracks,
                        frame.shape,
                    )
                    if (
                        event.command == CommandType.TOGGLE_INTERCEPTION
                        and not was_active
                        and self.command_router.state.interception_active
                    ):
                        target_id = (
                            self.command_router.state.locked_target_id
                            if self.command_router.state.target_locked
                            else self.tracker.selected_id
                        )
                        if target_id is None or target_id not in self.motion_estimates:
                            self.command_router.state.interception_active = False
                            self.command_router.state.feedback = "NO VALID TARGET FOR INTERCEPTION"
                    self.event_log.add(
                        f"GESTURE COMMAND: {event.command.value.upper()}",
                        EventSeverity.INFO,
                    )
                stable = self.gesture_update.stable_gesture
                if stable is not None and stable != GestureType.NONE:
                    self.event_log.add(f"GESTURE RECOGNIZED: {stable.value.upper()}")
            else:
                if suppress_system_command:
                    self.gesture_commands.reset()
                self.gesture_update = self.gesture_commands.update(HandGesture(), timestamp)
                self.command_router.set_hand(HandGesture())
            if self.ability_calibration.recording_ability is not None:
                self.ability_calibration.append_movement_frame(
                    self.anime_snapshot.hands,
                    timestamp,
                )
            self._last_gesture_timestamp = timestamp
        elif not self.gestures_enabled and not self.anime_gestures_enabled and not self.abilities_enabled:
            self.last_hand = HandGesture()
            self.command_router.set_hand(self.last_hand)
            self.gesture_update = self.gesture_commands.update(self.last_hand, timestamp)

        self.effects = self.effect_engine.update(timestamp, self.anime_snapshot)
        self.ability_effects = self.ability_effect_engine.update(
            timestamp,
            self.ability_snapshot,
            self.anime_snapshot.hands,
            frame,
        )
        target_id = (
            self.command_router.state.locked_target_id
            if self.command_router.state.target_locked
            else self.tracker.selected_id
        )
        target_visible = any(track.track_id == target_id for track in self.last_tracks)
        target_estimate = (
            self.motion_estimates.get(target_id)
            if target_id is not None and target_visible
            else None
        )
        simulation = self.interceptor.step(
            target_estimate,
            timestamp,
            enabled=self.command_router.state.interception_active,
            width=frame.shape[1],
            height=frame.shape[0],
            paused=self.command_router.state.tracking_paused,
        )
        if simulation.status != self.last_interception_status:
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
                self.event_log.add(f"INTERCEPTION: {simulation.status.value}", severity)
            self.last_interception_status = simulation.status
        self.last_timing = PipelineTiming(
            tracking_seconds=tracking_seconds,
            tracking_adapter_seconds=tracking_adapter_seconds,
            gesture_seconds=gesture_seconds,
            motion_seconds=motion_seconds,
            end_to_end_seconds=perf_counter() - frame_started,
        )
        return self._publish(timestamp, simulation)

    def _reset_tracking_state(self) -> None:
        self.tracker.reset()
        self.motion_estimator.reset()
        self.recognition_manager.reset()
        self.gesture_commands.reset()
        self.anime_gesture_recognizer.reset()
        self.ability_recognizer.reset()
        self.effect_engine.reset()
        self.ability_effect_engine.reset()
        self.anime_snapshot = AnimeRecognitionSnapshot()
        self.ability_snapshot = AbilityRecognitionSnapshot()
        self.effects = ()
        self.ability_effects = ()
        self.interceptor.reset()
        self.last_tracks = []
        self.motion_estimates = {}
        self.recognition = {}
        self.command_router.state.tracking_paused = False
        self.command_router.state.target_locked = False
        self.command_router.state.locked_target_id = None
        self.command_router.state.interception_active = False

    def _publish(
        self,
        timestamp: float,
        simulation: SimulationSnapshot | None = None,
        error: str | None = None,
    ) -> PipelineSnapshot:
        selected_id = self.tracker.selected_id
        current_state = self.command_router.state
        if error:
            system_state = "ERROR"
        elif current_state.tracking_paused:
            system_state = "PAUSED"
        elif simulation and simulation.status == InterceptionStatus.INTERCEPTED:
            system_state = "INTERCEPTED"
        elif current_state.interception_active:
            system_state = "INTERCEPTION ACTIVE"
        elif current_state.target_locked:
            system_state = "TARGET LOCKED"
        elif selected_id is not None:
            system_state = "TARGET SELECTED"
        elif self.detection_enabled:
            system_state = "TRACKING" if self.last_tracks else "DETECTING"
        else:
            system_state = "READY"
        inference_fps = (
            1.0 / float(np.mean(self._inference_intervals))
            if self._inference_intervals
            else 0.0
        )
        return PipelineSnapshot(
            timestamp=timestamp,
            tracks=tuple(self.last_tracks),
            position_histories=self.tracker.position_histories,
            motion_estimates=dict(self.motion_estimates),
            selected_id=selected_id,
            locked_target_id=current_state.locked_target_id,
            target_locked=current_state.target_locked,
            detection_enabled=self.detection_enabled,
            gestures_enabled=self.gestures_enabled,
            tracking_paused=current_state.tracking_paused,
            interception_active=current_state.interception_active,
            hand=self.last_hand,
            gesture_update=self.gesture_update,
            simulation=simulation or self.interceptor.snapshot,
            events=self.event_log.events,
            inference_fps=inference_fps,
            system_state=system_state,
            feedback=current_state.feedback,
            pointing_ray=self.command_router.pointing_ray,
            initialization_state=self.initialization_state,
            error=error,
            recognition=self.recognition_manager.snapshot(),
            classification_enabled=self.recognition_manager.enabled,
            anime_gestures_enabled=self.anime_gestures_enabled,
            anime_effects_enabled=self.anime_effects_enabled,
            anime=self.anime_snapshot,
            effects=self.effects,
            abilities_enabled=self.abilities_enabled,
            selected_ability_id=self.selected_ability_id,
            ability_samples=self.ability_calibration.sample_counts,
            ability_recognition=self.ability_snapshot,
            ability_effects=self.ability_effects,
            ability_movement_samples=self.ability_calibration.movement_sample_counts,
            ability_calibration_recording=(
                self.ability_calibration.recording_ability is not None
            ),
            ability_calibration_frames=self.ability_calibration.recording_frames,
            ability_similarity_threshold=self.ability_calibration.minimum_similarity,
        )
