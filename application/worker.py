"""Latest-frame vision worker; model initialization and inference stay off the UI thread."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import queue
import threading
import time

import config
from application.capture import LatestFrameCapture
from application.pipeline import PipelineCommand, PipelineSnapshot, VisionPipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerCommand:
    command: PipelineCommand
    target_id: int | None = None
    ability_id: str | None = None
    ability_threshold: float | None = None


@dataclass(frozen=True)
class VisionWorkerState:
    snapshot: PipelineSnapshot | None
    processed_sequence: int
    initialization_state: str
    error: str | None
    running: bool
    yolo_initialization_ms: float | None = None
    gesture_initialization_ms: float | None = None
    first_processed_frame_ms: float | None = None


class LatestFrameVisionWorker:
    """Run one pipeline on the newest captured frame without queuing stale work."""

    def __init__(
        self,
        capture: LatestFrameCapture,
        model_path: str,
        confidence: float,
        device: str | None,
        tracker_config: str,
        inference_size: int = config.TRACKING_INFERENCE_SIZE,
        detector_type: str = config.DETECTOR_TYPE,
        vocabulary: tuple[str, ...] = config.OPEN_VOCABULARY_CLASSES,
        classification_enabled: bool = config.CLASSIFICATION_ENABLED,
    ) -> None:
        self.capture = capture
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self.tracker_config = tracker_config
        self.inference_size = inference_size
        self.detector_type = detector_type
        self.vocabulary = vocabulary
        self.classification_enabled = classification_enabled
        self._commands: queue.SimpleQueue[WorkerCommand] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker_started_at: float | None = None
        self._gesture_init_started_at: float | None = None
        self._yolo_initialization_ms: float | None = None
        self._gesture_initialization_ms: float | None = None
        self._first_processed_frame_ms: float | None = None
        self._state = VisionWorkerState(None, 0, "Waiting to initialize", None, False)
        self._thread = threading.Thread(
            target=self._run,
            name="jarvis-vision-pipeline",
            daemon=True,
        )

    @property
    def latest(self) -> VisionWorkerState:
        with self._lock:
            return self._state

    def start(self) -> None:
        self._thread.start()

    def submit(
        self,
        command: PipelineCommand,
        target_id: int | None = None,
        ability_id: str | None = None,
        ability_threshold: float | None = None,
    ) -> None:
        self._commands.put(
            WorkerCommand(command, target_id, ability_id, ability_threshold)
        )

    def stop(self, timeout: float = 10.0) -> bool:
        self._stop.set()
        if not self._thread.is_alive():
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _publish(
        self,
        snapshot: PipelineSnapshot | None,
        processed_sequence: int,
        initialization_state: str,
        error: str | None = None,
        running: bool = True,
    ) -> None:
        with self._lock:
            self._state = VisionWorkerState(
                snapshot,
                processed_sequence,
                initialization_state,
                error,
                running,
                self._yolo_initialization_ms,
                self._gesture_initialization_ms,
                self._first_processed_frame_ms,
            )

    def _run(self) -> None:
        pipeline: VisionPipeline | None = None
        processed_sequence = 0
        worker_started_at = time.perf_counter()
        self._worker_started_at = worker_started_at

        def initialization_update(state: str) -> None:
            now = time.perf_counter()
            if state == "Initializing gesture recognition":
                self._yolo_initialization_ms = (
                    now - worker_started_at
                ) * 1_000.0
                self._gesture_init_started_at = now
            elif state == "System ready" and self._gesture_init_started_at is not None:
                self._gesture_initialization_ms = (
                    now - self._gesture_init_started_at
                ) * 1_000.0
            self._publish(None, 0, state)

        try:
            self._publish(None, 0, "Loading detection model")
            pipeline = VisionPipeline(
                model_path=self.model_path,
                confidence=self.confidence,
                device=self.device,
                tracker_config=self.tracker_config,
                inference_size=self.inference_size,
                detector_type=self.detector_type,
                vocabulary=self.vocabulary,
                classification_enabled=self.classification_enabled,
                on_initialization=initialization_update,
            )
            while not self._stop.is_set():
                command_processed = False
                while True:
                    try:
                        command = self._commands.get_nowait()
                    except queue.Empty:
                        break
                    command_snapshot = pipeline.apply_command(
                        command.command,
                        time.perf_counter(),
                        command.target_id,
                        command.ability_id,
                        command.ability_threshold,
                    )
                    command_processed = True

                captured = self.capture.latest
                if (
                    captured.frame is not None
                    and captured.sequence > processed_sequence
                ):
                    snapshot = pipeline.process_frame(captured.frame, captured.timestamp)
                    processed_sequence = captured.sequence
                    if self._first_processed_frame_ms is None:
                        self._first_processed_frame_ms = (
                            time.perf_counter() - worker_started_at
                        ) * 1_000.0
                    self._publish(
                        snapshot,
                        processed_sequence,
                        pipeline.initialization_state,
                    )
                elif command_processed:
                    self._publish(
                        command_snapshot,
                        processed_sequence,
                        pipeline.initialization_state,
                    )
                else:
                    self._stop.wait(0.005)
        except Exception as error:
            logger.exception("Vision pipeline stopped after an error")
            self._publish(
                self._state.snapshot,
                processed_sequence,
                "Vision pipeline error",
                str(error),
                running=False,
            )
        finally:
            if pipeline is not None:
                try:
                    pipeline.close()
                except Exception:
                    logger.exception("Failed to close vision pipeline resources")
            state = self.latest
            self._publish(
                state.snapshot,
                state.processed_sequence,
                state.initialization_state,
                state.error,
                running=False,
            )
