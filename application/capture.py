"""Latest-frame video capture with bounded buffering and source switching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np

import config
from utils.video import VideoSourceError, open_video_source


@dataclass(frozen=True)
class CapturedFrame:
    sequence: int
    timestamp: float
    frame: np.ndarray | None
    source: int | str | None
    status: str
    error: str | None
    running: bool


class LatestFrameCapture:
    """Capture in a dedicated thread, keeping only the newest available frame."""

    def __init__(self, source: int | str | None = None) -> None:
        self._source_queue: queue.SimpleQueue[int | str | None] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._state = CapturedFrame(0, 0.0, None, source, "Starting camera", None, False)
        self._thread = threading.Thread(
            target=self._run,
            name="jarvis-camera-capture",
            daemon=True,
        )
        self._initial_source = source

    @property
    def latest(self) -> CapturedFrame:
        with self._lock:
            return self._state

    def start(self) -> None:
        self._source_queue.put(self._initial_source)
        self._thread.start()

    def request_source(self, source: int | str | None) -> None:
        self._source_queue.put(source)

    def stop(self, timeout: float = 5.0) -> bool:
        self._stop.set()
        if not self._thread.is_alive():
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _publish(self, state: CapturedFrame) -> None:
        with self._lock:
            self._state = state

    def _run(self) -> None:
        capture: cv2.VideoCapture | None = None
        current_source = self._initial_source
        sequence = 0
        last_frame: np.ndarray | None = None
        status = "Initializing camera"
        error: str | None = None
        try:
            while not self._stop.is_set():
                requested_source = current_source
                has_request = False
                while True:
                    try:
                        requested_source = self._source_queue.get_nowait()
                        has_request = True
                    except queue.Empty:
                        break
                if has_request:
                    if capture is not None:
                        capture.release()
                        capture = None
                    current_source = requested_source
                    last_frame = None
                    if current_source is None:
                        status = "Camera stopped"
                        error = None
                    else:
                        status = "Connecting to camera" if isinstance(current_source, int) else "Opening video"
                        self._publish(
                            CapturedFrame(sequence, time.perf_counter(), None, current_source, status, None, False)
                        )
                        try:
                            capture = open_video_source(current_source)
                            status = "Camera ready"
                            error = None
                        except VideoSourceError as failure:
                            status = "Camera unavailable"
                            error = str(failure)
                            current_source = None

                if capture is None:
                    self._publish(
                        CapturedFrame(
                            sequence,
                            time.perf_counter(),
                            last_frame,
                            current_source,
                            status,
                            error,
                            False,
                        )
                    )
                    self._stop.wait(0.02)
                    continue

                read_started = time.perf_counter()
                ok, frame = capture.read()
                frame_time = time.perf_counter()
                if not ok:
                    capture.release()
                    capture = None
                    status = "End of video" if isinstance(current_source, str) else "Camera read failed"
                    error = None if isinstance(current_source, str) else "Camera returned no frame."
                    current_source = None
                    self._publish(
                        CapturedFrame(sequence, frame_time, last_frame, None, status, error, False)
                    )
                    continue

                sequence += 1
                last_frame = frame
                status = "Live camera" if isinstance(current_source, int) else "Video playing"
                self._publish(
                    CapturedFrame(
                        sequence,
                        frame_time,
                        frame,
                        current_source,
                        status,
                        None,
                        True,
                    )
                )
                source_fps = capture.get(cv2.CAP_PROP_FPS)
                frame_interval = (
                    1.0 / source_fps
                    if isinstance(current_source, str) and source_fps > 0
                    else 1.0 / config.CAPTURE_MAX_FPS
                )
                wait_seconds = max(0.0, frame_interval - (frame_time - read_started))
                if wait_seconds:
                    self._stop.wait(wait_seconds)
        finally:
            if capture is not None:
                capture.release()
            current_state = self.latest
            final_status = (
                current_state.status
                if current_state.status
                in ("End of video", "Camera unavailable", "Camera read failed")
                else "Camera stopped"
            )
            self._publish(
                CapturedFrame(
                    sequence,
                    time.perf_counter(),
                    last_frame,
                    current_source,
                    final_status,
                    error,
                    False,
                )
            )


def source_label(source: int | str | None) -> str:
    if source is None:
        return "NO INPUT"
    return f"CAM {source}" if isinstance(source, int) else Path(source).name
