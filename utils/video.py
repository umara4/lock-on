"""Video and camera input helpers."""

from __future__ import annotations

from pathlib import Path

import cv2


class VideoSourceError(Exception):
    """Raised when a camera or video file cannot be opened."""


def open_video_source(source: int | str) -> cv2.VideoCapture:
    """Open and validate a webcam index or prerecorded video path."""
    if isinstance(source, str):
        path = Path(source)
        if not path.is_file():
            raise VideoSourceError(f"video file does not exist: {path}")
        source = str(path)

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        if isinstance(source, int):
            raise VideoSourceError(
                f"could not open camera {source}; check that it is connected "
                "and not in use by another application"
            )
        raise VideoSourceError(f"could not open video file: {source}")
    return capture
