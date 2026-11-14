"""Tests for latest-frame capture, source shutdown, and bounded frame delivery."""

from __future__ import annotations

import time
import unittest
from unittest import mock

import numpy as np

from application.capture import LatestFrameCapture


class FakeCapture:
    def __init__(self) -> None:
        self.index = 0
        self.released = False

    def read(self) -> tuple[bool, np.ndarray | None]:
        time.sleep(0.002)
        if self.index >= 5:
            return False, None
        self.index += 1
        return True, np.full((24, 32, 3), self.index, dtype=np.uint8)

    def get(self, _property: int) -> float:
        return 0.0

    def release(self) -> None:
        self.released = True


class LatestFrameCaptureTests(unittest.TestCase):
    def test_video_capture_keeps_latest_frame_and_releases_at_end(self) -> None:
        fake = FakeCapture()
        with mock.patch("application.capture.open_video_source", return_value=fake):
            capture = LatestFrameCapture("test.mp4")
            capture.start()
            deadline = time.monotonic() + 2.0
            while capture.latest.status != "End of video" and time.monotonic() < deadline:
                time.sleep(0.005)
            stopped = capture.stop()

        self.assertTrue(stopped)
        self.assertTrue(fake.released)
        self.assertEqual(capture.latest.sequence, 5)
        self.assertEqual(capture.latest.status, "End of video")
        self.assertEqual(int(capture.latest.frame[0, 0, 0]), 5)

