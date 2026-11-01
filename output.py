"""Optional processed-video and keypoint-log outputs."""

import json
import time
from typing import Any

import cv2
import numpy as np


class OutputManager:
    """Write only processed frames and explicit inference results."""

    def __init__(
        self,
        video_path: str | None,
        log_path: str | None,
        fps: float,
        frame_size: tuple[int, int],
        realtime: bool,
    ) -> None:
        """Open both outputs now, so a bad path fails at startup instead of in the middle of a run."""
        self._fps = fps or 30.0  # some cameras report 0 FPS, so fall back to a common value
        self._realtime = realtime  # camera: match wall-clock time; video file: keep every frame exactly once
        self._writer: cv2.VideoWriter | None = None
        self._log_file = None
        self._start: float | None = None  # time of the first saved frame
        self._written = 0  # number of video frames written so far
        try:
            if video_path:
                self._writer = cv2.VideoWriter(video_path, cv2.VideoWriter.fourcc(*"mp4v"), self._fps, frame_size)
                if not self._writer.isOpened():  # OpenCV does not raise, it just returns a closed writer
                    raise OSError(f"could not open video output: {video_path}")
            if log_path:
                self._log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115 - kept open for the whole run, closed in close()
        except OSError:
            self.close()  # don't leave the video file open if the log file fails
            raise

    def write(
        self,
        frame_number: int,
        frame: np.ndarray,
        inference_ok: bool,
        error: str | None,
        persons: list[dict[str, Any]],
    ) -> None:
        """Save one processed frame and its log line."""
        if self._writer is not None:
            copies = self._copies_due() if self._realtime else 1  # a video file already has the right number of frames
            for _ in range(copies):
                self._writer.write(frame)
            self._written += copies
        if self._log_file is not None:
            self._log_file.write(json.dumps({
                "frame": frame_number,
                "inference_ok": inference_ok,
                "error": error,
                "persons": persons,
            }) + "\n")  # one JSON object per line (JSONL)
            self._log_file.flush()  # write to disk now, so the log survives a crash

    def _copies_due(self) -> int:
        """How many times to write this camera frame, so the video plays at real speed."""
        now = time.perf_counter()
        if self._start is None:
            self._start = now
        due = int((now - self._start) * self._fps) + 1  # how many frames the video should hold by now, at its FPS
        return max(1, due - self._written)  # repeat a frame when we process slower than the camera

    def close(self) -> None:
        """Release all output resources."""
        if self._writer is not None:
            self._writer.release()  # finishes the video file so it can be played
        if self._log_file is not None:
            self._log_file.close()
