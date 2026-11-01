"""Optional processed-video and keypoint-log outputs."""

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class OutputManager:
    """Write only processed frames and explicit inference results."""

    def __init__(self, video_path: str | None, log_path: str | None, fps: float) -> None:
        self._video_path = video_path
        self._fps = fps or 30.0
        self._writer: cv2.VideoWriter | None = None
        self._log_file = open(log_path, "w", encoding="utf-8") if log_path else None

    def write(
        self,
        frame_number: int,
        frame: np.ndarray,
        inference_ok: bool,
        error: str | None,
        persons: list[dict[str, Any]],
    ) -> None:
        """Create outputs lazily, after the frame dimensions are known."""
        if self._video_path and self._writer is None:
            height, width = frame.shape[:2]
            self._writer = cv2.VideoWriter(
                self._video_path,
                cv2.VideoWriter.fourcc(*"mp4v"),
                self._fps,
                (width, height),
            )
            if not self._writer.isOpened():
                raise RuntimeError(f"could not open video output: {self._video_path}")
        if self._writer is not None:
            self._writer.write(frame)
        if self._log_file is not None:
            self._log_file.write(json.dumps({
                "frame": frame_number,
                "inference_ok": inference_ok,
                "error": error,
                "persons": persons,
            }) + "\n")
            self._log_file.flush()

    def close(self) -> None:
        """Release all output resources."""
        if self._writer is not None:
            self._writer.release()
        if self._log_file is not None:
            self._log_file.close()

    def __enter__(self) -> "OutputManager":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
