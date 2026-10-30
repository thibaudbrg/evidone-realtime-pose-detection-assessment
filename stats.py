"""On-screen stats (bonus): frames per second and model time."""

import time
from collections.abc import Mapping

import cv2
import numpy as np

TEXT_COLOR = (255, 255, 255)  # white
OUTLINE_COLOR = (0, 0, 0)  # black outline, so the text is readable on any background


class FpsMeter:
    """Counts frames per second, smoothed so the number doesn't jitter."""

    def __init__(self, smoothing: float = 0.9) -> None:
        self.smoothing = smoothing  # how much of the old value to keep each frame (0 = no smoothing)
        self.fps = 0.0
        self._last_time: float | None = None

    def tick(self) -> float:
        """Call once per frame. Returns the smoothed FPS."""
        now = time.perf_counter()  # a precise clock, in seconds
        if self._last_time is not None:
            instant_fps = 1.0 / max(now - self._last_time, 1e-6)  # 1 / seconds per frame (e.g. 0.0167 for 60 FPS). max() avoids division by zero if the clock is too coarse.
            if self.fps == 0:
                self.fps = instant_fps  # first measurement: nothing to smooth with yet
            else:
                # Exponential moving average: mostly the old value, a little of the new one
                self.fps = self.smoothing * self.fps + (1 - self.smoothing) * instant_fps # (e.g. 0.9 * 60 + 0.1 * 30 = 57 FPS)
        self._last_time = now
        return self.fps


def model_ms(speed: Mapping[str, float | None]) -> float:
    """Total model time in ms, from Ultralytics' result.speed.

    We add the three phases because on the GPU (mps) the work runs in the background:
    "inference" looks tiny and the waiting shows up in "postprocess". Only the total is honest.
    """
    return sum(speed.get(phase) or 0.0 for phase in ("preprocess", "inference", "postprocess"))  # "or 0": a missing phase counts as 0


def draw_stats(frame: np.ndarray, fps: float, ms: float, persons: int) -> None:
    """Write the stats in the top-left corner, in place."""
    text = f"FPS {fps:4.1f} | model {ms:3.0f} ms | {persons} person{'s' if persons != 1 else ''}"
    for color, thickness in [(OUTLINE_COLOR, 4), (TEXT_COLOR, 2)]:  # thick black first, thin white on top
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness, cv2.LINE_AA)
