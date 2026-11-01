"""Temporal smoothing for pose keypoints."""

import time

import numpy as np


class OneEuroFilter:
    """Smooth a signal while reacting quickly when it starts moving."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        derivative_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self._value: np.ndarray | None = None
        self._derivative: np.ndarray | None = None
        self._time: float | None = None

    @staticmethod
    def _alpha(cutoff: float, elapsed: float) -> float:
        smoothing_time = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + smoothing_time / elapsed)

    def filter(self, value: np.ndarray) -> np.ndarray:
        """Return a smoothed copy of the current value."""
        now = time.perf_counter()
        value = np.asarray(value, dtype=np.float64)
        if self._value is None or self._time is None:
            self._value = value.copy()
            self._derivative = np.zeros_like(value)
            self._time = now
            return self._value.copy()

        elapsed = max(now - self._time, 1e-6)
        raw_derivative = (value - self._value) / elapsed
        derivative_alpha = self._alpha(self.derivative_cutoff, elapsed)
        assert self._derivative is not None
        self._derivative += derivative_alpha * (raw_derivative - self._derivative)

        cutoff = self.min_cutoff + self.beta * np.abs(self._derivative)
        value_alpha = 1.0 / (1.0 + (1.0 / (2 * np.pi * cutoff)) / elapsed)
        self._value += value_alpha * (value - self._value)
        self._time = now
        return self._value.copy()


class KeypointSmoother:
    """Keep an independent One Euro filter for each tracked person."""

    def __init__(self) -> None:
        self._filters: dict[int, OneEuroFilter] = {}

    def update(self, track_id: int, keypoints: np.ndarray) -> np.ndarray:
        """Smooth keypoints for a track; leave untracked detections unchanged."""
        if track_id < 0:
            return keypoints.copy()
        smoother = self._filters.setdefault(track_id, OneEuroFilter())
        return smoother.filter(keypoints)

    def remove(self, active_ids: set[int]) -> None:
        """Forget filters for tracks that are no longer active."""
        for track_id in set(self._filters) - active_ids:
            del self._filters[track_id]
