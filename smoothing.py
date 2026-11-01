"""Temporal smoothing for pose keypoints and held face boxes."""

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
    def _alpha(cutoff: float | np.ndarray, elapsed: float) -> float | np.ndarray:
        """How much of the new value to take (0 = keep the old value, 1 = jump to the new one)."""
        smoothing_time = 1.0 / (2 * np.pi * cutoff)  # a low cutoff means a long smoothing time
        return 1.0 / (1.0 + smoothing_time / elapsed)  # works on one number or on a whole array of cutoffs

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

        cutoff = self.min_cutoff + self.beta * np.abs(self._derivative)  # fast movement -> higher cutoff -> less smoothing, less lag
        value_alpha = self._alpha(cutoff, elapsed)  # one alpha per coordinate
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


def grow_box(corners: np.ndarray, factor: float) -> np.ndarray:
    """Grow a polygon around its centre, keeping a held privacy box conservative."""
    center = corners.mean(axis=0)
    grown = center + (corners - center) * (1 + factor)  # push every corner away from the centre
    return np.where(grown < center, np.floor(grown), np.ceil(grown)).astype(np.int32)  # round away from the centre, so small boxes still grow by at least a pixel


class FaceHold:
    """Keep hiding a tracked person's face for a few frames after the model loses them (no flicker)."""

    def __init__(self, hold_frames: int = 10, growth: float = 0.05) -> None:
        self.hold_frames = hold_frames  # 10 frames is about a third of a second at 30 FPS
        self.growth = growth  # grow the box 5% per missed frame, because the person may move while undetected
        self._boxes: dict[int, tuple[np.ndarray, int]] = {}  # track ID -> (last face box, frames missed since then)

    @property
    def ids(self) -> set[int]:
        """Track IDs whose face is still remembered."""
        return set(self._boxes)

    def seen(self, track_id: int, box: np.ndarray) -> None:
        """Remember the face box of a person found in this frame."""
        if track_id >= 0:  # -1 = not tracked yet, so there is no ID to remember it under
            self._boxes[track_id] = (box, 0)  # 0 = missed for 0 frames

    def missing_boxes(self, seen_ids: set[int]) -> list[np.ndarray]:
        """Return the boxes to keep hiding for people the model missed in this frame."""
        held = []
        for track_id, (box, missed) in list(self._boxes.items()):  # list() because we delete while looping
            if track_id in seen_ids:
                continue  # found this frame, so its face was already hidden
            if missed >= self.hold_frames:
                del self._boxes[track_id]  # gone for too long: the person most likely left
                continue
            box = grow_box(box, self.growth)  # at most 1.05 ** 10 ≈ 1.6× the last seen size
            self._boxes[track_id] = (box, missed + 1)
            held.append(box)
        return held
