"""Read frames in a background thread, so inference never waits for the camera (bonus: concurrency)."""

import contextlib
import queue
import threading

import cv2
import numpy as np


class FrameReader:
    """Read frames from a VideoCapture in a background thread. Use read() like cap.read()."""

    def __init__(self, cap: cv2.VideoCapture, keep_latest_only: bool) -> None:
        self._cap = cap
        self._keep_latest_only = keep_latest_only  # camera: skip old frames to show the newest; video file: keep them all
        self._frames: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1 if keep_latest_only else 4)  # None = end of stream
        self._stop = threading.Event()  # set by stop() to end the thread
        self._thread = threading.Thread(target=self._run, daemon=True)  # daemon: never keeps the program alive on its own
        self._thread.start()

    def _run(self) -> None:
        """Background loop: read frames until the stream ends or stop() is called."""
        while not self._stop.is_set():
            ok, frame = self._cap.read()  # blocks until the camera has a new frame
            item = frame if ok else None
            if self._keep_latest_only:
                with contextlib.suppress(queue.Empty):  # empty queue = the main loop already took the last frame
                    self._frames.get_nowait()  # drop the frame nobody took yet, the new one is fresher
            self._put(item)
            if item is None:
                return  # stream ended: the main loop will get None and stop

    def _put(self, item: np.ndarray | None) -> None:
        """Wait for room in the queue, but give up if stop() is called meanwhile."""
        while not self._stop.is_set():
            try:
                self._frames.put(item, timeout=0.1)  # short timeout, so we notice stop() quickly
                return
            except queue.Full:
                continue

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return the next frame, waiting if the thread has not read it yet."""
        frame = self._frames.get()
        return frame is not None, frame  # same (ok, frame) shape as cv2.VideoCapture.read()

    def stop(self) -> None:
        """End the thread before the camera is released, so it never reads from a closed camera."""
        self._stop.set()
        self._thread.join(timeout=1.0)  # cap.read() returns within one frame time, so this is quick
