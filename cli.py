"""Command-line parsing and application resource setup."""

import argparse
import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

from output import OutputManager

DEFAULT_MODEL = "yolo26n-pose.pt"
ANONYMIZATION_MODES = ("blur", "pixelate", "fill")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time pose detection with face anonymization.")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="minimum person confidence, between 0 and 1 (default: 0.4)")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu",
                        help="where the model runs: cpu, or mps = the Apple Silicon GPU (default: cpu)")
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True,
                        help="flip the image left-right like a selfie (default: on, --no-mirror to turn off)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"path to a YOLO pose model (default: {DEFAULT_MODEL}, downloaded if absent)")
    parser.add_argument("--mode", choices=ANONYMIZATION_MODES, default="blur",
                        help="face anonymization mode (default: blur; press b, p or f to switch)")
    parser.add_argument("--benchmark", type=int, default=0, metavar="N",
                        help="process N frames, print read/infer/draw timings, then exit (default: off)")
    parser.add_argument("--save", metavar="PATH",
                        help="save processed, anonymized video to PATH (default: off)")
    parser.add_argument("--log", dest="log_path", metavar="PATH",
                        help="write per-frame tracked keypoints as JSONL to PATH (default: off)")
    parser.add_argument("--threads", type=int, metavar="N",
                        help="set PyTorch CPU thread count before loading the model (default: PyTorch default)")
    args = parser.parse_args()
    if not 0 <= args.conf <= 1:
        parser.error(f"--conf must be between 0 and 1, got {args.conf}")  # prints usage and exits with code 2
    if args.benchmark < 0:
        parser.error(f"--benchmark must be 0 or greater, got {args.benchmark}")
    if args.threads is not None and args.threads < 1:
        parser.error(f"--threads must be at least 1, got {args.threads}")
    return args


def check_device(device: str) -> None:
    """Exit with code 1 and a clear message if the chosen device is not available on this machine."""
    if device == "mps" and not torch.backends.mps.is_available():
        sys.exit("Error: --device mps needs an Apple Silicon Mac with a recent PyTorch. Use --device cpu instead.")


def load_model(path: str) -> YOLO:
    """Load the pose model once, or exit with code 1 and a clear message."""
    # Ultralytics auto-downloads official weights by name. We allow that only for the default,
    # so a typo in --model fails loudly instead of silently fetching another file.
    if path != DEFAULT_MODEL and not Path(path).exists():
        sys.exit(f"Error: model file not found: {path}")
    try:
        model = YOLO(path)
    except Exception as e:
        sys.exit(f"Error: could not load model {path}: {e}")
    if model.task != "pose":
        sys.exit(f"Error: {path} is a '{model.task}' model, expected a pose model such as {DEFAULT_MODEL}.")
    return model


def open_camera(index: int) -> cv2.VideoCapture:
    """Open the camera, or exit with code 1 and a clear message."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"Error: could not open camera {index}. Check the index and that no other app is using it.")
    # On macOS a camera can "open" without permission but then return no frames, so test one read.
    ok, _ = cap.read()
    if not ok:
        cap.release()
        sys.exit(f"Error: camera {index} opened but returned no frame. Check camera permissions "
                 "(System Settings > Privacy & Security > Camera).")
    return cap


def open_outputs(cap: cv2.VideoCapture, video_path: str | None, log_path: str | None) -> OutputManager:
    """Open --save and --log at startup, or exit with code 1 and a clear message."""
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # (width, height), the flip keeps it
    try:
        return OutputManager(video_path, log_path, cap.get(cv2.CAP_PROP_FPS), frame_size)
    except OSError as exc:  # e.g. the folder does not exist or is read-only
        cap.release()  # turn the camera off before leaving
        sys.exit(f"Error: could not open output file: {exc}")


def make_tracker() -> BYTETracker:
    """Create the ByteTrack tracker that gives each person an ID that stays the same between frames.

    We call the tracker ourselves instead of model.track(), because model.track() hides people the
    tracker has not confirmed yet (their first frame), so the face of a new person would leak.
    """
    config = YAML.load(check_yaml("bytetrack.yaml"))  # Ultralytics' default ByteTrack settings
    return BYTETracker(IterableSimpleNamespace(**config))  # the tracker reads its settings as attributes
