"""Real-time pose detection with face anonymization."""

import argparse
import sys
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from stats import FpsMeter, draw_stats, model_ms

DEFAULT_MODEL = "yolo26n-pose.pt"
WINDOW_NAME = "Pose + Face Anonymization"
QUIT_KEYS = {ord("q"), 27}  # 27 is the Esc key code
KPT_CONF = 0.5  # keypoints the model is less sure about are not drawn
KPT_COLOR = (0, 255, 0)  # OpenCV colours are BGR, so this is green
LINE_COLOR = (255, 128, 0)  # light blue
FACE_BOX_COLOR = (0, 0, 255)  # red

# The model returns 17 keypoints per person, always in this order (the COCO convention):
#    0 nose
#    1 left eye        2 right eye
#    3 left ear        4 right ear
#    5 left shoulder   6 right shoulder
#    7 left elbow      8 right elbow
#    9 left wrist     10 right wrist
#   11 left hip       12 right hip
#   13 left knee      14 right knee
#   15 left ankle     16 right ankle
#
# SKELETON: the pairs of keypoints joined by a line
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), #e.g. first parenthesis is nose->left eye
    (3, 5), (4, 6),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
FACE_KPTS = [0, 1, 2, 3, 4]  # nose, left eye, right eye, left ear, right ear
# The face points form a flat strip across the middle of the face, so we pad each side by a fraction
# of the strip's size: a little on the sides (the ears are already near the edge of the head),
# a lot above (forehead and hair) and below (mouth and chin). The result is a tall rectangle.
FACE_PAD_SIDE = 0.15
FACE_PAD_TOP = 0.7
FACE_PAD_BOTTOM = 0.6
# Fallbacks for the head size when only part of the face is visible (Step 11)
EYE_EAR_TO_HEAD = 2.0  # head width ≈ 2 × the distance from an eye to the ear on the same side
ONE_POINT_TO_HEAD = 0.5  # with a single face point, head width ≈ half the person's box width
MIN_FACE_SIZE = 20  # pixels: far-away faces still get a box big enough to hide them
# Blur strength: the face is shrunk to this many pixels across before blurring, whatever its size,
# so near (big) and far (small) faces are equally hidden. Lower = blurrier.
BLUR_CELLS = 12


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
    args = parser.parse_args()
    if not 0 <= args.conf <= 1:
        parser.error(f"--conf must be between 0 and 1, got {args.conf}")  # prints usage and exits with code 2
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


def to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Ultralytics returns tensors (possibly on the GPU); OpenCV needs numpy arrays in CPU memory."""
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else x


# Recap: what the two zips walk over (real values from the bus test image, 4 people).
#
# The outer zip(kpts_xy, kpts_conf) pairs the two arrays along their FIRST dimension only (people i).
# Each turn hands out one row: element 0 goes into this_person_xy, element 1 into this_person_conf.
# Columns j are the 17 keypoints: 0 nose, 1 L eye, 2 R eye, 3 L ear, 4 R ear, ..., 16 R ankle.
#
#   item i | element 0: this_person_xy (17, 2)                     | element 1: this_person_conf (17,)
#   -------+-------------------------------------------------------+-----------------------------------
#     0    | [142,445] [149,437] [133,436] [153,442] ... [74,862]  | 0.99 0.94 1.00 0.16 ... 1.00
#     1    | [789,439] [793,431] [789,430] [802,434] ... [725,840] | 0.26 0.23 0.04 0.28 ... 0.43
#     2    | [295,449] [300,440] [284,442] [307,445] ... [274,826] | 0.99 0.98 1.00 0.20 ... 0.96
#     3    | [8,570]   [4,563]   [7,561]   [0,561]   ... [9,849]   | 0.05 0.03 0.03 0.02 ... 0.16
#
# The inner zip(this_person_xy, this_person_conf) then walks the 17 columns of that row, e.g. row 0:
#
#   j  | (x, y)     | conf | >= KPT_CONF ?
#   ---+------------+------+--------------
#   0  | (142, 445) | 0.99 | draw
#   3  | (153, 442) | 0.16 | skip
#   16 | (74, 862)  | 1.00 | draw
def draw_pose(frame: np.ndarray, kpts_xy: np.ndarray, kpts_conf: np.ndarray) -> None:
    """Draw the skeleton lines and keypoint dots of every person, in place.

    kpts_xy has shape (persons, 17, 2) and kpts_conf has shape (persons, 17).
    """
    for this_person_xy, this_person_conf in zip(kpts_xy, kpts_conf):
        # Lines first, so the dots are drawn on top of them
        for a, b in SKELETON: 
            if this_person_conf[a] >= KPT_CONF and this_person_conf[b] >= KPT_CONF:  # both ends must be confident
                xa, ya = this_person_xy[a]
                xb, yb = this_person_xy[b]
                cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), LINE_COLOR, 2, cv2.LINE_AA) # draw the line
        for (x, y), conf in zip(this_person_xy, this_person_conf):
            if conf >= KPT_CONF:
                cv2.circle(frame, (int(x), int(y)), 4, KPT_COLOR, -1, cv2.LINE_AA) # draw the dots


def face_angle(person_xy: np.ndarray, person_conf: np.ndarray) -> float:
    """How much the head is tilted (0 = straight), read from the eye line, or the ear line if an eye is missing."""
    for a, b in [(1, 2), (3, 4)]:  # try the eyes first, then the ears
        if person_conf[a] >= KPT_CONF and person_conf[b] >= KPT_CONF:
            dx, dy = person_xy[a] - person_xy[b]
            if dx < 0:  # measure left to right, so the box never ends up upside down
                dx, dy = -dx, -dy
            return float(np.arctan2(dy, dx))  # angle of that line
    return 0.0  # no pair visible: assume a straight head


def eye_ear_distance(person_xy: np.ndarray, person_conf: np.ndarray) -> float:
    """Distance from an eye to the ear on the same side (the larger side), or 0 if no such pair is visible.

    Unlike ear-to-ear, this stays large when the head turns to a profile, so it is a good head-size hint.
    """
    dists = [float(np.linalg.norm(person_xy[eye] - person_xy[ear]))
             for eye, ear in [(1, 3), (2, 4)]  # left eye -> left ear, right eye -> right ear
             if person_conf[eye] >= KPT_CONF and person_conf[ear] >= KPT_CONF] # if conf
    return max(dists, default=0.0)


def face_box(person_xy: np.ndarray, person_conf: np.ndarray, person_w: float) -> np.ndarray | None:
    """The 4 corners of a box around the face, tilted with the head. None if no face point is confident.

    person_w is the width of the person's detection box, used when only one face point is visible.
    """
    # Keep the confident face points
    pts = np.array([person_xy[k] for k in FACE_KPTS if person_conf[k] >= KPT_CONF])
    if len(pts) == 0:
        return None

    # "Straighten" the face: express the points in axes that follow the head
    #    (u = along the eyes, v = from forehead to chin)
    angle = face_angle(person_xy, person_conf)
    u = np.array([np.cos(angle), np.sin(angle)])
    v = np.array([-np.sin(angle), np.cos(angle)])
    a, b = pts @ u, pts @ v # project the points (eyes etc...) onto the u,v orthonormal basis

    #In those axes it's the same as before: tightest box around the points, then pad it
    a1, a2, b1, b2 = a.min(), a.max(), b.min(), b.max() # a1 = min along u (usually right ear), a2 = max along u, b1 = min along v (usually eyes (top)), b2 = max along v

    # Head size: take the largest of several estimates, so a partly visible face never gets a too-small box
    size = max(
        a2 - a1, b2 - b1,  # spread of the points (enough for a front view, ear to ear)
        EYE_EAR_TO_HEAD * eye_ear_distance(person_xy, person_conf),  # profile: points bunch up, this doesn't
        MIN_FACE_SIZE,  # tiny far-away faces
    )
    if len(pts) == 1:  # a single point gives no size at all: guess from the person's body
        size = max(size, ONE_POINT_TO_HEAD * person_w)

    # Make the box at least one head wide, centred on the points (in a front view it's already wide enough)
    mid = (a1 + a2) / 2
    half = max((a2 - a1) / 2, size / 2)
    a1 = mid - half - FACE_PAD_SIDE * size # add the padding to the box, so it covers more of the face
    a2 = mid + half + FACE_PAD_SIDE * size
    b1 -= FACE_PAD_TOP * size
    b2 += FACE_PAD_BOTTOM * size

    # bring back the coords to the original axes (frame coords) and return the 4 corners of the box
    corners = [a1 * u + b1 * v, a2 * u + b1 * v, a2 * u + b2 * v, a1 * u + b2 * v]
    return np.array(corners).astype(np.int32)


#  1. straight rectangle        2. blur all of it          3. keep only the tilted part
#  ┌─────────────┐              ┌─────────────┐            ┌─────────────┐
#  │   ╱‾‾‾‾╲    │              │░░░░░░░░░░░░░│            │   ╱░░░░╲    │
#  │  ╱ face ╲   │      →       │░░░░░░░░░░░░░│     →      │  ╱░░░░░░╲   │
#  │  ╲      ╱   │              │░░░░░░░░░░░░░│            │  ╲░░░░░░╱   │
#  │   ╲____╱    │              │░░░░░░░░░░░░░│            │   ╲____╱    │
#  └─────────────┘              └─────────────┘            └─────────────┘
def anonymize(frame: np.ndarray, corners: np.ndarray) -> None:
    """Blur everything inside the (tilted) face box, in place."""
    frame_h, frame_w = frame.shape[:2]

    #cut out the straight rectgle that contains the tilted box, kept inside the image
    x, y, w, h = cv2.boundingRect(corners)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame_w, x + w), min(frame_h, y + h)
    if x2 <= x1 or y2 <= y1:
        return  # the box is completely outside the image
    roi = frame[y1:y2, x1:x2]  # roi is the rectangle cut (straight one)

    #blur that whole rectangle. A direct blur of a big face is very slow (a 500 px face took 200 ms),
    # so we shrink it to a few pixels, blur that tiny image, and stretch it back: same look, ~1 ms.
    roi_h, roi_w = roi.shape[:2]
    scale = min(1.0, BLUR_CELLS / max(roi_w, roi_h))
    small = cv2.resize(roi, (max(1, round(roi_w * scale)), max(1, round(roi_h * scale))),
                       interpolation=cv2.INTER_AREA)  # INTER_AREA averages the pixels it merges: already a blur
    small = cv2.GaussianBlur(small, (3, 3), 0)  # soften a bit more, so no blocks show
    blurred = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)  # stretch back, smoothly

    #Paste back only the pixels inside the tilted box (the rectangle's corners stay sharp)
    mask = np.zeros(roi.shape[:2], np.uint8) # masks with 0s outside the box and 255 inside, same size as roi (:2 takes only h and w coords of the array (3rd coord is color vector))
    cv2.fillPoly(mask, [corners - np.array([x1, y1], np.int32)], 255)  # put the 255, and find relative coords (substrasct the rectangle's top-left corner)
    cv2.copyTo(blurred, mask, roi)  # copy blurred into roi only where mask is 255 (same as roi[mask > 0] = blurred[mask > 0], but ~300x faster)


def main() -> None:
    args = parse_args()
    check_device(args.device)
    model = load_model(args.model)  # before the camera, so a bad model fails without turning it on
    cap = open_camera(args.camera)
    fps_meter = FpsMeter()
    try:
        while True:
            ok, frame = cap.read() # ok is a bool indicating if the frame was read successfully, frame is the image array
            if not ok:
                break
            if args.mirror:
                # Flip left-right like a selfie. Done before the model, so keypoints match what is displayed.
                frame = cv2.flip(frame, 1)  # 1 = around the vertical axis
            # nms=False selects YOLO26's NMS-free head (Ultralytics otherwise still runs NMS by default)
            result = cast(list[Results], model.predict(frame, conf=args.conf, device=args.device, nms=False, verbose=False))[0] # predict returns a list (one result per image), we then cast it to a list of Results (optional). we take the first item as we pass only one frame
            kpts = result.keypoints
            if kpts is not None and kpts.conf is not None and result.boxes is not None:
                kpts_xy, kpts_conf = to_numpy(kpts.xy), to_numpy(kpts.conf)
                persons_w = to_numpy(result.boxes.xywh)[:, 2]  # width of each person's detection box
                for this_person_xy, this_person_conf, person_w in zip(kpts_xy, kpts_conf, persons_w):
                    box = face_box(this_person_xy, this_person_conf, person_w)
                    if box is not None:
                        anonymize(frame, box)
                        # A tilted box is a 4-sided polygon, so polylines instead of rectangle.
                        # OpenCV clips the drawing at the image border by itself.
                        cv2.polylines(frame, [box], True, FACE_BOX_COLOR, 2, cv2.LINE_AA)
                draw_pose(frame, kpts_xy, kpts_conf)  # after the blur, so the skeleton stays visible on top
            draw_stats(frame, fps_meter.tick(), model_ms(result.speed), len(result)) # once per frame
            cv2.imshow(WINDOW_NAME, frame) # display the frame in a window
            if (cv2.waitKey(1) & 0xFF) in QUIT_KEYS: # check if q or esc key is pressed
                break
    finally:
        cap.release() # release the video capture object
        cv2.destroyAllWindows() # destroy all the windows created by OpenCV


if __name__ == "__main__":
    main()
