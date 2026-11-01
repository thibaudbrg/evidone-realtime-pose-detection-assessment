"""Pose drawing, head-box geometry, and face anonymization."""

import cv2
import numpy as np

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
PIXELATE_CELLS = 6


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
def draw_pose(
    frame: np.ndarray,
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    person_conf: np.ndarray | None = None,
    person_threshold: float | None = None,
) -> None:
    """Draw the skeleton lines and keypoint dots of every person, in place."""
    people = zip(kpts_xy, kpts_conf) if person_conf is None or person_threshold is None else (
        (person_xy, keypoint_conf)
        for person_xy, keypoint_conf, confidence in zip(kpts_xy, kpts_conf, person_conf)
        if confidence >= person_threshold
    )
    for this_person_xy, this_person_conf in people:
        # Lines first, so the dots are drawn on top of them
        for a, b in SKELETON:
            if this_person_conf[a] >= KPT_CONF and this_person_conf[b] >= KPT_CONF:
                xa, ya = this_person_xy[a]
                xb, yb = this_person_xy[b]
                cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), LINE_COLOR, 2, cv2.LINE_AA)
        for (x, y), conf in zip(this_person_xy, this_person_conf):
            if conf >= KPT_CONF:
                cv2.circle(frame, (int(x), int(y)), 4, KPT_COLOR, -1, cv2.LINE_AA)


def face_angle(person_xy: np.ndarray, person_conf: np.ndarray) -> float:
    """How much the head is tilted, read from the eye line or ear line."""
    for a, b in [(1, 2), (3, 4)]:
        if person_conf[a] >= KPT_CONF and person_conf[b] >= KPT_CONF:
            dx, dy = person_xy[a] - person_xy[b]
            if dx < 0:
                dx, dy = -dx, -dy
            return float(np.arctan2(dy, dx))
    return 0.0


def eye_ear_distance(person_xy: np.ndarray, person_conf: np.ndarray) -> float:
    """Return the largest confident eye-to-ear distance."""
    dists = [float(np.linalg.norm(person_xy[eye] - person_xy[ear]))
             for eye, ear in [(1, 3), (2, 4)]
             if person_conf[eye] >= KPT_CONF and person_conf[ear] >= KPT_CONF]
    return max(dists, default=0.0)


def face_box(person_xy: np.ndarray, person_conf: np.ndarray, person_w: float) -> np.ndarray | None:
    """Return a tilted face box, or None if no face point is confident."""
    pts = np.array([person_xy[k] for k in FACE_KPTS if person_conf[k] >= KPT_CONF])
    if len(pts) == 0:
        return None

    angle = face_angle(person_xy, person_conf)
    u = np.array([np.cos(angle), np.sin(angle)])
    v = np.array([-np.sin(angle), np.cos(angle)])
    a, b = pts @ u, pts @ v
    a1, a2, b1, b2 = a.min(), a.max(), b.min(), b.max()

    # Head size: take the largest estimate so a partly visible face never gets too small.
    size = max(
        a2 - a1, b2 - b1,
        EYE_EAR_TO_HEAD * eye_ear_distance(person_xy, person_conf),
        MIN_FACE_SIZE,
    )
    if len(pts) == 1:
        size = max(size, ONE_POINT_TO_HEAD * person_w)

    mid = (a1 + a2) / 2
    half = max((a2 - a1) / 2, size / 2)
    a1 = mid - half - FACE_PAD_SIDE * size
    a2 = mid + half + FACE_PAD_SIDE * size
    b1 -= FACE_PAD_TOP * size
    b2 += FACE_PAD_BOTTOM * size
    corners = [a1 * u + b1 * v, a2 * u + b1 * v, a2 * u + b2 * v, a1 * u + b2 * v]
    return np.array(corners).astype(np.int32)


def fallback_head_box(person_xy: np.ndarray, person_conf: np.ndarray) -> np.ndarray | None:
    """Estimate a head box above the shoulders when no face point is confident."""
    left_shoulder, right_shoulder = 5, 6
    if person_conf[left_shoulder] < KPT_CONF or person_conf[right_shoulder] < KPT_CONF:
        return None
    shoulder_mid = (person_xy[left_shoulder] + person_xy[right_shoulder]) / 2
    shoulder_vector = person_xy[right_shoulder] - person_xy[left_shoulder]
    shoulder_width = float(np.linalg.norm(shoulder_vector))
    if shoulder_width <= 0:
        return None

    # The perpendicular points along the person's upward direction in image coordinates.
    u = shoulder_vector / shoulder_width
    v = np.array([-u[1], u[0]])
    if v[1] > 0:
        v = -v
    head_width = max(0.65 * shoulder_width, MIN_FACE_SIZE)
    head_height = max(0.9 * shoulder_width, MIN_FACE_SIZE)
    if person_conf[11] >= KPT_CONF and person_conf[12] >= KPT_CONF:
        hip_mid = (person_xy[11] + person_xy[12]) / 2
        torso_length = float(np.linalg.norm(hip_mid - shoulder_mid))
        head_width = max(head_width, 0.35 * torso_length)
        head_height = max(head_height, 0.5 * torso_length)
    center = shoulder_mid + v * (0.55 * head_height)
    half_width, half_height = head_width / 2, head_height / 2
    corners = [
        center - half_width * u - half_height * v,
        center + half_width * u - half_height * v,
        center + half_width * u + half_height * v,
        center - half_width * u + half_height * v,
    ]
    return np.array(corners).astype(np.int32)


def grow_box(corners: np.ndarray, factor: float) -> np.ndarray:
    """Grow a polygon around its centre, keeping a held privacy box conservative."""
    center = corners.mean(axis=0)
    grown = center + (corners - center) * (1 + factor)
    # Round away from the centre so small boxes still grow by at least a pixel.
    return np.where(grown < center, np.floor(grown), np.ceil(grown)).astype(np.int32)


def union_boxes(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return a polygon covering both boxes, so smoothing never reduces privacy coverage."""
    points = np.vstack((first, second)).astype(np.int32)
    return cv2.convexHull(points).reshape(-1, 2)


def anonymize(frame: np.ndarray, corners: np.ndarray, mode: str) -> None:
    """Anonymize everything inside the (tilted) face box, in place."""
    frame_h, frame_w = frame.shape[:2]

    # Cut out the straight rectangle that contains the tilted box, kept inside the image.
    x, y, w, h = cv2.boundingRect(corners)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame_w, x + w), min(frame_h, y + h)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]

    roi_h, roi_w = roi.shape[:2]
    if mode == "fill":
        anonymized = np.zeros_like(roi)
    else:
        cells = BLUR_CELLS if mode == "blur" else PIXELATE_CELLS
        scale = min(1.0, cells / max(roi_w, roi_h))
        small = cv2.resize(
            roi,
            (max(1, round(roi_w * scale)), max(1, round(roi_h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        if mode == "blur":
            small = cv2.GaussianBlur(small, (3, 3), 0)
            interpolation = cv2.INTER_LINEAR
        else:
            interpolation = cv2.INTER_NEAREST
        anonymized = cv2.resize(small, (roi_w, roi_h), interpolation=interpolation)

    # Paste back only the pixels inside the tilted box; the rectangle's corners stay sharp.
    mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.fillPoly(mask, [corners - np.array([x1, y1], np.int32)], 255)
    cv2.copyTo(anonymized, mask, roi)
