"""Real-time pose detection with face anonymization."""

import sys
import time
from typing import cast

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Results

from cli import check_device, load_model, open_camera, parse_args, to_numpy
from output import OutputManager
from pose import (
    FACE_BOX_COLOR,
    anonymize,
    draw_pose,
    face_box,
    fallback_head_box,
    grow_box,
    union_boxes,
)
from smoothing import KeypointSmoother
from stats import BenchmarkStats, FpsMeter, draw_stats, model_ms

WINDOW_NAME = "Pose + Face Anonymization"
QUIT_KEYS = {ord("q"), 27}  # 27 is the Esc key code
PRIVACY_CONF = 0.15  # low threshold: missing a person is worse than an extra anonymization
HOLD_FRAMES = 10  # keep anonymizing a tracked person for about a third of a second
HOLD_GROWTH = 0.05  # grow a held box because the person may move while undetected


def main() -> None:
    args = parse_args()
    check_device(args.device)
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    model = load_model(args.model)  # before the camera, so a bad model fails without turning it on
    cap = open_camera(args.camera)
    fps_meter = FpsMeter()
    mode = args.mode
    show_skeleton = True
    tracked_boxes: dict[int, tuple[np.ndarray, int]] = {} # person_id -> (last face box, number of frames missed since last seen)
    keypoint_smoother = KeypointSmoother()
    benchmark = BenchmarkStats()
    outputs = OutputManager(args.save, args.log_path, cap.get(cv2.CAP_PROP_FPS))
    frame_number = 0

    try:
        while True:
            read_start = time.perf_counter()
            ok, frame = cap.read() # ok is a bool indicating if the frame was read successfully, frame is the image array
            read_seconds = time.perf_counter() - read_start
            if not ok:
                break
            if args.mirror:
                # Flip left-right like a selfie. Done before the model, so keypoints match what is displayed.
                frame = cv2.flip(frame, 1)  # 1 = around the vertical axis

            inference_start = time.perf_counter()
            inference_error: str | None = None
            result: Results | None = None
            try:
                # nms=False selects YOLO26's NMS-free head (Ultralytics otherwise still runs NMS by default)
                # persist=True keeps ByteTrack's person identities between frames.
                result = cast(list[Results], model.track( #track is like predict but keeps id between frame of each person
                    frame,
                    conf=PRIVACY_CONF, # PRIVACY_CONF is 0.15, so detect more ppl for safety (anonymize all above .15, and draw skeleton only for > .4)
                    device=args.device,
                    tracker="bytetrack.yaml",
                    persist=True,
                    nms=False,
                    verbose=False,
                ))[0]  # track returns a list (one result per image), and we pass only one frame
            except Exception as exc:
                inference_error = str(exc)
                print(f"Error: inference failed on frame {frame_number}: {exc}", file=sys.stderr)
            infer_seconds = time.perf_counter() - inference_start
            draw_start = time.perf_counter()

            if result is None:
                # Never display or save a raw frame if inference fails.
                frame = np.zeros_like(frame)
                kpts = None
                kpts_xy: np.ndarray = np.empty((0, 17, 2))
                persons_ids = []
                persons_conf = []
                seen_ids = set()
                person_count = 0
                speed = {}
            else:
                kpts = result.keypoints
                kpts_xy = np.empty((0, 17, 2))
                persons_ids = []
                persons_conf = []
                seen_ids: set[int] = set()
                person_count = len(result)
                speed = result.speed

            smoothed_kpts_xy: list[np.ndarray] = []
            if result is not None and kpts is not None and kpts.conf is not None and result.boxes is not None:
                kpts_xy, kpts_conf = to_numpy(kpts.xy), to_numpy(kpts.conf)
                persons_w = to_numpy(result.boxes.xywh)[:, 2]  # width of each person's detection box
                persons_conf = to_numpy(result.boxes.conf)  # confidence of each person's detection
                persons_ids = (
                    to_numpy(result.boxes.id)
                    if result.boxes.id is not None
                    else np.full(len(kpts_xy), -1)
                )
                for this_person_xy, this_person_conf, person_w, person_id in zip(
                    kpts_xy, kpts_conf, persons_w, persons_ids
                ):
                    track_id = int(person_id)
                    if track_id >= 0:
                        seen_ids.add(track_id)
                    held_box = False
                    smoothed_xy = keypoint_smoother.update(track_id, this_person_xy)
                    smoothed_kpts_xy.append(smoothed_xy)
                    box = face_box(this_person_xy, this_person_conf, person_w) # normal face box, None if no face point is confident
                    if box is None:
                        box = fallback_head_box(this_person_xy, this_person_conf) # fallback head box, None if no shoulder pair is confident
                    smoothed_box = face_box(smoothed_xy, this_person_conf, person_w)
                    if smoothed_box is None:
                        smoothed_box = fallback_head_box(smoothed_xy, this_person_conf)
                    if box is not None and smoothed_box is not None:
                        box = union_boxes(box, smoothed_box)
                    if box is None and track_id in tracked_boxes:
                        previous_box, missed = tracked_boxes[track_id]
                        missed += 1
                        box = grow_box(previous_box, HOLD_GROWTH) # grow the box because the person may move while undetected
                        tracked_boxes[track_id] = (box, missed)
                        held_box = True
                    if box is not None:
                        if track_id >= 0 and not held_box:
                            tracked_boxes[track_id] = (box, 0) # reset the missed counter for this person
                        anonymize(frame, box, mode)
                        # A tilted box is a 4-sided polygon, so polylines instead of rectangle.
                        # OpenCV clips the drawing at the image border by itself.
                        cv2.polylines(frame, [box], True, FACE_BOX_COLOR, 2, cv2.LINE_AA)
                if show_skeleton:
                    draw_pose(frame, np.array(smoothed_kpts_xy), kpts_conf, persons_conf, args.conf) # after anonymization, so the skeleton stays visible on top

            for track_id, (previous_box, missed) in list(tracked_boxes.items()):
                if track_id in seen_ids:
                    continue
                missed += 1
                if missed > HOLD_FRAMES: # if a person has been missed for more than HOLD_FRAMES, we stop tracking them
                    del tracked_boxes[track_id]
                    continue
                box = grow_box(previous_box, HOLD_GROWTH) # keep growing the box for each missed frame, because the person may move while undetected
                tracked_boxes[track_id] = (box, missed)
                anonymize(frame, box, mode)
                cv2.polylines(frame, [box], True, FACE_BOX_COLOR, 2, cv2.LINE_AA)

            keypoint_smoother.remove(seen_ids | set(tracked_boxes))
            draw_stats(frame, fps_meter.tick(), model_ms(speed), person_count) # once per frame
            draw_seconds = time.perf_counter() - draw_start
            persons = [
                {
                    "track_id": int(person_id),
                    "confidence": float(person_conf),
                    "keypoints": person_xy.tolist(),
                }
                for person_id, person_conf, person_xy in zip(persons_ids, persons_conf, kpts_xy)
            ]
            outputs.write(frame_number, frame, inference_error is None, inference_error, persons)
            benchmark.add(read_seconds, infer_seconds, draw_seconds)
            frame_number += 1

            if args.benchmark and benchmark.frames >= args.benchmark:
                break
            cv2.imshow(WINDOW_NAME, frame) # display the frame in a window
            key = cv2.waitKey(1) & 0xFF
            if key in QUIT_KEYS:
                break
            if key == ord("b"):
                mode = "blur"
            elif key == ord("p"):
                mode = "pixelate"
            elif key == ord("f"):
                mode = "fill"
            elif key == ord("s"):
                show_skeleton = not show_skeleton
    finally:
        cap.release() # release the video capture object
        outputs.close()
        cv2.destroyAllWindows() # destroy all the windows created by OpenCV

    if args.benchmark:
        print(benchmark.report())


if __name__ == "__main__":
    main()
