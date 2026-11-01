"""Real-time pose detection with face anonymization."""

import sys
import time

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from cli import check_device, load_model, make_tracker, open_camera, open_outputs, parse_args
from pose import draw_pose, hide_face, privacy_box
from smoothing import FaceHold, KeypointSmoother
from stats import BenchmarkStats, FpsMeter, draw_stats, model_ms

WINDOW_NAME = "Pose + Face Anonymization"
QUIT_KEYS = {ord("q"), 27}  # 27 is the Esc key code
MODE_KEYS = {ord("b"): "blur", ord("p"): "pixelate", ord("f"): "fill"}  # key -> anonymization mode
PRIVACY_CONF = 0.15  # low threshold: missing a person is worse than an extra anonymization
MAX_FAILED_FRAMES = 30  # stop after about one second of inference errors in a row


def track_ids(tracker: BYTETracker, boxes: Boxes, frame: np.ndarray) -> np.ndarray:
    """Give each detected person its tracker ID, or -1 if the tracker is not following them (yet)."""
    ids = np.full(len(boxes), -1)  # -1 = new person not confirmed yet, or confidence too low to start a track
    tracks = tracker.update(boxes, frame)  # one row per tracked person: [x1, y1, x2, y2, track_id, score, cls, detection_index]
    if len(tracks):
        ids[tracks[:, 7].astype(int)] = tracks[:, 4].astype(int)  # put each track ID at the index of its detection
    return ids


def main() -> None:
    args = parse_args()
    check_device(args.device)
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    model = load_model(args.model)  # before the camera, so a bad model fails without turning it on
    cap = open_camera(args.camera)
    outputs = open_outputs(cap, args.save, args.log_path)  # exits with a clear message if --save or --log can't be written
    tracker = make_tracker()
    detect_conf = min(PRIVACY_CONF, args.conf)  # detect at the lower threshold, so --conf below 0.15 still finds those people
    fps_meter = FpsMeter()
    keypoint_smoother = KeypointSmoother()
    face_hold = FaceHold()  # keeps faces hidden for a few frames when the model misses someone
    benchmark = BenchmarkStats()
    mode = args.mode
    show_skeleton = True
    frame_number = 0
    failed_frames = 0  # inference errors in a row

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
            try:
                result = model.predict(  # predict, not track: every detection comes back, even a person seen for the first time
                    frame,
                    conf=detect_conf,  # anonymize everyone above 0.15, draw the skeleton only above --conf
                    device=args.device,
                    nms=False,  # selects YOLO26's NMS-free head (Ultralytics otherwise still runs NMS by default)
                    verbose=False,
                )[0]  # predict returns a list (one result per image), and we pass only one frame
                boxes = result.boxes.cpu().numpy()  # OpenCV and the tracker need numpy arrays in CPU memory
                ids = track_ids(tracker, boxes, frame)
                failed_frames = 0
            except Exception as exc:
                result = None
                inference_error = str(exc)
                failed_frames += 1
                print(f"Error: inference failed on frame {frame_number}: {exc}", file=sys.stderr)
                if failed_frames >= MAX_FAILED_FRAMES:
                    sys.exit(f"Error: inference failed {failed_frames} frames in a row, stopping.")  # finally still releases the camera
            infer_seconds = time.perf_counter() - inference_start
            draw_start = time.perf_counter()

            persons = []  # what goes into the --log file for this frame
            if result is None:
                frame = np.zeros_like(frame)  # never display or save a raw frame if inference fails
                speed = {}
            else:
                speed = result.speed
                kpts = result.keypoints.cpu().numpy()
                kpts_xy = kpts.xy  # (people, 17, 2): x, y of each keypoint
                kpts_conf = kpts.conf if kpts.conf is not None else np.ones(kpts_xy.shape[:2])  # (people, 17); a model without keypoint confidence counts every point as sure
                smoothed_xy = np.array(
                    [keypoint_smoother.update(int(i), xy) for i, xy in zip(ids, kpts_xy, strict=True)]
                ).reshape(-1, 17, 2)  # reshape keeps the (people, 17, 2) shape when nobody is detected

                for raw_xy, smooth_xy, conf, person_xyxy, track_id, person_conf in zip(
                    kpts_xy, smoothed_xy, kpts_conf, boxes.xyxy, ids, boxes.conf, strict=True
                ):
                    box = privacy_box(raw_xy, smooth_xy, conf, person_xyxy)  # face points, else shoulders, else top of the person's box
                    face_hold.seen(int(track_id), box)  # remember it in case the model misses this person next frame
                    hide_face(frame, box, mode)
                    persons.append({"track_id": int(track_id), "confidence": float(person_conf), "keypoints": raw_xy.tolist()})

                seen_ids = {int(i) for i in ids if i >= 0}  # people the tracker found in this frame
                for box in face_hold.missing_boxes(seen_ids):  # people missed for a frame or two keep their face hidden
                    hide_face(frame, box, mode)
                keypoint_smoother.remove(seen_ids | face_hold.ids)  # forget the smoothing of people who are gone

                if show_skeleton:
                    draw_pose(frame, smoothed_xy, kpts_conf, boxes.conf, args.conf)  # after anonymization, so the skeleton stays visible on top

            draw_stats(frame, fps_meter.tick(), model_ms(speed), len(persons)) # once per frame
            draw_seconds = time.perf_counter() - draw_start
            outputs.write(frame_number, frame, inference_error is None, inference_error, persons)
            benchmark.add(read_seconds, infer_seconds, draw_seconds)
            frame_number += 1

            if args.benchmark and benchmark.frames >= args.benchmark:
                break
            cv2.imshow(WINDOW_NAME, frame) # display the frame in a window
            key = cv2.waitKey(1) & 0xFF
            if key in QUIT_KEYS or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:  # < 1 = window closed with its X button
                break
            mode = MODE_KEYS.get(key, mode)  # b, p or f switch the mode; any other key keeps it
            if key == ord("s"):
                show_skeleton = not show_skeleton
    finally:
        cap.release() # release the video capture object
        outputs.close()
        cv2.destroyAllWindows() # destroy all the windows created by OpenCV

    if args.benchmark:
        print(benchmark.report())


if __name__ == "__main__":
    main()
