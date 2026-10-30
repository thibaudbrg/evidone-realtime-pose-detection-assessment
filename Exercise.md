# Real-Time Pose Detection with Face Anonymization

## Overview

Build a small Python desktop app that opens a camera and runs a body keypoint detection model on every frame. It blurs or pixelates each person's face and draws their body skeleton on the live video window.

- **Time budget:** Max 4~5 hours. If you are not finished, list what is left in your README.
- **Language:** Python.

```
camera ─► frame ─► YOLO pose ─► keypoints ─► anonymize faces + draw skeleton ─► window
```

## Requirements

The *bonus* items are optional.

1. **Input:** open a camera by index with OpenCV (`cv2.VideoCapture`).
   - *Bonus:* also accept a video file path.
2. **Model:** load a YOLO pose model once, at startup. See the [Ultralytics pose estimation docs](https://docs.ultralytics.com/tasks/pose/).
3. **Per frame:** run the pose model on each frame.
4. **Face anonymization:** for each detected person, use the keypoints to find the face region. The COCO face keypoints are at indices 0–4: nose, eyes and ears.
   - Blur or pixelate that region strongly enough that the face cannot be recognized.
   - Handle the edge cases (person in profile for example)
   - Use only the pose model. Do not add a second model such as a face detector.
5. **Drawing:** don't use Ultralytics' `results.plot()`. Write your own drawing code that shows the keypoints and skeleton lines for each person.
6. **Command line:** the camera index and the confidence threshold can be set from the command line, and each has a sensible default.
7. *Bonus:* **On-screen stats:** show the current FPS and the model's inference time in milliseconds.
8. **Exit:** `q` or `Esc` quits. The app releases the camera and closes its windows cleanly.
9. **Errors:** if the camera will not open or the model file is missing, print a clear message and exit with a non-zero code

Example usage (your script name may differ):

```bash
python app.py --camera 0 --conf 0.4
```

## Technical setup

### Model

Use a YOLO26 pose model, for example `yolo26n-pose.pt`. YOLO26 gives final detections with no non-maximum suppression (NMS) step. An older version such as YOLO11 is also fine, as long as NMS is applied to its output.

### Python

- Python 3.10+
- Packages `ultralytics` and `opencv-python` for model and camera

### Hardware

- The app must run on a laptop CPU. Using a GPU is optional.
- You need a webcam. A laptop's built-in camera is fine. Otherwise you can use a video with moving people.

### AI coding assistants

You may use AI coding assistants without restrictions. Describe how you used them in your README.

## Bonus goals (optional)

You can try to implement one or a few of these only after the core requirements work, if you have still have time.

- **No flicker:** keep faces hidden when the model misses a person for a frame or two, for example by tracking people or smoothing the face region over time. Explain the privacy trade-off you chose.
- **Concurrency:** read frames from the camera in a separate thread from inference, so the display stays smooth. Measure FPS before and after, and report both numbers.
- **Modes:** switch between blur, pixelate and solid fill from the command line or the keyboard, and toggle the skeleton on and off.
- **Recording:** save the anonymized video, or write a per-frame JSON log of the keypoints.

## Deliverables

- Send a link to a Git repository with your implementation.
- Add a README.md in the repo that covers your implementation.
- Do not commit the model weights.

## Submission and questions

- **Deadline:** within 7 days of receiving this brief.

If something is unclear, make a reasonable assumption and write it down in the README.
