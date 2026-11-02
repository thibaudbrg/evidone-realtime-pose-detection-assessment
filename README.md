# Real-time pose detection with face anonymization

Small Python app that opens a webcam (or a video file), runs YOLO26 pose on each frame, hides the faces using only the pose keypoints, and draws the skeletons. Everything runs on a laptop CPU.

## Setup

Python 3.10+ is needed.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The model weights are not in the repo. With the default `--model`, Ultralytics downloads `yolo26n-pose.pt` the first time you run the app.

## Run

```bash
python app.py
python app.py --camera 1 --conf 0.5
python app.py --video people.mp4 --save out.mp4 --log keypoints.jsonl
```

Keys while it runs:

- `q` or `Esc` : quit (closing the window also works)
- `b` / `p` / `f` : blur, pixelate, solid fill
- `s` : skeleton on/off

Options (see `python app.py --help` for all of them):

| option | default | what it does |
|---|---|---|
| `--camera N` | 0 | camera index |
| `--video PATH` | off | read a video file instead of the camera |
| `--conf X` | 0.4 | person confidence needed to draw the skeleton |
| `--mode` | blur | blur, pixelate or fill |
| `--device` | cpu | cpu, or mps on Apple Silicon |
| `--mirror` / `--no-mirror` | on for camera, off for video | selfie flip |
| `--threaded` / `--no-threaded` | on | read frames in a separate thread |
| `--save PATH` | off | save the anonymized video |
| `--log PATH` | off | one JSON line per frame with the keypoints |
| `--benchmark N` | off | run N frames, print timings and exit |
| `--threads N` | torch default | number of CPU threads for torch |

If the camera doesn't open, the model file is missing, the video can't be read, or `--save`/`--log` can't be written, the app prints an error and exits with code 1.

The files:

- `app.py` main loop
- `cli.py` arguments, loading the model, opening the camera/video/outputs, tracker setup
- `pose.py` skeleton drawing, face box geometry, blur/pixelate/fill
- `smoothing.py` keypoint smoothing (One Euro filter) and the "hold" of face boxes
- `capture.py` the reader thread
- `stats.py` FPS / model time on screen, benchmark numbers
- `output.py` video and JSONL writing

For each frame:

1. read the frame (from the reader thread), flip it if mirror is on
2. `model.predict(...)` with `nms=False`, so YOLO26 uses its NMS-free head
3. give the detections to ByteTrack to get an id per person
4. for each person compute a face box and anonymize it
5. keep hiding the faces of people the model lost for a few frames
6. draw the skeletons (only for persons above `--conf`), then the stats
7. show / save / log

### Face box

The COCO face points are 0 to 4 (nose, eyes, ears). They are a flat line across the middle of the face, so the box is built around them and then padded: a bit on the sides, a lot above (forehead, hair) and below (mouth, chin). The box is rotated with the eye line (or ear line), so a tilted head still gets covered.

Size of the box: I take the biggest of a few estimates, so it doesn't get too small when only part of the face is visible:

- spread of the visible face points
- 2x the eye-to-ear distance (useful in profile, where only one eye and one ear are visible)
- half of the person's box width if only one face point is visible
- a minimum of 20 px for people far away

Fallbacks when the face points are not good enough:

- no confident face point -> estimate the head above the shoulders (using shoulder width and torso length)
- no shoulders either -> hide the top strip of the person's detection box

So any detected person always gets something hidden. This last fallback can be quite big (for example a person cut at the border of the image), I prefer that to missing a face.

Blur and pixelate shrink the face to a fixed number of cells before scaling it back up, so a big close face and a small far one are hidden the same way. Blur is 12 cells + gaussian, pixelate is 6 cells.

### Thresholds

There are a few different thresholds and that's on purpose:

- detection runs at 0.15 (or at `--conf` if that's lower). Everybody above that gets anonymized.
- the skeleton is only drawn for people above `--conf` (0.4 by default).
- face keypoints are used for the face box from 0.3 confidence, but only drawn from 0.5. A rough face position is better than no box.

### Tracking, no flicker

I first used `model.track()`, but it doesn't return a person on the first frame they appear (the tracker waits one frame to confirm a new track). So the face was visible for 1 frame each time someone came in. Now I call `model.predict()` and run ByteTrack myself. Every detection is anonymized, tracked or not. A new person just has id -1 for one frame.

To avoid flicker:

- the keypoints of each tracked person go through a One Euro filter. The box from the smoothed points and the box from the raw points are merged, so smoothing can never uncover the face when someone moves fast.
- when a tracked person is missed by the model, their last face box is still hidden for 10 frames (~1/3 s at 30 fps), and grows 5% per missed frame since they can move.

Privacy trade-off: I chose to hide too much rather than too little. Low detection threshold, low keypoint threshold for the box, big fallback boxes and the hold. The cost is that sometimes a box stays a bit after the person left, or covers more than the face. After 10 missed frames the box is dropped, so if someone is undetected for longer than that (heavy occlusion, very bad light) their face can show. If inference crashes on a frame, a black frame is shown instead of the raw one, and the app stops after 30 failures in a row.

### Threading

`capture.py` reads the camera in a background thread. For a camera it only keeps the newest frame. For a video file it keeps all frames (small queue) so nothing is skipped. Only the reading is in a thread, the display stays in the main loop after anonymization. I didn't want to display frames while inference runs in the background, because then a frame could be shown before its face is hidden.

Results:

| | no thread | thread |
|---|---|---|
| FPS, video file | 17.4 / 18.8 | 17.0 / 18.5 |
| FPS, simulated 30 fps webcam | 16.2 / 18.6 | 18.9 / 18.9 |
| age of the frame when shown (webcam with a 4 frame buffer) | ~224 ms | ~69 ms |

(two runs each, `yolo26n-pose`, 4-core CPU, `--benchmark`)

FPS doesn't really change. Inference is ~50 ms and reading is < 1 ms, so there isn't much to overlap. What changes is the lag. Without the thread the camera buffers frames while we run the model, and we then process old frames, so the image is ~0.2 s late. With the thread we always take the newest frame.

To compare on your machine:

```bash
python app.py --benchmark 300 --no-threaded
python app.py --benchmark 300
```

## AI assistants

I used Claude (Claude Code) during the project: to go through the Ultralytics docs, write parts of the code, review it against the brief, and find bugs. The first-frame face leak with `model.track()` and the held box that kept growing were found during that review. I checked and tested the changes myself and wrote the comments the way I like them.

## What's left / could be better

- No unit tests. I tested by running on webcam and on videos.
- The head fallbacks are simple rules (ratios of shoulder width and box size), they could be tuned on more videos.
- A person missed for more than 10 frames is not hidden anymore.
- Only the nano model was tried. A bigger model would detect better but be slower on CPU.
