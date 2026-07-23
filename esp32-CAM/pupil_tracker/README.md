# pupil_tracker — ESP32-S3-CAM eye tracking

Turns the ESP32-S3-CAM (mounted as a near-eye camera, not a room-facing one)
into a gaze source: detects the pupil in each frame, calibrates pupil
position against on-screen look points, and republishes normalized gaze
coordinates over the same websocket contract used elsewhere in this repo
(`eye_tracking/gaze_server.py`, `GazeServer.exe`,
`directional_mic.gaze_source.WebSocketGazeSource`).

The pupil-detection algorithm (crop, darkest-region search, adaptive
threshold, contour filtering, ellipse fit) is ported from
[JEOresearch/EyeTracker](https://github.com/JEOresearch/EyeTracker)
(`OrloskyPupilDetectorLite.py`, MIT licensed — see [`LICENSE`](LICENSE)),
stripped of its file-picker/video-writer scaffolding.

## How it differs from the other module in `webcam_opencv/`

`../webcam_opencv/` treats the ESP32-S3-CAM as a normal outward-facing
webcam. This module assumes the *same firmware* but a camera **mounted to
look at the user's eye** — think a cheap VR/AR eye-tracking rig, not a
webcam on a desk. Framing, focus distance, and (usually) IR illumination
all need to be set up for that use case; see Hardware notes below.

## Pipeline

```
ESP32-S3-CAM (CameraWebServer firmware)
  --MJPEG stream (:81/stream)-->
pupil_detector.detect_pupil()      # pupil ellipse in eye-image pixel coords
  --normalized pupil position-->
calibration.Calibration.apply()    # polynomial fit -> normalized screen coords
  --{"x", "y", "timestamp"} JSON-->
websocket :8765
  --> directional_mic.gaze_source.WebSocketGazeSource (or any client)
```

## Quick start

1. Flash and boot the ESP32-S3-CAM per [`../README.md`](../README.md); note its IP.
2. Install deps:
   ```bash
   uv pip install opencv-python numpy websockets
   ```
3. Optional: sanity-check the detector against your laptop webcam before
   touching the ESP32 at all — see [Testing without hardware](#testing-without-hardware).
4. Mount the camera facing one eye (see Hardware notes) and calibrate —
   look at each red dot as it appears:
   ```bash
   cd esp32-CAM/pupil_tracker
   python esp32cam_gaze_server.py --ip 172.20.10.5 --calibrate
   ```
   This writes `calibration.json` in the current directory. Re-run whenever
   the camera is remounted or slips — the mapping is specific to the exact
   camera/eye geometry, not just the user.
4. Run the server:
   ```bash
   python esp32cam_gaze_server.py --ip 172.20.10.5
   ```
   Then point a consumer at it, e.g. the beamformer:
   ```bash
   python -m directional_mic.runtime --gaze websocket --gaze-uri ws://localhost:8765 ...
   ```

Pass `--no-display` to skip the live debug overlay window (ellipse + center
dot drawn on the eye-camera feed), or `--port` / `--host` to change the
websocket bind address.

## Testing without hardware

`test_webcam.py` runs `detect_pupil()` against your laptop's built-in/USB
webcam instead of the ESP32 stream — no board, no calibration, just a live
window with the fitted ellipse overlaid, for checking the detector itself
works before dealing with the near-eye mount:

```bash
python test_webcam.py            # cv2 camera index 0
python test_webcam.py --camera 1 # external webcam
```

Hold the webcam close to one eye (a few inches) — at normal laptop
distance the "darkest region" heuristic will lock onto eyebrows, nostrils,
or background instead of the pupil, since it isn't a near-eye mount. Press
`q` to quit.

## Hardware notes

- **This is a near-eye camera, not a room camera.** Mount it close enough
  that one eye fills most of the frame — a few centimeters, typically on a
  headband/glasses rig, not on a desk.
- **IR illumination matters.** The pupil detector works by finding the
  darkest region of the frame under the assumption that's the pupil; visible-light
  hot spots (glare, uneven lighting) confuse it. JEOresearch's reference
  hardware uses an IR-pass camera + IR LED so the pupil reads as
  uniformly dark regardless of iris color or ambient light. The OV3660 on
  this board ships with a visible-light-only lens by default; see
  [`../README.md`](../README.md)'s "Removing the IR filter from the camera
  module" reference if you need to convert it. Until then, expect the
  detector to work best on darker irises under flat, glare-free lighting.
- **Focus.** Most of these clone boards have a manually-adjustable lens
  ring — refocus it for the few-centimeter working distance; the factory
  focus is set for room-scale distances.

## Calibration model

Pupil position in the eye image is *not* gaze direction on screen — the
mapping between them depends on eye/camera/screen geometry and needs a
per-mount calibration pass. `calibration.py` fits a 2nd-order polynomial
(6 terms per axis: `1, x, y, xy, x², y²`) from pupil-normalized coordinates
to screen-normalized coordinates using a 9-point grid (`DEFAULT_GRID`),
least-squares. This is the same class of technique used by budget webcam
eye trackers — good enough for coarse gaze-steering, not diagnostic-grade.

`Calibration.save()` / `Calibration.load()` round-trip to JSON
(`coeffs_x`, `coeffs_y`), so a calibration only needs to be redone when the
physical mount changes.

## Code locations

| Path | What |
|---|---|
| `pupil_detector.py` | Ported pupil-ellipse detection (`detect_pupil()`, `draw_debug_overlay()`) |
| `calibration.py` | Pupil -> screen mapping (`fit()`, `Calibration`, `run_calibration()`) |
| `esp32cam_gaze_server.py` | CLI: connects to the stream, runs detection + calibration, serves the websocket |
| `test_webcam.py` | CLI: runs the detector against a local webcam, no ESP32/calibration needed |
| `LICENSE` | MIT license from JEOresearch/EyeTracker, covering the ported detection code |

## Limitations

- Monocular (one eye) — no vergence/depth.
- No blink detection: a closed eye or lost pupil currently just stops
  updates (last-known position stays live on the consumer side) rather than
  being explicitly reported.
- Calibration is per-mount, not just per-user; assume it needs to be redone
  after adjusting the rig.
- No head-pose compensation — the mapping degrades if the headset/rig
  shifts after calibrating.

## To-Do

- [ ] Explicit blink / eye-lost signal instead of silent staleness
- [ ] Validate against a real IR-illuminated mount (current default assumes
      visible light, darkest-pixel heuristic)
- [ ] Tune `detect_pupil()`'s `added_threshold` / `mask_size` defaults once
      tested against real footage
