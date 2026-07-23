# webcam_opencv — OpenCV consumer

Reads the ESP32-S3-CAM MJPEG stream into OpenCV for processing / recording. The ESP32 firmware side lives in [`../CameraWebServer/`](../CameraWebServer); flash that first and get its IP.

## Quick Start

1. Install deps:
   ```bash
   uv pip install opencv-python        # or: pip install opencv-python
   ```
2. Flash the firmware and read the IP from Serial (see [`../README.md`](../README.md)).
3. View the stream:
   ```bash
   python stream_view.py --ip 172.20.10.5
   ```
   Keys: `q` quit, `s` save current frame to `frames/`.

## Code Locations

| Path | What |
|---|---|
| `stream_view.py` | Open the MJPEG stream, display, save frames |
| `frames/` | Saved frames (created on first `s`) |

## Notes

- Stream URL is `http://<ip>:81/stream` (port **81**, path `/stream`). Port 80 is the web UI.
- On an iPhone hotspot, Mac and ESP32 must be on the *same* hotspot; some iOS versions isolate clients.
- Latency/jitter is WiFi-bound; drop frame size or JPEG quality on the firmware for lower latency.

## To-Do

- [ ] Wire frames into the gaze / beamforming demo
- [ ] Add reconnect-on-drop backoff
- [ ] Optional recording to video file
