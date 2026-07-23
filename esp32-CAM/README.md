# ESP32-S3-CAM

Turn a **GOOUUU ESP32-S3-CAM** (ESP32-S3-WROOM-1 N16R8 + OV2640) into a WiFi webcam and consume its video in OpenCV for the directional-mic demo.

Two parts:

| Folder | Side | What |
|---|---|---|
| [`CameraWebServer/`](CameraWebServer) | ESP32 firmware | Streams MJPEG over WiFi (`http://<ip>:81/stream`) |
| [`webcam_opencv/`](webcam_opencv) | Python / OpenCV | Reads the stream, displays and saves frames |

## Quick Start

1. Set WiFi credentials in `CameraWebServer/CameraWebServer.ino` (lines 13–14).
2. Apply the **Board Settings** below in Arduino IDE (these are the whole battle — get them exact).
3. Enter download mode (**hold BOOT → tap RESET → release BOOT**) and **Upload**.
4. Open **Serial Monitor @ 115200**, tap RESET, copy the IP: `Camera Ready! Use 'http://<ip>'`.
5. Consume it in OpenCV — see [`webcam_opencv/`](webcam_opencv):
   ```bash
   cd webcam_opencv && python stream_view.py --ip <ip>
   ```

## Board Settings (Arduino IDE → Tools)

| Setting | Value |
|---|---|
| Board | ESP32S3 Dev Module (**not** ESP32-S3-USB-OTG — it hides PSRAM/Flash) |
| PSRAM | **OPI PSRAM** (N16R8 = octal; wrong value boot-loops) |
| Flash Size | 16MB (128Mb) |
| Partition Scheme | Huge APP (3MB No OTA/1MB SPIFFS) |
| USB CDC On Boot | Enabled (native USB port) |
| Port | `/dev/cu.usbmodem*` — renames between run/download mode, re-check each time |

Camera model in `CameraWebServer/board_config.h` must be `CAMERA_MODEL_ESP32S3_EYE` — the GOOUUU board shares that exact pinout.

## To-Do

- [ ] Move WiFi credentials out of the sketch (avoid committing hotspot password)
- [ ] Wire the stream into the gaze / beamforming demo
- [ ] Tune frame size / JPEG quality for latency vs. resolution

## Reference

- [GOOUUU ESP32-S3-CAM pinout](https://github.com/profharris/GOOUUU_ESP32-S3-CAM)
- [arduino-esp32 CameraWebServer discussion](https://github.com/espressif/arduino-esp32/discussions/9249)
- [Changing camera video settings (frame size / quality)](https://youtu.be/O3q-6ga4zlA?t=216)
- [Removing the IR filter from the camera module](https://www.youtube.com/watch?v=mRSLSeX3omA)
- Stream endpoints: `http://<ip>/` (web UI), `http://<ip>:81/stream` (raw MJPEG)
