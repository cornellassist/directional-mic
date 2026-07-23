"""ESP32-S3-CAM -> pupil detection -> screen-gaze -> WebSocket bridge.

Reads the MJPEG stream from the firmware in `../CameraWebServer` (see
`../README.md`), detects the pupil ellipse in each frame with
`pupil_detector` (ported from JEOresearch/EyeTracker -- see LICENSE),
maps the pupil position to normalized screen coordinates via a saved
`Calibration`, and republishes it as JSON frames `{"x": float, "y":
float, "timestamp": float}` over a websocket. This matches the
contract used by `eye_tracking/gaze_server.py` and `GazeServer.exe`
(see `directional_mic.gaze_source.WebSocketGazeSource`), so this is a
drop-in gaze source:

    --gaze websocket --gaze-uri ws://<this-host-ip>:8765

Usage:
    # one-time calibration -- look at each dot as it appears (writes calibration.json)
    python esp32cam_gaze_server.py --ip 172.20.10.5 --calibrate

    # run the server using a saved calibration
    python esp32cam_gaze_server.py --ip 172.20.10.5

Keys while the server window is focused: 'q' quits.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import cv2
import websockets

from calibration import Calibration, run_calibration
from pupil_detector import detect_pupil, draw_debug_overlay


def build_stream_url(ip: str) -> str:
    return f"http://{ip}:81/stream"


async def _broadcast(cap: cv2.VideoCapture, calib: Calibration, poll_hz: float,
                     clients: set, show_debug: bool) -> None:
    period = 1.0 / max(poll_hz, 1.0)
    while True:
        ok, frame = cap.read()
        if not ok:
            await asyncio.sleep(0.05)
            continue

        result = detect_pupil(frame)
        if show_debug:
            cv2.imshow("esp32cam-pupil", draw_debug_overlay(result))
            cv2.waitKey(1)

        if result.found and clients:
            sx, sy = calib.apply(*result.center_norm)
            payload = json.dumps({"x": sx, "y": sy, "timestamp": time.time()})
            dead = []
            for ws in list(clients):
                try:
                    await ws.send(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                clients.discard(ws)

        await asyncio.sleep(period)


async def _serve(host: str, port: int, cap: cv2.VideoCapture, calib: Calibration,
                 poll_hz: float, show_debug: bool) -> None:
    clients: set = set()

    async def handler(ws):
        clients.add(ws)
        print(f"[esp32cam-gaze] client connected ({len(clients)} total)", flush=True)
        try:
            await ws.wait_closed()
        finally:
            clients.discard(ws)
            print(f"[esp32cam-gaze] client disconnected ({len(clients)} left)", flush=True)

    async with websockets.serve(handler, host, port):
        print(f"[esp32cam-gaze] listening on ws://{host}:{port}", flush=True)
        await _broadcast(cap, calib, poll_hz, clients, show_debug)


def main() -> int:
    p = argparse.ArgumentParser(description="ESP32-S3-CAM pupil tracker -> WebSocket gaze server.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ip", help="ESP32-CAM IP, e.g. 172.20.10.5")
    src.add_argument("--url", help="full MJPEG stream URL")
    p.add_argument("--calib", default="calibration.json", help="calibration file to load/save")
    p.add_argument("--calibrate", action="store_true", help="run interactive calibration and exit")
    p.add_argument("--screen-w", type=int, default=1280, help="calibration window width (px)")
    p.add_argument("--screen-h", type=int, default=720, help="calibration window height (px)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--poll-hz", type=float, default=30.0)
    p.add_argument("--no-display", action="store_true", help="don't show the debug overlay window")
    args = p.parse_args()

    url = args.url or build_stream_url(args.ip)
    print(f"[esp32cam-gaze] connecting to {url} ...", flush=True)
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open stream {url}\n"
            "Check: board is powered, on the same network, and the IP is correct."
        )

    try:
        if args.calibrate:
            calib = run_calibration(cap, screen_size=(args.screen_w, args.screen_h))
            calib.save(args.calib)
            print(f"[esp32cam-gaze] wrote calibration to {args.calib}", flush=True)
            return 0

        try:
            calib = Calibration.load(args.calib)
        except FileNotFoundError:
            raise SystemExit(
                f"no calibration file at {args.calib!r} -- run with --calibrate first"
            )

        try:
            asyncio.run(_serve(args.host, args.port, cap, calib, args.poll_hz,
                               show_debug=not args.no_display))
        except KeyboardInterrupt:
            print("[esp32cam-gaze] shutting down", flush=True)
        return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
