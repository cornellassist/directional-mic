"""Python replacement for GazeServer.exe.

Runs on the Windows machine with the Tobii tracker attached. Uses the
existing 64-bit/32-bit split (``GazeTracker`` -> ``_listener_win32.py``
over the Windows named pipe) to read gaze samples, and republishes them
as JSON frames ``{"x": float, "y": float, "timestamp": float}`` over a
websocket on ``0.0.0.0:8765`` by default.

This matches the contract consumed by
``directional_mic.gaze_source.WebSocketGazeSource``.

Usage (on the Windows gaze host):

    python gaze_server.py --python-32bit "C:\\path\\to\\py32\\python.exe"

Then, from the beamformer host, point ``--gaze-uri`` at this machine:

    --gaze websocket --gaze-uri ws://<this-host-ip>:8765
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import websockets

from GazeTracker import GazeTracker


async def _broadcast(tracker: GazeTracker, poll_hz: float, clients: set) -> None:
    period = 1.0 / max(poll_hz, 1.0)
    while True:
        points = tracker.get_movement()
        if points:
            x, y = points[-1]
            payload = json.dumps({"x": float(x), "y": float(y),
                                  "timestamp": time.time()})
            dead = []
            for ws in list(clients):
                try:
                    await ws.send(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                clients.discard(ws)
        await asyncio.sleep(period)


async def _serve(host: str, port: int, tracker: GazeTracker,
                 poll_hz: float) -> None:
    clients: set = set()

    async def handler(ws):
        clients.add(ws)
        print(f"[gaze-server] client connected ({len(clients)} total)",
              flush=True)
        try:
            await ws.wait_closed()
        finally:
            clients.discard(ws)
            print(f"[gaze-server] client disconnected ({len(clients)} left)",
                  flush=True)

    async with websockets.serve(handler, host, port):
        print(f"[gaze-server] listening on ws://{host}:{port}", flush=True)
        await _broadcast(tracker, poll_hz, clients)


def main() -> int:
    p = argparse.ArgumentParser(description="Tobii -> WebSocket gaze server.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--poll-hz", type=float, default=60.0,
                   help="How often to drain the pipe and broadcast.")
    p.add_argument("--python-32bit", default=None,
                   help="Path to 32-bit Python interpreter that can import "
                        "TobiiEyeTracker.pyd (overrides the hardcoded "
                        "default in GazeTracker.py).")
    args = p.parse_args()

    tracker_kwargs = {}
    if args.python_32bit:
        tracker_kwargs["python_32bit"] = args.python_32bit
    tracker = GazeTracker(**tracker_kwargs)

    try:
        asyncio.run(_serve(args.host, args.port, tracker, args.poll_hz))
    except KeyboardInterrupt:
        print("[gaze-server] shutting down", flush=True)
    finally:
        tracker.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
