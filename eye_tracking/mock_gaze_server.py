"""Simulated gaze websocket server.

Serves fake gaze samples over the same JSON contract as
``GazeServer.exe`` / ``gaze_server.py``, so downstream consumers can be
exercised through ``--gaze websocket --gaze-uri ws://...`` without a
real Tobii tracker.

Runs anywhere (no pywin32, no Tobii SDK). Patterns:

* ``center``  — always (0.5, 0.5)
* ``sweep``   — x oscillates 0..1 over ``--period-s``; y = 0.5
* ``circle``  — (x, y) trace a unit circle around (0.5, 0.5) with
                radius 0.35 over ``--period-s``
* ``mouse``   — cursor position normalized to [0, 1] on the primary
                monitor (requires ``pynput``: ``pip install pynput``)
* ``file:path.csv`` — replay ``x,y`` rows at ``--rate-hz``, looping

Usage:

    python eye_tracking/mock_gaze_server.py --pattern sweep --period-s 4
    python eye_tracking/mock_gaze_server.py --pattern mouse

Then point the beamformer at it:

    --gaze websocket --gaze-uri ws://localhost:8765
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import time

import websockets


def _make_mouse_reader() -> callable:
    """Return a callable ``() -> (x, y)`` giving cursor position in [0, 1].

    Uses ``pynput.mouse`` for the cursor and ``tkinter`` for screen size
    (both in stdlib / light deps). Falls back to full-screen heuristics
    if tkinter isn't available.
    """
    from pynput.mouse import Controller  # noqa: PLC0415
    try:
        import tkinter  # noqa: PLC0415
        root = tkinter.Tk()
        root.withdraw()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()
    except Exception:
        screen_w, screen_h = 1920, 1080
    ctrl = Controller()
    inv_w = 1.0 / max(screen_w, 1)
    inv_h = 1.0 / max(screen_h, 1)

    def read() -> tuple[float, float]:
        px, py = ctrl.position
        x = max(0.0, min(1.0, px * inv_w))
        y = max(0.0, min(1.0, py * inv_h))
        return x, y

    print(f"[mock-gaze] mouse mode: screen={screen_w}x{screen_h}", flush=True)
    return read


def _load_csv(path: str) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"no numeric (x,y) rows in {path}")
    return rows


def _sample(pattern: str, t: float, period_s: float,
            csv_rows: list[tuple[float, float]] | None,
            csv_index: int,
            mouse_reader=None) -> tuple[float, float, int]:
    if pattern == "center":
        return 0.5, 0.5, csv_index
    if pattern == "sweep":
        phase = (t % period_s) / period_s
        return 0.5 + 0.5 * math.sin(2 * math.pi * phase), 0.5, csv_index
    if pattern == "circle":
        phase = (t % period_s) / period_s * 2 * math.pi
        return (0.5 + 0.35 * math.cos(phase),
                0.5 + 0.35 * math.sin(phase), csv_index)
    if pattern == "mouse" and mouse_reader is not None:
        x, y = mouse_reader()
        return x, y, csv_index
    if csv_rows is not None:
        x, y = csv_rows[csv_index % len(csv_rows)]
        return x, y, csv_index + 1
    return 0.5, 0.5, csv_index


async def _broadcast(clients: set, args, csv_rows, mouse_reader=None) -> None:
    period = 1.0 / max(args.rate_hz, 1.0)
    t0 = time.monotonic()
    csv_index = 0
    while True:
        t = time.monotonic() - t0
        x, y, csv_index = _sample(args.pattern, t, args.period_s,
                                   csv_rows, csv_index, mouse_reader)
        if clients:
            payload = json.dumps({"x": x, "y": y, "timestamp": time.time()})
            dead = []
            for ws in list(clients):
                try:
                    await ws.send(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                clients.discard(ws)
        await asyncio.sleep(period)


async def _serve(args, csv_rows, mouse_reader=None) -> None:
    clients: set = set()

    async def handler(ws):
        clients.add(ws)
        print(f"[mock-gaze] client connected ({len(clients)} total)",
              flush=True)
        try:
            await ws.wait_closed()
        finally:
            clients.discard(ws)
            print(f"[mock-gaze] client disconnected ({len(clients)} left)",
                  flush=True)

    async with websockets.serve(handler, args.host, args.port):
        print(f"[mock-gaze] listening on ws://{args.host}:{args.port} "
              f"pattern={args.pattern} rate={args.rate_hz}Hz", flush=True)
        await _broadcast(clients, args, csv_rows, mouse_reader)


def main() -> int:
    p = argparse.ArgumentParser(description="Simulated gaze websocket server.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--pattern", default="sweep",
                   help="center | sweep | circle | file:path.csv")
    p.add_argument("--period-s", type=float, default=4.0,
                   help="Period for sweep/circle patterns.")
    p.add_argument("--rate-hz", type=float, default=60.0)
    args = p.parse_args()

    logging.getLogger("websockets.server").setLevel(logging.CRITICAL)
    logging.getLogger("websockets").setLevel(logging.CRITICAL)

    csv_rows = None
    mouse_reader = None
    if args.pattern.startswith("file:"):
        csv_rows = _load_csv(args.pattern[len("file:"):])
    elif args.pattern == "mouse":
        try:
            mouse_reader = _make_mouse_reader()
        except ImportError:
            raise SystemExit(
                "[mock-gaze] --pattern mouse requires pynput "
                "(`pip install pynput`)."
            )

    try:
        asyncio.run(_serve(args, csv_rows, mouse_reader))
    except KeyboardInterrupt:
        print("[mock-gaze] shutting down", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
