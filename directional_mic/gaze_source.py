"""Gaze-source abstraction: mock for offline dev, websocket for live use.

Both implementations run a background thread that keeps ``latest()``
cheap and non-blocking so the real-time audio callback can poll it.

The websocket contract matches ``eye_tracking/GazeExample.py`` and
``GazeServer.exe``: JSON messages with fields ``x``, ``y`` in [0, 1] and
``timestamp``.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GazeSample:
    x: float
    y: float
    t: float  # seconds (monotonic local clock when the sample was stored)


class GazeSource(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def latest(self) -> GazeSample: ...


class _BaseGazeSource:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest = GazeSample(x=0.5, y=0.5, t=time.monotonic())
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def latest(self) -> GazeSample:
        with self._lock:
            return self._latest

    def _set(self, x: float, y: float) -> None:
        sample = GazeSample(x=float(x), y=float(y), t=time.monotonic())
        with self._lock:
            self._latest = sample

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)


class MockGazeSource(_BaseGazeSource):
    """Deterministic gaze pattern for offline development.

    Patterns:
    - ``center``: always (0.5, 0.5).
    - ``sweep``: x oscillates between 0 and 1 with the given period;
      y stays at 0.5.
    - ``file:path.csv``: replay a CSV with ``x,y`` rows (header optional),
      one row per ``1 / rate`` seconds, looping.
    """

    def __init__(self, pattern: str = "center", rate_hz: float = 60.0,
                 sweep_period_s: float = 4.0) -> None:
        super().__init__()
        self.pattern = pattern
        self.rate_hz = float(rate_hz)
        self.sweep_period_s = float(sweep_period_s)
        self._csv_rows: list[tuple[float, float]] | None = None
        if pattern.startswith("file:"):
            self._csv_rows = self._load_csv(pattern[len("file:") :])

    @staticmethod
    def _load_csv(path: str) -> list[tuple[float, float]]:
        rows: list[tuple[float, float]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue  # skip header or non-numeric lines
        if not rows:
            raise ValueError(f"no numeric (x,y) rows found in {path}")
        return rows

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        period = 1.0 / max(self.rate_hz, 1.0)
        t0 = time.monotonic()
        i = 0
        while not self._stop.is_set():
            t = time.monotonic() - t0
            if self.pattern == "center":
                self._set(0.5, 0.5)
            elif self.pattern == "sweep":
                phase = (t % self.sweep_period_s) / self.sweep_period_s
                x = 0.5 + 0.5 * math.sin(2 * math.pi * phase)
                self._set(x, 0.5)
            elif self._csv_rows is not None:
                x, y = self._csv_rows[i % len(self._csv_rows)]
                self._set(x, y)
                i += 1
            else:
                # Unknown pattern: hold center.
                self._set(0.5, 0.5)
            self._stop.wait(period)


class WebSocketGazeSource(_BaseGazeSource):
    """Client for ``GazeServer.exe``-style JSON gaze over a websocket.

    Runs its own asyncio loop on a dedicated thread so the synchronous
    audio callback doesn't have to coordinate with asyncio at all.
    Automatically reconnects with a small backoff if the connection drops.
    """

    def __init__(self, uri: str, reconnect_s: float = 1.0) -> None:
        super().__init__()
        self.uri = uri
        self.reconnect_s = float(reconnect_s)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        try:
            asyncio.run(self._consume())
        except Exception as e:  # noqa: BLE001 - background thread isolator
            print(f"[gaze] websocket loop terminated: {e}")

    async def _consume(self) -> None:
        import websockets  # lazy import so Mock-only users don't pay for it

        while not self._stop.is_set():
            try:
                async with websockets.connect(self.uri) as ws:
                    async for msg in ws:
                        if self._stop.is_set():
                            break
                        self._handle(msg)
            except Exception as e:  # noqa: BLE001 - reconnect on any failure
                print(f"[gaze] connection error ({e!r}); reconnecting in "
                      f"{self.reconnect_s:.1f}s")
                await asyncio.sleep(self.reconnect_s)

    def _handle(self, msg: str | bytes) -> None:
        try:
            payload = json.loads(msg)
        except (TypeError, ValueError):
            return
        x = payload.get("x")
        y = payload.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            self._set(x, y)


def make_gaze_source(kind: str, **kwargs) -> GazeSource:
    if kind == "mock":
        return MockGazeSource(**kwargs)
    if kind == "websocket":
        return WebSocketGazeSource(**kwargs)
    raise ValueError(f"unknown gaze source kind: {kind!r}")
