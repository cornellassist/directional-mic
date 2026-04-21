# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

Early-stage. The top-level project is `directional-mic`, but only the `eye_tracking/` submodule currently has code. The top-level `README.md` is a stub. The eye-tracking code is vendored/adapted from the PyEyetracker project (see `eye_tracking/README.md`, `eye_tracking/LICENSE`) — when editing inside `eye_tracking/`, assume upstream conventions unless the user asks otherwise.

The `eye_tracking/README.md` references sibling directories (`arduino/wheelchair_control_serial/`, `python/python_interface/python_interface_gaze.py`) that **do not exist in this repo**. Treat those references as aspirational/legacy — do not follow those setup steps. The wheelchair-control context hints at the eventual integration target but is not present.

## Deployment model

The eye tracker is **not run from this machine**. The Windows-only `eye_tracking/` stack (Tobii SDK, `TobiiEyeTracker.pyd`, `tobii_stream_engine.dll`, pywin32 named pipes) runs on a separate Windows host and streams gaze data over a network port to consumers in this repo. Treat `eye_tracking/` as reference/vendored code documenting the producer side — not something to execute here. Downstream code in `directional-mic` should consume gaze data over the wire (e.g., the websocket interface exposed by `GazeServer.exe` on port 8765, or an equivalent protocol), not by importing `GazeTracker` locally.

## Architecture: the 32/64-bit split

The non-obvious core design choice: `TobiiEyeTracker.pyd` is a **32-bit** C extension, but consumers want to run in **64-bit** Python. The code bridges this with a two-process pipe:

1. `eye_tracking/GazeTracker.py` (runs in the user's 64-bit Python) — the public API class `GazeTracker`. On construction it:
   - Launches `_listener_win32.py` as a subprocess using a hardcoded 32-bit Python interpreter path (`PYTHON_32BIT`).
   - Opens the client end of a Windows named pipe `\\.\pipe\TobiiGazeData`.
   - `get_movement()` reads packed `!ff` floats (x, y ∈ [0,1]) off the pipe and returns `[(x, y), ...]`.

2. `eye_tracking/_listener_win32.py` (runs in 32-bit Python) — imports `TobiiEyeTracker`, calls `TobiiEyeTracker.init()` to spawn the Tobii callback thread, then polls `TobiiEyeTracker.getBuffer()` every 100 ms and writes points into the named pipe. Exits after ~10 minutes with no active readers.

3. `eye_tracking/EyetrackerExtention.cpp` — source for `TobiiEyeTracker.pyd`. Spawns a detached `loopWrite` thread on `init()` that subscribes to `tobii_gaze_point_t` callbacks and pushes valid points onto a mutex-protected `std::queue<Point>`. `getBuffer()` drains the queue into a Python tuple of `(x, y)` tuples. The `.pyd` binary is committed; rebuilding requires Visual Studio + the (now-unavailable) `Tobii.StreamEngine` NuGet package — the README acknowledges this is effectively frozen.

There is also a separate, self-contained `GazeServer.exe` that exposes the same data over a WebSocket on `ws://localhost:8765` as JSON `{x, y, timestamp}`. `GazeExample.py` is a minimal websocket client for it. The `.exe` is a black box — no source is in this repo.

## The `PYTHON_32BIT` path gotcha

`GazeTracker.py:16` hardcodes `PYTHON_32BIT = r"C:\Users\brian\anaconda3\envs\py32\python.exe"`. This is someone else's machine path left in from upstream. Any user on a new machine must replace it (or pass `python_32bit=...` to `GazeTracker(...)`). The constructor accepts the arg but then **overwrites it** with the module-level constant on line 32 (`python_32bit = PYTHON_32BIT`) — that line is a bug: it defeats the parameter. Flag this if the user asks about initialization.

## Consuming gaze data from this repo

Expect gaze data to arrive over a network socket from the remote Windows producer. The reference client is `eye_tracking/GazeExample.py`: connect to `ws://<host>:8765` and parse JSON frames `{x, y, timestamp}` where `x, y ∈ [0, 1]` (normalized screen coordinates). The host/port may differ in practice — confirm with the user.

No test suite, linter config, or build script exists in this repo.

## User preferences

Global `~/.claude/CLAUDE.md` applies (no Co-Authored-By line in commits; prefer `uv` for Python envs — though that doesn't help here since the 32-bit Python interpreter is Windows-specific and conda-driven).
