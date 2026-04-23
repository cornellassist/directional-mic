"""Real-time CLI entry point.

Two supported input modes:

* **Single-device duplex** (``--input-device``): one audio device
  provides a stereo (or multichannel) input in the same Core Audio /
  WASAPI stream that drives the output. This is the path for USB
  stereo mics and Aggregate Devices.
* **Two-device** (``--inputs DEV1,DEV2``): two independent audio
  devices, one per mic — the intended setup when each phone is
  bridged to the PC by its own iVCam instance. Each device is opened
  as a separate ``sd.InputStream``; the output runs its own
  ``sd.OutputStream``; a ring buffer per device absorbs jitter.
  Independent device clocks will drift — fine for short demos, not
  for multi-minute sessions.

Most parameters are set as module-level constants below. Only device
selection is exposed on the CLI. Use ``--list-devices`` to enumerate
available inputs/outputs.
"""

from __future__ import annotations

import argparse
import queue
import signal
import sys
import threading
import time
import wave

import numpy as np

from .beamformer import STFTDelaySumBeamformer
from .gaze_source import (
    GazeSource,
    GazeUnavailableError,
    MockGazeSource,
    WebSocketGazeSource,
)
from .geometry import gaze_to_azimuth


# ---------------------------------------------------------------------------
# Configuration — edit these, not the CLI.
# ---------------------------------------------------------------------------

# Array geometry
MIC_SPACING_CM = 14.0
ORIENTATION = "broadside"  # "broadside" | "endfire"

# Screen / viewing geometry (for gaze-x → azimuth)
SCREEN_WIDTH_CM = 34.0
VIEW_DISTANCE_CM = 60.0

# Gaze source
GAZE_MODE = "websocket"  # "mock" | "websocket"
GAZE_URI = "ws://localhost:8765"
MOCK_PATTERN = "sweep"  # "center" | "sweep" | "file:path.csv"
MOCK_SWEEP_PERIOD_S = 4.0

# Channel picks and per-channel latency compensation
INPUT_CHANNELS = (0, 1)
LATENCY_OFFSET_SAMPLES = (0, 0)

# STFT / streaming DSP
FS = 16000
FRAME_SIZE = 512
HOP = 256
TAU_SMOOTHING_MS = 50.0

# Logging / liveness
LOG_EVERY_S = 1.0
GAZE_STARTUP_TIMEOUT_S = 10.0
GAZE_STALE_TIMEOUT_MS = 1000.0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="directional_mic.runtime",
        description="Real-time gaze-steered 2-mic STFT delay-and-sum beamformer.",
    )
    p.add_argument("--list-devices", action="store_true",
                   help="List available audio devices and exit.")
    p.add_argument("--input-device", default=None,
                   help="Single input device (duplex mode). Name or index.")
    p.add_argument("--inputs", default=None,
                   help="Two-device mode: comma-separated list of two "
                        "device names or indices, one per mic. When set, "
                        "overrides --input-device. Example: --inputs 7,9.")
    p.add_argument("--output-device", default=None,
                   help="Output device name or index.")
    p.add_argument("--record-out", default=None, metavar="PREFIX",
                   help="If set, stream the beamformed mono output to "
                        "<PREFIX>_processed.wav and the post-delay stereo "
                        "input to <PREFIX>_raw.wav (both int16 @ FS Hz).")
    return p.parse_args(argv)


def _list_devices() -> int:
    import sounddevice as sd
    hostapis = sd.query_hostapis()
    for i, d in enumerate(sd.query_devices()):
        roles = []
        if d["max_input_channels"] > 0:
            roles.append(f"in={d['max_input_channels']}")
        if d["max_output_channels"] > 0:
            roles.append(f"out={d['max_output_channels']}")
        api = hostapis[d["hostapi"]]["name"]
        print(f"{i:3d}  {'/'.join(roles):10s}  sr={d['default_samplerate']:.0f}  "
              f"api={api}  {d['name']!r}")
    return 0


def _resolve_device(spec: str | None) -> int | str | None:
    if spec is None:
        return None
    try:
        return int(spec)
    except ValueError:
        return spec


def _build_gaze_source() -> GazeSource:
    if GAZE_MODE == "mock":
        return MockGazeSource(
            pattern=MOCK_PATTERN,
            sweep_period_s=MOCK_SWEEP_PERIOD_S,
        )
    return WebSocketGazeSource(uri=GAZE_URI)


# ---------------------------------------------------------------------------
# Background WAV recorder: keep disk I/O off the audio callback thread.
# Writes int16 PCM via stdlib `wave` so we don't add a soundfile dependency.
# ---------------------------------------------------------------------------


class _WavRecorder:
    """Async writer for processed mono + post-delay stereo raw."""

    # Queue cap: ~16 s at fs=16k hop=256. Drops over this signal a stuck disk.
    _QUEUE_MAX = 1024

    def __init__(self, prefix: str, samplerate: int) -> None:
        self.processed_path = f"{prefix}_processed.wav"
        self.raw_path = f"{prefix}_raw.wav"
        self._q: queue.Queue = queue.Queue(maxsize=self._QUEUE_MAX)
        self._proc = wave.open(self.processed_path, "wb")
        self._proc.setnchannels(1)
        self._proc.setsampwidth(2)
        self._proc.setframerate(samplerate)
        self._raw = wave.open(self.raw_path, "wb")
        self._raw.setnchannels(2)
        self._raw.setsampwidth(2)
        self._raw.setframerate(samplerate)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._dropped = 0

    def start(self) -> None:
        self._thread.start()

    def submit(self, raw_stereo: np.ndarray, mono: np.ndarray) -> None:
        try:
            self._q.put_nowait((raw_stereo.copy(), mono.copy()))
        except queue.Full:
            self._dropped += 1

    @staticmethod
    def _to_i16(x: np.ndarray) -> bytes:
        return np.ascontiguousarray(
            (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
        ).tobytes()

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            raw_stereo, mono = item
            self._proc.writeframes(self._to_i16(mono))
            self._raw.writeframes(self._to_i16(raw_stereo))

    def close(self) -> None:
        self._q.put(None)
        self._thread.join()
        self._proc.close()
        self._raw.close()

    @property
    def dropped(self) -> int:
        return self._dropped


# ---------------------------------------------------------------------------
# Shared DSP step: given a raw 2-channel block, apply per-channel delay,
# query gaze, and return the beamformed mono block.
# ---------------------------------------------------------------------------


def _make_dsp_step(beamformer, gaze, delay_a, delay_b, recorder=None):
    max_delay = max(delay_a, delay_b, 0)
    delay_buf = (
        np.zeros((max_delay + HOP, 2), dtype=np.float64)
        if max_delay > 0
        else None
    )

    def step(stereo_block: np.ndarray) -> np.ndarray:
        sample = gaze.latest()
        az = gaze_to_azimuth(
            sample.x,
            screen_width_cm=SCREEN_WIDTH_CM,
            view_distance_cm=VIEW_DISTANCE_CM,
        )
        if delay_buf is not None:
            delay_buf[:-HOP] = delay_buf[HOP:]
            delay_buf[-HOP:] = stereo_block
            end = delay_buf.shape[0]
            a = delay_buf[end - HOP - delay_a : end - delay_a, 0]
            b = delay_buf[end - HOP - delay_b : end - delay_b, 1]
            stereo_block = np.stack([a, b], axis=1)
        mono = beamformer.process(stereo_block, az)
        if recorder is not None:
            recorder.submit(stereo_block, mono)
        return mono

    return step


# ---------------------------------------------------------------------------
# Single-device (duplex) path
# ---------------------------------------------------------------------------


def _run_single_device(args, sd, beamformer, gaze, shutdown, dsp_step, recorder=None):
    ch_a, ch_b = INPUT_CHANNELS
    if min(ch_a, ch_b) < 0 or ch_a == ch_b:
        raise ValueError(f"invalid INPUT_CHANNELS: {INPUT_CHANNELS!r}")
    max_ch = max(ch_a, ch_b) + 1

    underruns = 0

    def callback(indata, outdata, frames, time_info, status):
        nonlocal underruns
        if status:
            underruns += 1
        picked = np.stack([indata[:, ch_a], indata[:, ch_b]], axis=1).astype(
            np.float64, copy=False
        )
        mono = dsp_step(picked)
        outdata[:, 0] = mono.astype(outdata.dtype, copy=False)

    with sd.Stream(
        samplerate=FS,
        blocksize=HOP,
        dtype="float32",
        channels=(max_ch, 1),
        device=(_resolve_device(args.input_device),
                _resolve_device(args.output_device)),
        callback=callback,
    ):
        print(f"[rt] single-device mode @ fs={FS} block={HOP} "
              f"frame={FRAME_SIZE}; Ctrl-C to stop.")

        def status_summary():
            extra = f"out_xruns={underruns}"
            if recorder is not None:
                extra += f" rec_drops={recorder.dropped}"
            return extra

        _status_loop(gaze, status_summary, shutdown)


# ---------------------------------------------------------------------------
# Two-device (iVCam-style) path: one InputStream per device + OutputStream
# ---------------------------------------------------------------------------


def _run_two_device(args, sd, beamformer, gaze, shutdown, dsp_step, recorder=None):
    from .audio_input import MultiInputCapture

    dev_specs = [d.strip() for d in args.inputs.split(",")]
    if len(dev_specs) != 2:
        raise ValueError(
            f"--inputs must list exactly two devices (got {args.inputs!r})"
        )
    devs = [_resolve_device(s) for s in dev_specs]
    ch_a, ch_b = INPUT_CHANNELS
    if ch_a < 0 or ch_b < 0:
        raise ValueError(f"invalid INPUT_CHANNELS: {INPUT_CHANNELS!r}")

    capture = MultiInputCapture(
        devices=[(devs[0], ch_a), (devs[1], ch_b)],
        samplerate=FS,
        blocksize=HOP,
    )
    capture.start()

    underruns = 0

    def out_callback(outdata, frames, time_info, status):
        nonlocal underruns
        if status:
            underruns += 1
        block = capture.read_block(frames)  # (frames, 2)
        mono = dsp_step(block)
        outdata[:, 0] = mono.astype(outdata.dtype, copy=False)

    try:
        with sd.OutputStream(
            samplerate=FS,
            blocksize=HOP,
            dtype="float32",
            channels=1,
            device=_resolve_device(args.output_device),
            callback=out_callback,
        ):
            print(f"[rt] two-device mode @ fs={FS} block={HOP} "
                  f"frame={FRAME_SIZE}; devices={dev_specs}; Ctrl-C to stop.")

            def status_summary():
                extra = f"out_xruns={underruns} in_xruns={capture.xruns}"
                if recorder is not None:
                    extra += f" rec_drops={recorder.dropped}"
                return extra

            _status_loop(gaze, status_summary, shutdown)
    finally:
        capture.stop()


# ---------------------------------------------------------------------------
# Shared status-line loop
# ---------------------------------------------------------------------------


def _status_loop(gaze, status_summary, shutdown):
    while not shutdown.is_set():
        time.sleep(LOG_EVERY_S)
        now = time.monotonic()
        extra = status_summary() if callable(status_summary) else ""
        if not isinstance(extra, str):
            extra = f"xruns={extra}"
        try:
            sample = gaze.latest()
        except GazeUnavailableError:
            print(f"[rt] gaze=UNAVAILABLE (no samples received) {extra}",
                  flush=True)
            continue
        age_ms = (now - sample.t) * 1000.0
        stale = age_ms > GAZE_STALE_TIMEOUT_MS
        az_deg = np.degrees(gaze_to_azimuth(
            sample.x,
            screen_width_cm=SCREEN_WIDTH_CM,
            view_distance_cm=VIEW_DISTANCE_CM,
        ))
        stale_tag = " STALE" if stale else ""
        print(f"[rt] gaze=({sample.x:.2f},{sample.y:.2f}) "
              f"az={az_deg:+.1f}° age={age_ms:.0f}ms{stale_tag} {extra}",
              flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    import sounddevice as sd

    delay_a, delay_b = LATENCY_OFFSET_SAMPLES
    if delay_a < 0 or delay_b < 0:
        raise ValueError(f"LATENCY_OFFSET_SAMPLES must be >= 0 "
                         f"(got {LATENCY_OFFSET_SAMPLES!r})")

    beamformer = STFTDelaySumBeamformer(
        fs=FS,
        frame_size=FRAME_SIZE,
        hop=HOP,
        mic_spacing_m=MIC_SPACING_CM / 100.0,
        orientation=ORIENTATION,  # type: ignore[arg-type]
        tau_smoothing_s=TAU_SMOOTHING_MS / 1000.0,
    )

    gaze = _build_gaze_source()
    gaze.start()

    if not gaze.wait_for_first_sample(GAZE_STARTUP_TIMEOUT_S):
        gaze.stop()
        raise SystemExit(
            f"[rt] no gaze sample received within "
            f"{GAZE_STARTUP_TIMEOUT_S:.1f}s — aborting. "
            f"Check that the gaze producer is running and reachable "
            f"(GAZE_MODE={GAZE_MODE}, "
            f"GAZE_URI={GAZE_URI if GAZE_MODE == 'websocket' else 'n/a'})."
        )

    shutdown = threading.Event()

    def _sigint(_sig, _frame):
        shutdown.set()
    signal.signal(signal.SIGINT, _sigint)

    recorder: _WavRecorder | None = None
    if args.record_out:
        recorder = _WavRecorder(args.record_out, FS)
        recorder.start()
        print(f"[rt] recording -> {recorder.processed_path}, "
              f"{recorder.raw_path}", flush=True)

    dsp_step = _make_dsp_step(beamformer, gaze, delay_a, delay_b, recorder)

    try:
        if args.inputs:
            _run_two_device(
                args, sd, beamformer, gaze, shutdown, dsp_step, recorder
            )
        else:
            _run_single_device(
                args, sd, beamformer, gaze, shutdown, dsp_step, recorder
            )
    finally:
        gaze.stop()
        if recorder is not None:
            recorder.close()
            print(f"[rt] wrote {recorder.processed_path}, "
                  f"{recorder.raw_path} "
                  f"(dropped_blocks={recorder.dropped})", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.list_devices:
        return _list_devices()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
