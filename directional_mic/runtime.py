"""Real-time CLI entry point.

Two supported input modes:

* **Single-device duplex** (``--input-device``): one audio device
  provides a stereo (or multichannel) input in the same Core Audio /
  WASAPI stream that drives the output. This is the path for USB
  stereo mics and Aggregate Devices.
* **Two-device** (``--inputs DEV1,DEV2``): two independent Windows
  audio devices, one per mic — the intended setup when each phone is
  bridged to the PC by its own iVCam instance. Each device is opened
  as a separate ``sd.InputStream``; the output runs its own
  ``sd.OutputStream``; a ring buffer per device absorbs jitter.
  Independent device clocks will drift — fine for short demos, not
  for multi-minute sessions.

Run ``python -m directional_mic.runtime --help`` for the flag surface,
or ``--list-devices`` to enumerate available inputs/outputs.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time

import numpy as np

from .beamformer import STFTDelaySumBeamformer
from .gaze_source import GazeSource, MockGazeSource, WebSocketGazeSource
from .geometry import gaze_to_azimuth


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="directional_mic.runtime",
        description="Real-time gaze-steered 2-mic STFT delay-and-sum beamformer.",
    )
    p.add_argument("--list-devices", action="store_true",
                   help="List available audio devices and exit.")

    p.add_argument("--mic-spacing-cm", type=float,
                   help="Distance between the two mics in centimeters.")
    p.add_argument("--orientation", choices=["broadside", "endfire"],
                   default="broadside",
                   help="Array orientation relative to the listener.")
    p.add_argument("--screen-width-cm", type=float,
                   help="Visible width of the monitor the user faces.")
    p.add_argument("--view-distance-cm", type=float,
                   help="Eye-to-screen distance.")

    p.add_argument("--gaze", choices=["mock", "websocket"], default="mock",
                   help="Gaze source implementation.")
    p.add_argument("--gaze-uri", default="ws://localhost:8765",
                   help="WebSocket URI for the live gaze producer "
                        "(e.g. GazeServer.exe running locally).")
    p.add_argument("--mock-pattern", default="sweep",
                   help="Mock-gaze pattern: center | sweep | file:path.csv")
    p.add_argument("--mock-sweep-period-s", type=float, default=4.0)

    p.add_argument("--input-device", default=None,
                   help="Single input device (duplex mode). Name or index.")
    p.add_argument("--inputs", default=None,
                   help="Two-device mode: comma-separated list of two "
                        "device names or indices, one per mic. When set, "
                        "overrides --input-device. Example: --inputs 7,9 "
                        "for two iVCam instances.")
    p.add_argument("--output-device", default=None,
                   help="Output device name or index.")
    p.add_argument("--input-channels", default="0,1",
                   help="Two channel indices feeding the beamformer. In "
                        "single-device mode they are picked from the one "
                        "device's channels (e.g. '0,2'). In two-device "
                        "mode (--inputs) they pick which channel of each "
                        "device to use; default '0,0' is correct for "
                        "most mono mics.")
    p.add_argument("--latency-offset-samples", default="0,0",
                   help="Per-channel delay in samples applied before "
                        "STFT. Positive values delay that channel. Used "
                        "to compensate for asymmetric device latency.")

    p.add_argument("--fs", type=int, default=16000)
    p.add_argument("--frame", type=int, default=512,
                   help="STFT analysis frame size (samples).")
    p.add_argument("--hop", type=int, default=256,
                   help="STFT hop / audio blocksize (samples).")
    p.add_argument("--tau-smoothing-ms", type=float, default=50.0,
                   help="Exponential smoothing time constant for steering τ.")

    p.add_argument("--log-every-s", type=float, default=1.0,
                   help="How often to print a one-line status update.")

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


def _build_gaze_source(args: argparse.Namespace) -> GazeSource:
    if args.gaze == "mock":
        return MockGazeSource(
            pattern=args.mock_pattern,
            sweep_period_s=args.mock_sweep_period_s,
        )
    return WebSocketGazeSource(uri=args.gaze_uri)


def _parse_pair(spec: str, name: str, cast=int) -> tuple:
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 2:
        raise ValueError(f"{name} must be 'a,b' (got {spec!r})")
    return cast(parts[0]), cast(parts[1])


def _require_config(args: argparse.Namespace) -> None:
    missing = [
        name for name in ("mic_spacing_cm", "screen_width_cm", "view_distance_cm")
        if getattr(args, name) is None
    ]
    if missing:
        raise SystemExit("missing required flags: " +
                         ", ".join("--" + m.replace("_", "-") for m in missing))


# ---------------------------------------------------------------------------
# Shared DSP step: given a raw 2-channel block, apply per-channel delay,
# query gaze, and return the beamformed mono block.
# ---------------------------------------------------------------------------


def _make_dsp_step(beamformer, gaze, args, delay_a, delay_b):
    max_delay = max(delay_a, delay_b, 0)
    delay_buf = (
        np.zeros((max_delay + args.hop, 2), dtype=np.float64)
        if max_delay > 0
        else None
    )
    hop = args.hop

    def step(stereo_block: np.ndarray) -> np.ndarray:
        sample = gaze.latest()
        az = gaze_to_azimuth(
            sample.x,
            screen_width_cm=args.screen_width_cm,
            view_distance_cm=args.view_distance_cm,
        )
        if delay_buf is not None:
            delay_buf[:-hop] = delay_buf[hop:]
            delay_buf[-hop:] = stereo_block
            end = delay_buf.shape[0]
            a = delay_buf[end - hop - delay_a : end - delay_a, 0]
            b = delay_buf[end - hop - delay_b : end - delay_b, 1]
            stereo_block = np.stack([a, b], axis=1)
        return beamformer.process(stereo_block, az)

    return step


# ---------------------------------------------------------------------------
# Single-device (duplex) path
# ---------------------------------------------------------------------------


def _run_single_device(args, sd, beamformer, gaze, shutdown, dsp_step):
    ch_a, ch_b = _parse_pair(args.input_channels, "--input-channels")
    if min(ch_a, ch_b) < 0 or ch_a == ch_b:
        raise ValueError(f"invalid --input-channels: {args.input_channels!r}")
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
        samplerate=args.fs,
        blocksize=args.hop,
        dtype="float32",
        channels=(max_ch, 1),
        device=(_resolve_device(args.input_device),
                _resolve_device(args.output_device)),
        callback=callback,
    ):
        print(f"[rt] single-device mode @ fs={args.fs} block={args.hop} "
              f"frame={args.frame}; Ctrl-C to stop.")
        _status_loop(args, gaze, lambda: underruns, shutdown)


# ---------------------------------------------------------------------------
# Two-device (iVCam-style) path: one InputStream per device + OutputStream
# ---------------------------------------------------------------------------


def _run_two_device(args, sd, beamformer, gaze, shutdown, dsp_step):
    from .audio_input import MultiInputCapture

    dev_specs = [d.strip() for d in args.inputs.split(",")]
    if len(dev_specs) != 2:
        raise ValueError(
            f"--inputs must list exactly two devices (got {args.inputs!r})"
        )
    devs = [_resolve_device(s) for s in dev_specs]
    ch_a, ch_b = _parse_pair(args.input_channels, "--input-channels")
    if ch_a < 0 or ch_b < 0:
        raise ValueError(f"invalid --input-channels: {args.input_channels!r}")

    capture = MultiInputCapture(
        devices=[(devs[0], ch_a), (devs[1], ch_b)],
        samplerate=args.fs,
        blocksize=args.hop,
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
            samplerate=args.fs,
            blocksize=args.hop,
            dtype="float32",
            channels=1,
            device=_resolve_device(args.output_device),
            callback=out_callback,
        ):
            print(f"[rt] two-device mode @ fs={args.fs} block={args.hop} "
                  f"frame={args.frame}; devices={dev_specs}; Ctrl-C to stop.")

            def status_summary():
                return f"out_xruns={underruns} in_xruns={capture.xruns}"

            _status_loop(args, gaze, status_summary, shutdown)
    finally:
        capture.stop()


# ---------------------------------------------------------------------------
# Shared status-line loop
# ---------------------------------------------------------------------------


def _status_loop(args, gaze, status_summary, shutdown):
    while not shutdown.is_set():
        time.sleep(args.log_every_s)
        now = time.monotonic()
        sample = gaze.latest()
        az_deg = np.degrees(gaze_to_azimuth(
            sample.x,
            screen_width_cm=args.screen_width_cm,
            view_distance_cm=args.view_distance_cm,
        ))
        age_ms = (now - sample.t) * 1000.0
        extra = status_summary() if callable(status_summary) else ""
        if not isinstance(extra, str):
            extra = f"xruns={extra}"
        print(f"[rt] gaze=({sample.x:.2f},{sample.y:.2f}) "
              f"az={az_deg:+.1f}° age={age_ms:.0f}ms {extra}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    import sounddevice as sd

    _require_config(args)
    delay_a, delay_b = _parse_pair(args.latency_offset_samples,
                                   "--latency-offset-samples")
    if delay_a < 0 or delay_b < 0:
        raise ValueError(f"latency offsets must be >= 0 "
                         f"(got {args.latency_offset_samples!r})")

    beamformer = STFTDelaySumBeamformer(
        fs=args.fs,
        frame_size=args.frame,
        hop=args.hop,
        mic_spacing_m=args.mic_spacing_cm / 100.0,
        orientation=args.orientation,  # type: ignore[arg-type]
        tau_smoothing_s=args.tau_smoothing_ms / 1000.0,
    )

    gaze = _build_gaze_source(args)
    gaze.start()

    shutdown = threading.Event()

    def _sigint(_sig, _frame):
        shutdown.set()
    signal.signal(signal.SIGINT, _sigint)

    dsp_step = _make_dsp_step(beamformer, gaze, args, delay_a, delay_b)

    try:
        if args.inputs:
            _run_two_device(args, sd, beamformer, gaze, shutdown, dsp_step)
        else:
            _run_single_device(args, sd, beamformer, gaze, shutdown, dsp_step)
    finally:
        gaze.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.list_devices:
        return _list_devices()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
