"""Real-time CLI entry point.

Wires a ``GazeSource`` and ``STFTDelaySumBeamformer`` into a full-duplex
``sounddevice.Stream`` — stereo in, mono out. The audio callback is the
only place that touches the beamformer; it polls the gaze source
non-blockingly and hands the current azimuth to the DSP.

Run ``python -m directional_mic.runtime --help`` for the flag surface.
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
    p.add_argument("--mic-spacing-cm", type=float, required=True,
                   help="Distance between the two mics in centimeters.")
    p.add_argument("--orientation", choices=["broadside", "endfire"],
                   default="broadside",
                   help="Array orientation relative to the listener.")
    p.add_argument("--screen-width-cm", type=float, required=True,
                   help="Visible width of the monitor the user faces.")
    p.add_argument("--view-distance-cm", type=float, required=True,
                   help="Eye-to-screen distance.")

    p.add_argument("--gaze", choices=["mock", "websocket"], default="mock",
                   help="Gaze source implementation.")
    p.add_argument("--gaze-uri", default="ws://localhost:8765",
                   help="WebSocket URI for the live gaze producer.")
    p.add_argument("--mock-pattern", default="sweep",
                   help="Mock-gaze pattern: center | sweep | file:path.csv")
    p.add_argument("--mock-sweep-period-s", type=float, default=4.0)

    p.add_argument("--input-device", default=None,
                   help="Input device name or index (see `python -m sounddevice`).")
    p.add_argument("--output-device", default=None,
                   help="Output device name or index.")
    p.add_argument("--input-channels", default="0,1",
                   help="Zero-based indices of the two channels on the input "
                        "device to feed the beamformer (e.g. '0,2').")
    p.add_argument("--latency-offset-samples", default="0,0",
                   help="Per-channel delay in samples applied before STFT. "
                        "Positive values delay that channel. Used to compensate "
                        "for asymmetric device latency when combining two phones "
                        "via an Aggregate Device.")

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


def run(args: argparse.Namespace) -> int:
    import sounddevice as sd  # imported late so --help works without PortAudio

    ch_a, ch_b = _parse_pair(args.input_channels, "--input-channels")
    delay_a, delay_b = _parse_pair(args.latency_offset_samples,
                                   "--latency-offset-samples")
    if min(ch_a, ch_b) < 0 or ch_a == ch_b:
        raise ValueError(f"invalid --input-channels: {args.input_channels!r}")
    max_ch = max(ch_a, ch_b) + 1

    beamformer = STFTDelaySumBeamformer(
        fs=args.fs,
        frame_size=args.frame,
        hop=args.hop,
        mic_spacing_m=args.mic_spacing_cm / 100.0,
        orientation=args.orientation,  # type: ignore[arg-type]
        tau_smoothing_s=args.tau_smoothing_ms / 1000.0,
    )

    # Per-channel delay rings: hold the last `max_delay` samples of each
    # selected channel so we can pull out a delayed copy without
    # reallocating. The DSP sees channel-pair samples offset in time by
    # `delay_a` / `delay_b` samples.
    max_delay = max(delay_a, delay_b, 0)
    delay_buf = np.zeros((max_delay + args.hop, 2), dtype=np.float64) if max_delay > 0 else None

    gaze = _build_gaze_source(args)
    gaze.start()

    shutdown = threading.Event()

    def _sigint(_sig, _frame):
        shutdown.set()
    signal.signal(signal.SIGINT, _sigint)

    underruns = 0

    def callback(indata, outdata, frames, time_info, status):
        nonlocal underruns
        if status:
            underruns += 1
        sample = gaze.latest()
        az = gaze_to_azimuth(
            sample.x,
            screen_width_cm=args.screen_width_cm,
            view_distance_cm=args.view_distance_cm,
        )
        # Slice the two chosen channels out of the multichannel input.
        picked = np.stack(
            [indata[:, ch_a], indata[:, ch_b]], axis=1
        ).astype(np.float64, copy=False)

        if delay_buf is not None:
            # Append new samples, then pull out each channel delayed by
            # the requested per-channel amount.
            delay_buf[: -args.hop] = delay_buf[args.hop :]
            delay_buf[-args.hop :] = picked
            end = delay_buf.shape[0]
            a = delay_buf[end - args.hop - delay_a : end - delay_a, 0]
            b = delay_buf[end - args.hop - delay_b : end - delay_b, 1]
            block = np.stack([a, b], axis=1)
        else:
            block = picked

        mono = beamformer.process(block, az)
        outdata[:, 0] = mono.astype(outdata.dtype, copy=False)

    try:
        with sd.Stream(
            samplerate=args.fs,
            blocksize=args.hop,
            dtype="float32",
            channels=(max_ch, 1),
            device=(_resolve_device(args.input_device),
                    _resolve_device(args.output_device)),
            callback=callback,
        ):
            print(f"[rt] streaming @ fs={args.fs} block={args.hop} "
                  f"frame={args.frame}; Ctrl-C to stop.")
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
                print(f"[rt] gaze=({sample.x:.2f},{sample.y:.2f}) "
                      f"az={az_deg:+.1f}° age={age_ms:.0f}ms "
                      f"xruns={underruns}")
                last_log = now
    finally:
        gaze.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
