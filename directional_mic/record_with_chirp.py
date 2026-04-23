"""Record one mic on this Mac while emitting a short chirp at t=0 so a
second device (e.g. AirPods paired to an iPhone running Voice Memos)
can be time-aligned in post.

Usage:

    python -m directional_mic.record_with_chirp \\
        --device 6 --samplerate 24000 --channels 1 \\
        --seconds 8 --out recording_mac_airpods.wav

The chirp is played to the system's default output. Point that output
at the Mac speakers (NOT at the AirPods you're recording from) so the
chirp is acoustically picked up by both the Mac-side AirPods mic and
the iPhone-side AirPods mic. That shared transient is your alignment
landmark.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import numpy as np
import sounddevice as sd
from scipy.io import wavfile


def _make_chirp(samplerate: int, duration: float = 0.05, f0: float = 2000.0, f1: float = 6000.0) -> np.ndarray:
    n = int(round(duration * samplerate))
    t = np.arange(n) / samplerate
    k = (f1 - f0) / duration
    x = np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t))
    env = np.ones(n)
    fade = max(1, int(0.005 * samplerate))
    env[:fade] = np.linspace(0.0, 1.0, fade)
    env[-fade:] = np.linspace(1.0, 0.0, fade)
    return (0.6 * x * env).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="directional_mic.record_with_chirp")
    p.add_argument("--device", type=int, required=True, help="Input device index (see `python -m sounddevice`).")
    p.add_argument("--samplerate", type=int, default=24000)
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--chirp-delay", type=float, default=0.3, help="Seconds from record-start to chirp onset.")
    p.add_argument("--chirp-duration", type=float, default=0.05)
    p.add_argument("--chirp-f0", type=float, default=2000.0)
    p.add_argument("--chirp-f1", type=float, default=6000.0)
    p.add_argument("--no-chirp", action="store_true", help="Skip the chirp (record only).")
    p.add_argument(
        "--output-device",
        type=int,
        default=None,
        help="Output device index for the chirp. Defaults to system default; set to the Mac speakers index to avoid Bluetooth output latency when default is AirPods.",
    )
    args = p.parse_args(argv)

    n = int(round(args.seconds * args.samplerate))
    chirp = _make_chirp(args.samplerate, args.chirp_duration, args.chirp_f0, args.chirp_f1)
    data = np.zeros((n, args.channels), dtype=np.float32)

    print(
        f"[rec] device={args.device} sr={args.samplerate} ch={args.channels} "
        f"dur={args.seconds:.1f}s -> {args.out}",
        flush=True,
    )

    write_idx = 0

    def _in_cb(indata, frames, time_info, status):
        nonlocal write_idx
        if status:
            print(f"[rec] input status: {status}", file=sys.stderr, flush=True)
        end = min(write_idx + frames, n)
        data[write_idx:end] = indata[: end - write_idx]
        write_idx = end

    t0 = time.perf_counter()
    in_stream = sd.InputStream(
        samplerate=args.samplerate,
        channels=args.channels,
        device=args.device,
        dtype="float32",
        callback=_in_cb,
    )
    in_stream.start()

    chirp_thread: threading.Thread | None = None
    if not args.no_chirp:
        def _play_chirp():
            time.sleep(args.chirp_delay)
            t_chirp = time.perf_counter() - t0
            print(f"[rec] chirp onset at t={t_chirp:.3f}s (target={args.chirp_delay:.3f}s)", flush=True)
            try:
                out_stream = sd.OutputStream(
                    samplerate=args.samplerate,
                    channels=1,
                    dtype="float32",
                    device=args.output_device,
                )
                out_stream.start()
                out_stream.write(chirp.reshape(-1, 1))
                out_stream.stop()
                out_stream.close()
            except Exception as e:
                print(f"[rec] chirp playback failed: {e}", file=sys.stderr, flush=True)

        chirp_thread = threading.Thread(target=_play_chirp, daemon=True)
        chirp_thread.start()

    while write_idx < n:
        time.sleep(0.01)
    in_stream.stop()
    in_stream.close()
    if chirp_thread is not None:
        chirp_thread.join(timeout=2.0)
    t1 = time.perf_counter()

    int16 = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
    wavfile.write(args.out, args.samplerate, int16)
    print(f"[rec] wrote {args.out} ({n} samples) elapsed={t1 - t0:.3f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
