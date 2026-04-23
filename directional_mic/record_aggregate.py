"""Record from the macOS Aggregate Device as a single Core Audio stream,
then split channels into per-device WAV files.

Rationale: opening the iPhone Continuity Microphone and the USB iPad
input as two independent PortAudio streams in one process fails
(-9986 Audio Hardware Not Running), and even across two processes
the Continuity plug-in silences the iPad system-wide while it is
active. A single stream on the user's pre-built Aggregate Device lets
Core Audio arbitrate routing and clocks for us — one open, one clock
domain, no cross-stream conflict.

Assumes the Aggregate Device has been configured in Audio MIDI Setup
with the iPhone first (1 ch) and the iPad second (2 ch), for a total
of 3 input channels at 48 kHz. Verify with ``--probe`` if unsure.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import sounddevice as sd
from scipy.io import wavfile


def _find_aggregate(name_hint: str = "Aggregate") -> int:
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and name_hint.lower() in d["name"].lower():
            return i
    raise RuntimeError(f"no input device matching {name_hint!r} found")


def _probe(device: int) -> None:
    info = sd.query_devices(device)
    print(
        f"device {device}: {info['name']!r}  "
        f"in_ch={info['max_input_channels']}  "
        f"default_sr={info['default_samplerate']}",
        flush=True,
    )


def _record(
    device: int,
    samplerate: int,
    channels: int,
    seconds: float,
    iphone_out: str,
    ipad_out: str,
) -> int:
    n = int(round(seconds * samplerate))
    t0 = time.perf_counter()
    data = sd.rec(
        n,
        samplerate=samplerate,
        channels=channels,
        device=device,
        dtype="float32",
    )
    sd.wait()
    t1 = time.perf_counter()

    if data.shape[1] != channels:
        print(
            f"warning: got {data.shape[1]} channels, expected {channels}",
            file=sys.stderr,
        )

    peaks = np.max(np.abs(data), axis=0)
    print(
        f"captured {data.shape[0]} frames x {data.shape[1]} ch "
        f"in {t1 - t0:.3f}s; per-channel peak: "
        + ", ".join(f"ch{i}={p:.3f}" for i, p in enumerate(peaks)),
        flush=True,
    )

    iphone = data[:, 0:1]
    ipad = data[:, 1:3] if data.shape[1] >= 3 else data[:, 1:]

    def _to_i16(x: np.ndarray) -> np.ndarray:
        return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)

    wavfile.write(iphone_out, samplerate, _to_i16(iphone))
    wavfile.write(ipad_out, samplerate, _to_i16(ipad))
    print(f"wrote {iphone_out} (1ch) and {ipad_out} ({ipad.shape[1]}ch)", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="directional_mic.record_aggregate")
    p.add_argument(
        "--device",
        type=int,
        default=None,
        help="Input device index. Defaults to the first device whose name contains 'Aggregate'.",
    )
    p.add_argument("--samplerate", type=int, default=48000)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--iphone-out", default="recording_iphone.wav")
    p.add_argument("--ipad-out", default="recording_ipad.wav")
    p.add_argument(
        "--probe",
        action="store_true",
        help="Print device info and exit without recording.",
    )
    args = p.parse_args(argv)

    device = args.device if args.device is not None else _find_aggregate()
    if args.probe:
        _probe(device)
        return 0
    _probe(device)
    return _record(
        device,
        args.samplerate,
        args.channels,
        args.seconds,
        args.iphone_out,
        args.ipad_out,
    )


if __name__ == "__main__":
    sys.exit(main())
