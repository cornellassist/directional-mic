"""Record from N audio devices in parallel by spawning one Python
subprocess per device.

Why a subprocess per device: opening the iPhone Continuity Microphone
and a USB-attached iPad input from the same process triggers PortAudio
-9986 ("Audio Hardware Not Running") on macOS — the Continuity
plug-in does not coexist with another live Core Audio input in the
same process, and once a stream open fails the host's PortAudio state
stays dirty until the process exits. Splitting captures across
processes gives each device its own clean PortAudio state.

The parent launches workers back-to-back via ``subprocess.Popen`` and
waits for them, so on a quiet host the recordings start within a few
ms of each other. Each worker writes its own WAV file at the device's
native rate (no on-the-fly resampling), so callers downstream that
need sample alignment must resample to a common rate.

Usage (one --spec per device):

    python -m directional_mic.record_dual --seconds 5 \\
        --spec 0:48000:1:recording_iphone.wav \\
        --spec 1:44100:2:recording_ipad.wav
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass
class _Spec:
    device: int
    samplerate: int
    channels: int
    out: str

    @classmethod
    def parse(cls, s: str) -> "_Spec":
        parts = s.split(":")
        if len(parts) != 4:
            raise argparse.ArgumentTypeError(
                f"--spec must be 'device:samplerate:channels:path' (got {s!r})"
            )
        try:
            return cls(int(parts[0]), int(parts[1]), int(parts[2]), parts[3])
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"bad --spec {s!r}: {e}") from e


def _record_one(spec: _Spec, seconds: float) -> int:
    import numpy as np
    import sounddevice as sd
    from scipy.io import wavfile

    n = int(round(seconds * spec.samplerate))
    t0 = time.perf_counter()
    data = sd.rec(
        n,
        samplerate=spec.samplerate,
        channels=spec.channels,
        device=spec.device,
        dtype="float32",
    )
    sd.wait()
    t1 = time.perf_counter()

    int16 = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
    wavfile.write(spec.out, spec.samplerate, int16)

    print(
        f"[worker dev={spec.device}] wrote {spec.out} "
        f"({n} samples, {spec.channels}ch @ {spec.samplerate} Hz) "
        f"elapsed={t1 - t0:.3f}s",
        flush=True,
    )
    return 0


def _run_parent(specs: list[_Spec], seconds: float, stagger_ms: float) -> int:
    procs: list[subprocess.Popen] = []
    launch_t0 = time.perf_counter()
    for i, spec in enumerate(specs):
        if i > 0 and stagger_ms > 0:
            time.sleep(stagger_ms / 1000.0)
        cmd = [
            sys.executable,
            "-m",
            "directional_mic.record_dual",
            "--worker",
            "--seconds",
            str(seconds),
            "--spec",
            f"{spec.device}:{spec.samplerate}:{spec.channels}:{spec.out}",
        ]
        procs.append(subprocess.Popen(cmd))
    launch_ms = (time.perf_counter() - launch_t0) * 1000
    print(
        f"[parent] launched {len(procs)} worker(s) in {launch_ms:.1f} ms "
        f"(stagger={stagger_ms:.0f}ms); waiting for {seconds:.1f}s of audio...",
        flush=True,
    )
    rcs = [p.wait() for p in procs]
    bad = [(spec.out, rc) for spec, rc in zip(specs, rcs) if rc != 0]
    if bad:
        for name, rc in bad:
            print(f"[parent] {name}: worker exited rc={rc}", file=sys.stderr)
        return 1
    print("[parent] both workers done.", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="directional_mic.record_dual",
        description=(
            "Record from N audio devices in parallel, each in its own "
            "Python subprocess (one PortAudio state per device)."
        ),
    )
    p.add_argument(
        "--spec",
        action="append",
        type=_Spec.parse,
        required=True,
        help=(
            "Repeatable. Format: device:samplerate:channels:path. "
            "Example: --spec 0:48000:1:recording_iphone.wav"
        ),
    )
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument(
        "--stagger-ms",
        type=float,
        default=0.0,
        help=(
            "Wait this many ms between launching each worker. Useful when "
            "two devices fight at the Core Audio routing layer if opened "
            "back-to-back (e.g. iPhone Continuity vs USB iPad)."
        ),
    )
    p.add_argument(
        "--worker",
        action="store_true",
        help="Internal flag: run a single-device capture. Requires one --spec.",
    )
    args = p.parse_args(argv)

    if args.worker:
        if len(args.spec) != 1:
            p.error("--worker requires exactly one --spec")
        return _record_one(args.spec[0], args.seconds)
    return _run_parent(args.spec, args.seconds, args.stagger_ms)


if __name__ == "__main__":
    sys.exit(main())
