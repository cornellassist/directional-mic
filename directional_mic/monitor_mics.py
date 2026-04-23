"""Live two-mic sanity monitor.

Opens the same two input devices that ``--inputs`` picks in the runtime
and shows a rolling waveform + RMS level meter per mic. Useful while the
beamformer is running in another terminal: tap/speak into each phone in
turn and watch the corresponding plot move.

Usage:

    uv run python -m directional_mic.monitor_mics --inputs 0,1

The monitor opens its own InputStreams, so run it instead of — or
alongside — the runtime depending on whether the input devices allow
multiple simultaneous readers (most Core Audio inputs do; some
drivers don't).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .audio_input import MultiInputCapture


FS = 16000
BLOCK = 256
WINDOW_S = 1.0
RMS_WINDOW_S = 5.0
REFRESH_MS = 30


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="directional_mic.monitor_mics",
        description="Live two-mic waveform + level-meter monitor.",
    )
    p.add_argument(
        "--inputs",
        required=True,
        help="Two devices (name or index), comma-separated. Example: 0,1.",
    )
    p.add_argument(
        "--channels",
        default="0,0",
        help="Per-device channel index to read (default 0,0 — first "
             "channel of each device).",
    )
    p.add_argument("--fs", type=int, default=FS)
    p.add_argument("--window-s", type=float, default=WINDOW_S,
                   help="Rolling waveform window length in seconds.")
    p.add_argument("--rms-window-s", type=float, default=RMS_WINDOW_S,
                   help="Rolling RMS-history window length in seconds.")
    return p.parse_args(argv)


def _resolve(spec: str) -> int | str:
    try:
        return int(spec)
    except ValueError:
        return spec


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    dev_specs = [s.strip() for s in args.inputs.split(",")]
    if len(dev_specs) != 2:
        raise SystemExit(f"--inputs must list exactly two devices (got {args.inputs!r})")
    ch_specs = [s.strip() for s in args.channels.split(",")]
    if len(ch_specs) != 2:
        raise SystemExit(f"--channels must list exactly two indices (got {args.channels!r})")

    devs = [_resolve(s) for s in dev_specs]
    chs = [int(s) for s in ch_specs]

    import matplotlib
    # Force an interactive GUI backend. Without this, `uv run` sometimes
    # picks the headless `Agg` backend, which makes `plt.show()` return
    # immediately — at which point the `finally` below stops the capture
    # and the iPhone/iPad mics "disconnect" a second after opening.
    try:
        matplotlib.use("macosx")
    except Exception:
        matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    print(f"[monitor] matplotlib backend: {matplotlib.get_backend()}", flush=True)

    window_n = int(args.window_s * args.fs)
    rms_hops = max(1, int(args.rms_window_s * args.fs / BLOCK))

    capture = MultiInputCapture(
        devices=[(devs[0], chs[0]), (devs[1], chs[1])],
        samplerate=args.fs,
        blocksize=BLOCK,
        buffer_seconds=max(args.window_s, 1.0),
    )
    capture.start()

    fig, axes = plt.subplots(2, 2, figsize=(10, 5), constrained_layout=True)
    fig.suptitle(
        f"mic monitor — devices={dev_specs} channels={chs} fs={args.fs}Hz "
        f"(Ctrl-C or close window to quit)"
    )

    t_axis = np.arange(window_n) / args.fs - args.window_s
    rms_t = np.arange(rms_hops) * (BLOCK / args.fs) - args.rms_window_s

    wave_lines = []
    rms_lines = []
    titles = []
    rms_hist = [np.zeros(rms_hops), np.zeros(rms_hops)]

    for i, (dev, ch) in enumerate(zip(dev_specs, chs)):
        ax_wave = axes[0, i]
        ax_wave.set_xlim(t_axis[0], t_axis[-1])
        ax_wave.set_ylim(-1.0, 1.0)
        ax_wave.set_xlabel("time (s, latest=0)")
        ax_wave.set_ylabel("amplitude")
        (line_w,) = ax_wave.plot(t_axis, np.zeros(window_n), lw=0.8)
        title = ax_wave.set_title(f"dev {dev} ch {ch}  —  peak=0.00  rms=-inf dBFS")
        wave_lines.append(line_w)
        titles.append(title)

        ax_rms = axes[1, i]
        ax_rms.set_xlim(rms_t[0], rms_t[-1])
        ax_rms.set_ylim(-80, 0)
        ax_rms.set_xlabel("time (s, latest=0)")
        ax_rms.set_ylabel("level (dBFS)")
        ax_rms.grid(True, alpha=0.3)
        (line_r,) = ax_rms.plot(rms_t, np.full(rms_hops, -80.0), lw=1.0)
        rms_lines.append(line_r)

    def _update(_frame):
        block = capture.read_block(window_n)  # (window_n, 2)
        for i in range(2):
            ch_data = block[:, i]
            wave_lines[i].set_ydata(ch_data)

            peak = float(np.max(np.abs(ch_data))) if ch_data.size else 0.0
            rms = float(np.sqrt(np.mean(ch_data[-BLOCK * 4 :] ** 2))) if ch_data.size else 0.0
            rms_db = 20.0 * np.log10(rms + 1e-12)
            rms_db = max(-80.0, min(0.0, rms_db))

            rms_hist[i] = np.roll(rms_hist[i], -1)
            rms_hist[i][-1] = rms_db
            rms_lines[i].set_ydata(rms_hist[i])

            xr = capture.xruns[i]
            titles[i].set_text(
                f"dev {dev_specs[i]} ch {chs[i]}  —  "
                f"peak={peak:.2f}  rms={rms_db:+.1f} dBFS  xruns={xr}"
            )
        return (*wave_lines, *rms_lines, *titles)

    # Keep a reference on the figure so GC can't kill the animation while
    # the event loop is running (some backends only hold a weakref).
    fig._mic_monitor_ani = FuncAnimation(
        fig, _update, interval=REFRESH_MS, blit=False, cache_frame_data=False
    )

    try:
        plt.show(block=True)
    finally:
        capture.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
