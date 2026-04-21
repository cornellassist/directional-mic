"""Multi-device synchronized audio capture.

When the two mic channels come from **separate** audio devices — each
with its own driver and clock, e.g. two iVCam instances on Windows,
each bridging a different iPhone — we can't rely on a single duplex
stream to deliver them aligned sample-by-sample. This module opens one
``sd.InputStream`` per device and exposes a ``read_block(n)`` method
that returns the most recent ``n`` samples from each device as a
single ``(n, N_dev)`` array.

Limitations:
  * Independent device clocks will drift. ``sounddevice`` does not do
    drift correction for us; long-running sessions will accumulate
    sample slip on the order of tens of µs per second. Fine for
    short demos, not fine for >minute sessions.
  * Per-device startup latency differs. Use
    ``runtime --latency-offset-samples`` (or the callback latency
    reported per device) to compensate for the constant component.
  * If a device falls behind momentarily, its column is zero-padded on
    the left rather than stalling the output.
"""

from __future__ import annotations

import threading
from typing import Sequence

import numpy as np


class MultiInputCapture:
    """Open N independent input devices; read synchronized blocks."""

    def __init__(
        self,
        devices: Sequence[tuple[int | str, int]],
        samplerate: int,
        blocksize: int,
        buffer_seconds: float = 0.5,
        dtype: str = "float32",
    ) -> None:
        import sounddevice as sd  # lazy: keep help/tests portable

        if len(devices) < 1:
            raise ValueError("need at least one device")

        self.samplerate = int(samplerate)
        self.blocksize = int(blocksize)
        self.dtype = dtype
        self.devices = list(devices)
        self.n_dev = len(self.devices)

        buf_len = max(self.blocksize * 8, int(buffer_seconds * self.samplerate))
        self._rings: list[np.ndarray] = [
            np.zeros(buf_len, dtype=np.float64) for _ in range(self.n_dev)
        ]
        self._write_idx = [0] * self.n_dev
        self._samples_written = [0] * self.n_dev
        self._xruns = [0] * self.n_dev
        self._lock = threading.Lock()

        self._streams = []
        for i, (dev, channel_idx) in enumerate(self.devices):
            info = sd.query_devices(dev, "input")
            max_ch = int(info["max_input_channels"])
            if channel_idx < 0 or channel_idx >= max_ch:
                raise ValueError(
                    f"device {dev!r}: channel_idx {channel_idx} out of range "
                    f"(device reports {max_ch} input channels)"
                )
            # Open just enough channels to reach the selected one.
            n_ch_open = channel_idx + 1
            self._streams.append(
                sd.InputStream(
                    samplerate=self.samplerate,
                    blocksize=self.blocksize,
                    device=dev,
                    channels=n_ch_open,
                    dtype=self.dtype,
                    callback=self._make_callback(i, channel_idx),
                )
            )

    # ------- lifecycle --------------------------------------------------

    def start(self) -> None:
        for s in self._streams:
            s.start()

    def stop(self) -> None:
        for s in self._streams:
            try:
                s.stop()
                s.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass

    # ------- callback plumbing -----------------------------------------

    def _make_callback(self, dev_i: int, channel_idx: int):
        """Build a closure used as the per-device InputStream callback."""
        ring = self._rings[dev_i]
        cap = ring.shape[0]

        def cb(indata, frames, time_info, status):  # noqa: ARG001
            if status:
                # input overflow / underflow reports from PortAudio.
                self._xruns[dev_i] += 1
            data = np.asarray(indata[:, channel_idx], dtype=np.float64)
            idx = self._write_idx[dev_i]
            end = idx + frames
            if end <= cap:
                ring[idx:end] = data
            else:
                first = cap - idx
                ring[idx:] = data[:first]
                ring[: end - cap] = data[first:]
            # The GIL makes these two writes effectively atomic for a
            # Python-side reader that also holds the lock during read.
            self._write_idx[dev_i] = end % cap
            self._samples_written[dev_i] += frames

        return cb

    # ------- consumer API ----------------------------------------------

    def read_block(self, n: int) -> np.ndarray:
        """Most recent ``n`` samples from each device as ``(n, N_dev)``.

        If a device hasn't produced ``n`` samples yet, its column is
        zero-padded on the left.
        """
        out = np.zeros((n, self.n_dev), dtype=np.float64)
        with self._lock:
            for i in range(self.n_dev):
                ring = self._rings[i]
                cap = ring.shape[0]
                idx = self._write_idx[i]
                available = min(self._samples_written[i], cap)
                if available == 0:
                    continue
                k = min(n, available)
                start = (idx - k) % cap
                end = start + k
                if end <= cap:
                    recent = ring[start:end]
                else:
                    recent = np.concatenate([ring[start:], ring[: end - cap]])
                out[n - k : n, i] = recent
        return out

    # ------- diagnostics -----------------------------------------------

    @property
    def xruns(self) -> list[int]:
        return list(self._xruns)

    @property
    def samples_written(self) -> list[int]:
        return list(self._samples_written)
