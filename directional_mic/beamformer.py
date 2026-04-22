"""STFT delay-and-sum beamformer for a 2-mic array.

Streaming block-based processing via overlap-add. The steering direction
can change every block; ``τ`` (the time-difference-of-arrival used for
the per-bin phase rotation) is exponentially smoothed inside the
processor to avoid clicks when gaze jumps.
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import Orientation, tdoa


class STFTDelaySumBeamformer:
    """Block-streaming STFT delay-and-sum beamformer.

    Call ``process(block, azimuth_rad)`` with a ``(hop, 2)`` float array
    and get a ``(hop,)`` mono output. The first ``frame_size - hop``
    output samples are warm-up (returned near-zero) because overlap-add
    needs one full frame of input before producing valid output — in a
    streaming loop that shows up as a single ``frame_size``-long latency
    at startup.
    """

    def __init__(
        self,
        fs: int,
        frame_size: int,
        hop: int,
        mic_spacing_m: float,
        orientation: Orientation = "broadside",
        tau_smoothing_s: float = 0.05,
    ) -> None:
        if frame_size <= 0 or hop <= 0:
            raise ValueError("frame_size and hop must be positive")
        if frame_size % hop != 0:
            raise ValueError("frame_size must be a multiple of hop for clean OLA")
        if mic_spacing_m <= 0:
            raise ValueError("mic_spacing_m must be positive")

        self.fs = int(fs)
        self.frame_size = int(frame_size)
        self.hop = int(hop)
        self.mic_spacing_m = float(mic_spacing_m)
        self.orientation: Orientation = orientation

        # Sqrt-Hann on both analysis and synthesis gives perfect COLA on
        # w^2 at 50% overlap: sum_k w[n]^2 shifted = 1 in the interior.
        hann = np.hanning(self.frame_size).astype(np.float64)
        self.window = np.sqrt(hann)

        self.freqs = np.fft.rfftfreq(self.frame_size, d=1.0 / self.fs)

        self._in_buf = np.zeros((self.frame_size, 2), dtype=np.float64)
        self._out_buf = np.zeros(self.frame_size, dtype=np.float64)

        dt = self.hop / self.fs
        self._alpha = 1.0 if tau_smoothing_s <= 0 else 1.0 - math.exp(-dt / tau_smoothing_s)
        self._tau_current: float | None = None

    def reset(self) -> None:
        self._in_buf.fill(0.0)
        self._out_buf.fill(0.0)
        self._tau_current = None

    def _target_tau(self, azimuth_rad: float) -> float:
        return tdoa(
            azimuth_rad,
            mic_spacing_m=self.mic_spacing_m,
            orientation=self.orientation,
        )

    def process(self, block: np.ndarray, azimuth_rad: float) -> np.ndarray:
        if block.ndim != 2 or block.shape[1] != 2:
            raise ValueError(f"expected (hop, 2) block, got shape {block.shape}")
        if block.shape[0] != self.hop:
            raise ValueError(f"expected block length {self.hop}, got {block.shape[0]}")

        # Slide the input ring: drop oldest `hop`, append new `hop`.
        self._in_buf[: -self.hop] = self._in_buf[self.hop :]
        self._in_buf[-self.hop :] = block

        # Smooth τ toward the target direction.
        target_tau = self._target_tau(azimuth_rad)
        if self._tau_current is None:
            self._tau_current = target_tau
        else:
            self._tau_current += self._alpha * (target_tau - self._tau_current)

        # Apply per-channel panning gains before delay-and-sum
        pan = max(-1.0, min(1.0, math.sin(azimuth_rad)))

        # stronger left when cursor is left, stronger right when cursor is right
        left_gain = 0.2 + 0.8 * (1.0 - pan) / 2.0
        right_gain = 0.2 + 0.8 * (1.0 + pan) / 2.0

        # Analyze: windowed rFFT of each channel with panning gains applied.
        ch1 = self._in_buf[:, 0] * self.window * left_gain
        ch2 = self._in_buf[:, 1] * self.window * right_gain
        X1 = np.fft.rfft(ch1)
        X2 = np.fft.rfft(ch2)

        # Delay-and-sum: channel 2 is phase-rotated to align with channel 1
        # for a source at the steering direction, then averaged.
        steering = np.exp(-1j * 2.0 * math.pi * self.freqs * self._tau_current)
        Y = X1 + X2 * steering

        # Synthesize: iFFT, apply synthesis window, overlap-add.
        frame = np.fft.irfft(Y, n=self.frame_size) * self.window
        self._out_buf += frame

        out = self._out_buf[: self.hop].copy()
        self._out_buf[: -self.hop] = self._out_buf[self.hop :]
        self._out_buf[-self.hop :] = 0.0
        
        return out
