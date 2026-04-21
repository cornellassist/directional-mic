"""Tests for the STFT delay-and-sum beamformer.

Strategy: drive the processor with synthesized 2-channel signals whose
inter-channel delay corresponds to a known azimuth, then check that:
  1. steering to the true azimuth gives near-unity output amplitude,
  2. steering to the opposite azimuth attenuates the output,
  3. mono-identical input steered to 0° passes through near-unity.

We let the beamformer warm up for several frames before measuring,
because the first ``frame_size`` samples out are OLA startup.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from directional_mic.beamformer import STFTDelaySumBeamformer
from directional_mic.geometry import tdoa


def _make_beamformer(fs=16000, frame_size=512, hop=256, spacing=0.15):
    return STFTDelaySumBeamformer(
        fs=fs,
        frame_size=frame_size,
        hop=hop,
        mic_spacing_m=spacing,
        orientation="broadside",
        tau_smoothing_s=0.0,  # no smoothing -> step response is instant
    )


def _tone_pair(freq_hz, fs, n, tau_s):
    """Synthesize a 2-mic recording of a source whose TDOA is tau_s.

    Matches the geometry convention: positive ``tau_s`` means the source
    reaches mic 2 first and mic 1 ``tau_s`` seconds later (mic 2 is the
    closer mic, which is the case for a source at positive broadside
    azimuth). So channel 1 is the delayed copy.
    """
    t = np.arange(n) / fs
    ch1 = np.sin(2 * math.pi * freq_hz * (t - tau_s))
    ch2 = np.sin(2 * math.pi * freq_hz * t)
    return np.stack([ch1, ch2], axis=1)


def _run(beamformer, stereo, azimuth_rad):
    hop = beamformer.hop
    out_blocks = []
    n_blocks = stereo.shape[0] // hop
    for i in range(n_blocks):
        blk = stereo[i * hop : (i + 1) * hop]
        out_blocks.append(beamformer.process(blk, azimuth_rad))
    return np.concatenate(out_blocks)


def _rms(x):
    return float(np.sqrt(np.mean(x * x)))


def _measure_tail(out, beamformer):
    """RMS over the steady-state tail (skip warm-up)."""
    skip = 4 * beamformer.frame_size
    return _rms(out[skip:])


class TestDelaySumBeamformer:
    def test_mono_identical_passes_through_at_zero_azimuth(self):
        bf = _make_beamformer()
        # Same signal on both channels -> τ=0 steering should pass it through.
        fs = bf.fs
        n = fs  # 1 second
        t = np.arange(n) / fs
        mono = 0.5 * np.sin(2 * math.pi * 1000 * t)
        stereo = np.stack([mono, mono], axis=1)
        out = _run(bf, stereo, azimuth_rad=0.0)

        in_rms = _rms(mono[4 * bf.frame_size :])
        out_rms = _measure_tail(out, bf)
        # Passthrough within ~0.5 dB.
        ratio = out_rms / in_rms
        assert 0.94 < ratio < 1.06, f"passthrough gain {20 * math.log10(ratio):.2f} dB"

    def test_on_axis_gain_near_unity(self):
        bf = _make_beamformer()
        fs = bf.fs
        n = fs
        azimuth = math.radians(30.0)
        tau = tdoa(azimuth, mic_spacing_m=bf.mic_spacing_m)
        stereo = _tone_pair(1000.0, fs, n, tau_s=tau)

        out = _run(bf, stereo, azimuth_rad=azimuth)
        # Target RMS of a unit sine is 1/sqrt(2); out should be close.
        out_rms = _measure_tail(out, bf)
        ref_rms = 1.0 / math.sqrt(2.0)
        ratio = out_rms / ref_rms
        assert 0.9 < ratio < 1.1, f"on-axis gain {20 * math.log10(ratio):.2f} dB"

    def test_off_axis_is_attenuated(self):
        bf_on = _make_beamformer()
        bf_off = _make_beamformer()
        fs = bf_on.fs
        n = fs
        # Source actually at +30°.
        true_az = math.radians(30.0)
        tau = tdoa(true_az, mic_spacing_m=bf_on.mic_spacing_m)
        stereo = _tone_pair(1500.0, fs, n, tau_s=tau)

        out_on = _run(bf_on, stereo, azimuth_rad=true_az)
        out_off = _run(bf_off, stereo, azimuth_rad=-true_az)

        rms_on = _measure_tail(out_on, bf_on)
        rms_off = _measure_tail(out_off, bf_off)
        # Steering to the wrong side of broadside should attenuate.
        assert rms_off < 0.9 * rms_on, (
            f"off-axis not attenuated: on={rms_on:.4f} off={rms_off:.4f}"
        )

    def test_process_rejects_wrong_shape(self):
        bf = _make_beamformer()
        with pytest.raises(ValueError):
            bf.process(np.zeros((bf.hop, 3)), 0.0)
        with pytest.raises(ValueError):
            bf.process(np.zeros((bf.hop + 1, 2)), 0.0)

    def test_rejects_incompatible_frame_hop(self):
        with pytest.raises(ValueError):
            STFTDelaySumBeamformer(
                fs=16000, frame_size=500, hop=200, mic_spacing_m=0.15
            )
