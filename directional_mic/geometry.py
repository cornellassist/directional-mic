"""Pure geometry helpers: gaze-to-azimuth and azimuth-to-TDOA.

No state, no I/O — everything here is unit-testable without hardware.
"""

from __future__ import annotations

import math
from typing import Literal

SPEED_OF_SOUND_M_S = 343.0

Orientation = Literal["broadside", "endfire"]


def gaze_to_azimuth(
    x: float,
    screen_width_cm: float,
    view_distance_cm: float,
) -> float:
    """Map horizontal gaze x ∈ [0, 1] to azimuth (radians).

    Assumes the user sits centered in front of a monitor of width
    ``screen_width_cm`` at distance ``view_distance_cm``. Returns 0 when
    the gaze is at the horizontal center, positive toward the right side
    of the screen, negative toward the left. The gaze ``y`` is ignored —
    a 2-mic horizontal array cannot resolve elevation.
    """
    if screen_width_cm <= 0 or view_distance_cm <= 0:
        raise ValueError("screen_width_cm and view_distance_cm must be positive")
    offset_cm = (x - 0.5) * screen_width_cm
    return math.atan2(offset_cm, view_distance_cm)


def tdoa(
    azimuth_rad: float,
    mic_spacing_m: float,
    c: float = SPEED_OF_SOUND_M_S,
    orientation: Orientation = "broadside",
) -> float:
    """Time difference of arrival between the two mics, in seconds.

    Sign convention: positive TDOA means the source reaches mic 2 *later*
    than mic 1, i.e. channel 2 must be advanced (or channel 1 delayed) to
    align them for constructive summation.

    - ``broadside``: mics lie along the left-right axis; a source at
      azimuth 0 (straight ahead) is equidistant from both mics.
      TDOA = (d / c) · sin(θ).
    - ``endfire``: mics lie along the front-back axis; a source straight
      ahead (θ = 0) reaches the front mic first.
      TDOA = (d / c) · cos(θ).
    """
    if mic_spacing_m <= 0:
        raise ValueError("mic_spacing_m must be positive")
    if orientation == "broadside":
        return (mic_spacing_m / c) * math.sin(azimuth_rad)
    if orientation == "endfire":
        return (mic_spacing_m / c) * math.cos(azimuth_rad)
    raise ValueError(f"unknown orientation: {orientation!r}")
