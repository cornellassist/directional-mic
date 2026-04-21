import math

import pytest

from directional_mic.geometry import SPEED_OF_SOUND_M_S, gaze_to_azimuth, tdoa


class TestGazeToAzimuth:
    def test_center_gaze_is_zero(self):
        assert gaze_to_azimuth(0.5, 60.0, 60.0) == 0.0

    def test_right_edge_with_square_geometry(self):
        # x=1.0, W=60, D=60 -> atan2(30, 60) ≈ 26.5651°
        angle = gaze_to_azimuth(1.0, 60.0, 60.0)
        assert math.isclose(angle, math.atan2(30.0, 60.0))
        assert math.isclose(math.degrees(angle), 26.56505117707799)

    def test_left_edge_is_negative(self):
        assert gaze_to_azimuth(0.0, 60.0, 60.0) == -gaze_to_azimuth(1.0, 60.0, 60.0)

    def test_rejects_nonpositive_screen(self):
        with pytest.raises(ValueError):
            gaze_to_azimuth(0.5, 0.0, 60.0)
        with pytest.raises(ValueError):
            gaze_to_azimuth(0.5, 60.0, -1.0)


class TestTdoa:
    def test_broadside_straight_ahead_is_zero(self):
        assert tdoa(0.0, mic_spacing_m=0.15) == 0.0

    def test_broadside_ninety_degrees(self):
        # 15 cm / 343 m/s ≈ 437 µs
        t = tdoa(math.pi / 2, mic_spacing_m=0.15)
        assert math.isclose(t, 0.15 / SPEED_OF_SOUND_M_S)
        assert math.isclose(t, 4.3731778e-4, rel_tol=1e-5)

    def test_broadside_sign_flips_with_azimuth(self):
        t_pos = tdoa(math.radians(30), mic_spacing_m=0.15)
        t_neg = tdoa(math.radians(-30), mic_spacing_m=0.15)
        assert math.isclose(t_pos, -t_neg)
        assert t_pos > 0

    def test_endfire_straight_ahead_is_max(self):
        t_front = tdoa(0.0, mic_spacing_m=0.15, orientation="endfire")
        t_side = tdoa(math.pi / 2, mic_spacing_m=0.15, orientation="endfire")
        assert math.isclose(t_front, 0.15 / SPEED_OF_SOUND_M_S)
        assert abs(t_side) < 1e-12

    def test_rejects_bad_orientation(self):
        with pytest.raises(ValueError):
            tdoa(0.0, mic_spacing_m=0.15, orientation="diagonal")  # type: ignore[arg-type]

    def test_rejects_nonpositive_spacing(self):
        with pytest.raises(ValueError):
            tdoa(0.0, mic_spacing_m=0.0)
