"""Pupil-position -> screen-gaze calibration.

`pupil_detector.detect_pupil()` only tells you where the pupil sits
*within the eye-camera image* -- not where on screen the user is
looking. Turning one into the other needs a per-session calibration
pass, the same way commodity webcam eye trackers do it: show the user
a handful of known screen points, record where the pupil sits while
they fixate on each one, and fit a mapping between the two spaces.

This fits a 2nd-order polynomial (6 terms per axis: 1, x, y, xy, x^2,
y^2), which tolerates the mild nonlinearity of eye rotation better than
a plain affine fit while still only needing ~6-9 calibration points.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import cv2
import numpy as np

from pupil_detector import detect_pupil

DEFAULT_GRID: list[tuple[float, float]] = [
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
    (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
]


def _poly_terms(px: float, py: float) -> np.ndarray:
    return np.array([1.0, px, py, px * py, px * px, py * py])


@dataclass
class Calibration:
    coeffs_x: np.ndarray  # shape (6,)
    coeffs_y: np.ndarray  # shape (6,)

    def apply(self, px: float, py: float) -> tuple[float, float]:
        terms = _poly_terms(px, py)
        sx = float(np.dot(self.coeffs_x, terms))
        sy = float(np.dot(self.coeffs_y, terms))
        return min(1.0, max(0.0, sx)), min(1.0, max(0.0, sy))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"coeffs_x": self.coeffs_x.tolist(), "coeffs_y": self.coeffs_y.tolist()}, f)

    @classmethod
    def load(cls, path: str) -> "Calibration":
        with open(path) as f:
            data = json.load(f)
        return cls(np.array(data["coeffs_x"]), np.array(data["coeffs_y"]))


def fit(samples: list[tuple[tuple[float, float], tuple[float, float]]]) -> Calibration:
    """`samples`: list of ((pupil_x, pupil_y), (screen_x, screen_y)), all normalized [0, 1]."""
    if len(samples) < 6:
        raise ValueError("need at least 6 calibration samples to fit the polynomial mapping")
    a = np.array([_poly_terms(px, py) for (px, py), _ in samples])
    bx = np.array([sx for _, (sx, _sy) in samples])
    by = np.array([sy for _, (_sx, sy) in samples])
    coeffs_x, *_ = np.linalg.lstsq(a, bx, rcond=None)
    coeffs_y, *_ = np.linalg.lstsq(a, by, rcond=None)
    return Calibration(coeffs_x, coeffs_y)


def run_calibration(
    cap: cv2.VideoCapture,
    screen_size: tuple[int, int] = (1280, 720),
    grid: list[tuple[float, float]] = DEFAULT_GRID,
    samples_per_point: int = 20,
    dwell_s: float = 1.5,
    window_name: str = "calibration",
) -> Calibration:
    """Interactive calibration: show a dot at each `grid` point, sample the pupil while
    the user fixates on it, then fit the pupil -> screen mapping. Runs entirely on the
    calling thread (OpenCV's HighGUI windows are not safe to drive from a background
    thread), so call this before starting the websocket server, not concurrently with it.

    Press 'q' at any point to abort (raises KeyboardInterrupt).
    """
    width, height = screen_size
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    samples: list[tuple[tuple[float, float], tuple[float, float]]] = []

    for gx, gy in grid:
        dot = (int(gx * width), int(gy * height))
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.circle(canvas, dot, 14, (0, 0, 255), -1)

        t_start = time.time()
        while time.time() - t_start < dwell_s:
            cv2.imshow(window_name, canvas)
            cap.read()  # drain the buffer so sampling starts on a fresh frame
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyWindow(window_name)
                raise KeyboardInterrupt("calibration aborted")

        collected: list[tuple[float, float]] = []
        while len(collected) < samples_per_point:
            ok, frame = cap.read()
            cv2.imshow(window_name, canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyWindow(window_name)
                raise KeyboardInterrupt("calibration aborted")
            if not ok:
                continue
            result = detect_pupil(frame)
            if result.found:
                collected.append(result.center_norm)

        px = float(np.mean([c[0] for c in collected]))
        py = float(np.mean([c[1] for c in collected]))
        samples.append(((px, py), (gx, gy)))
        print(f"[calibration] point ({gx:.1f}, {gy:.1f}) -> pupil ({px:.3f}, {py:.3f})", flush=True)

    cv2.destroyWindow(window_name)
    return fit(samples)
