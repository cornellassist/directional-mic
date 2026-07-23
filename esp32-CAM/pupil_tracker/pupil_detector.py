"""Pupil-ellipse detection for near-eye camera frames.

Core detection steps (crop, darkest-region search, adaptive threshold,
square mask, contour filtering, ellipse fit) are ported from
JEOresearch/EyeTracker's ``OrloskyPupilDetectorLite.py``
(https://github.com/JEOresearch/EyeTracker, MIT licensed -- see
``LICENSE`` in this directory), stripped of its Tk file picker / video
writer and wrapped in a single ``detect_pupil()`` entry point that
returns normalized coordinates instead of driving its own display loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PupilResult:
    found: bool
    center_px: tuple[float, float]  # pupil center, pixel coords in `frame`
    center_norm: tuple[float, float]  # pupil center normalized to [0, 1] in `frame`
    axes_px: tuple[float, float]  # fitted ellipse (minor, major) axis lengths
    angle_deg: float
    frame: np.ndarray  # the cropped/resized BGR frame the result is relative to


def crop_to_aspect_ratio(image: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    current_height, current_width = image.shape[:2]
    desired_ratio = width / height
    current_ratio = current_width / current_height

    if current_ratio > desired_ratio:
        new_width = int(desired_ratio * current_height)
        offset = (current_width - new_width) // 2
        cropped = image[:, offset:offset + new_width]
    else:
        new_height = int(current_width / desired_ratio)
        offset = (current_height - new_height) // 2
        cropped = image[offset:offset + new_height, :]

    return cv2.resize(cropped, (width, height))


def apply_binary_threshold(image: np.ndarray, darkest_pixel_value: int, added_threshold: int) -> np.ndarray:
    threshold = darkest_pixel_value + added_threshold
    _, thresholded = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    return thresholded


def get_darkest_area(image: np.ndarray) -> tuple[int, int] | None:
    """Scan a coarse grid for the darkest small patch (the pupil is the darkest region)."""
    ignore_bounds = 20
    image_skip = 10
    search_area = 20
    internal_skip = 5

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_sum = float("inf")
    darkest_point = None

    for y in range(ignore_bounds, gray.shape[0] - ignore_bounds, image_skip):
        for x in range(ignore_bounds, gray.shape[1] - ignore_bounds, image_skip):
            current_sum = 0
            num_pixels = 0
            for dy in range(0, search_area, internal_skip):
                if y + dy >= gray.shape[0]:
                    break
                for dx in range(0, search_area, internal_skip):
                    if x + dx >= gray.shape[1]:
                        break
                    current_sum += int(gray[y + dy][x + dx])
                    num_pixels += 1

            if num_pixels > 0 and current_sum < min_sum:
                min_sum = current_sum
                darkest_point = (x + search_area // 2, y + search_area // 2)

    return darkest_point


def mask_outside_square(image: np.ndarray, center: tuple[int, int], size: int) -> np.ndarray:
    x, y = center
    half = size // 2
    mask = np.zeros_like(image)
    top_left_x = max(0, x - half)
    top_left_y = max(0, y - half)
    bottom_right_x = min(image.shape[1], x + half)
    bottom_right_y = min(image.shape[0], y + half)
    mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 255
    return cv2.bitwise_and(image, mask)


def filter_contours_by_area_and_return_largest(contours, pixel_thresh: int, ratio_thresh: float):
    max_area = 0
    largest = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < pixel_thresh:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        length_to_width_ratio = max(w / h, h / w)
        if length_to_width_ratio <= ratio_thresh and area > max_area:
            max_area = area
            largest = contour
    return [largest] if largest is not None else []


def detect_pupil(frame: np.ndarray, added_threshold: int = 15, mask_size: int = 250) -> PupilResult:
    """Locate the pupil ellipse in a single near-eye camera frame.

    Returns a `PupilResult` with `found=False` and a best-effort center
    (the darkest-patch location) if no ellipse could be fit -- callers
    should check `found` before trusting `center_norm` for gaze mapping.
    """
    frame = crop_to_aspect_ratio(frame)
    height, width = frame.shape[:2]

    darkest_point = get_darkest_area(frame)
    if darkest_point is None:
        return PupilResult(False, (width / 2, height / 2), (0.5, 0.5), (0.0, 0.0), 0.0, frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    darkest_value = int(gray[darkest_point[1], darkest_point[0]])
    thresholded = apply_binary_threshold(gray, darkest_value, added_threshold)
    thresholded = mask_outside_square(thresholded, darkest_point, mask_size)

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresholded, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = filter_contours_by_area_and_return_largest(contours, pixel_thresh=1000, ratio_thresh=3)

    if not candidates or len(candidates[0]) <= 5:
        px, py = darkest_point
        return PupilResult(False, (px, py), (px / width, py / height), (0.0, 0.0), 0.0, frame)

    (cx, cy), (minor, major), angle = cv2.fitEllipse(candidates[0])
    return PupilResult(
        found=True,
        center_px=(cx, cy),
        center_norm=(cx / width, cy / height),
        axes_px=(minor, major),
        angle_deg=angle,
        frame=frame,
    )


def draw_debug_overlay(result: PupilResult) -> np.ndarray:
    """Return a copy of `result.frame` with the fitted ellipse and center drawn on it."""
    frame = result.frame.copy()
    if result.found:
        ellipse = (result.center_px, result.axes_px, result.angle_deg)
        cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
        cv2.circle(frame, (int(result.center_px[0]), int(result.center_px[1])), 3, (255, 255, 0), -1)
    return frame
