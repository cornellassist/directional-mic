"""Sanity-check `pupil_detector` against your laptop webcam -- no ESP32
hardware required.

This does *not* test calibration or the websocket server, just whether
`detect_pupil()` finds a sane ellipse given a live feed. A laptop webcam
at normal distance won't look like the near-eye mount the ESP32-CAM is
meant to use, so hold it close to one eye (a few inches) for a
meaningful test -- from further away it'll just lock onto whatever's
darkest in frame (eyebrows, nostrils, background).

Usage:
    python test_webcam.py [--camera 0]

Keys:
    q  quit

MOST IMPORTANT PART!!!!!!
In terminal:
    navigate to pupil_tracker
    ./.venv/bin/python test_webcam.py
"""

from __future__ import annotations

import argparse

import cv2

from pupil_detector import detect_pupil, draw_debug_overlay


def main() -> int:
    p = argparse.ArgumentParser(description="Run pupil detection on a local webcam.")
    p.add_argument("--camera", type=int, default=0, help="cv2 camera index")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera index {args.camera}")

    print("Press 'q' to quit.", flush=True)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("no frame from camera", flush=True)
                break

            result = detect_pupil(frame)
            overlay = draw_debug_overlay(result)
            status = "found" if result.found else "not found"
            cx, cy = result.center_norm
            cv2.putText(
                overlay, f"pupil: {status}  center_norm=({cx:.2f}, {cy:.2f})",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
            cv2.imshow("pupil_detector test (webcam)", overlay)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
