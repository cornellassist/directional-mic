"""Read the ESP32-S3-CAM MJPEG stream and display it in OpenCV.

Usage:
    python stream_view.py --ip 172.20.10.5
    python stream_view.py --url http://172.20.10.5:81/stream

Keys:
    q  quit
    s  save current frame to ./frames/
"""

import argparse
import os
import time

import cv2


def build_url(ip: str) -> str:
    # port 81 /stream is the raw MJPEG endpoint served by CameraWebServer
    return f"http://{ip}:81/stream"


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ip", help="ESP32 IP, e.g. 172.20.10.5")
    g.add_argument("--url", help="full stream URL")
    ap.add_argument("--save-dir", default="frames", help="where 's' saves frames")
    args = ap.parse_args()

    url = args.url or build_url(args.ip)
    print(f"Connecting to {url} ...")

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open stream {url}\n"
            "Check: board is powered, on the same network, and the IP is correct."
        )

    os.makedirs(args.save_dir, exist_ok=True)
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No frame (stream dropped). Retrying...")
            time.sleep(0.5)
            continue

        cv2.imshow("esp32-cam", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            path = os.path.join(args.save_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(path, frame)
            print(f"saved {path}")
            saved += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
