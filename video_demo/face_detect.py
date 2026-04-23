"""Small-YOLO face detection from a camera, with face positions usable as
steering input for a directional-microphone array.

Run:
    python face_detect.py                    # webcam 0, live window
    python face_detect.py --source 1         # pick a different camera
    python face_detect.py --source video.mp4 # from a file
    python face_detect.py --no-show --json   # headless, emit JSON lines on stdout
    python face_detect.py --fov 70           # horizontal FOV for azimuth output

The model file `yolov8n-face.pt` auto-downloads on first run into this folder.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import cv2
from ultralytics import YOLO


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "yolov8n-face.pt"
MODEL_URL = (
    "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt"
)


@dataclass
class FaceDetection:
    bbox_xyxy: tuple[float, float, float, float]  # pixel coords
    conf: float
    center_norm: tuple[float, float]              # (x, y) in [0, 1]
    azimuth_deg: float | None                     # horizontal angle from camera boresight
    elevation_deg: float | None                   # vertical angle from boresight

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox_xyxy"] = list(self.bbox_xyxy)
        d["center_norm"] = list(self.center_norm)
        return d


def ensure_model(path: Path) -> Path:
    if path.exists():
        return path
    print(f"[face_detect] downloading {path.name} -> {path}", file=sys.stderr)
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, path)
    return path


def angle_from_norm(u: float, fov_deg: float | None) -> float | None:
    """Pinhole-ish mapping from normalized coord in [0,1] to angle in degrees.

    Uses atan so wide-FOV cameras don't get exaggerated near the edges:
        f = 0.5 / tan(FOV/2)
        angle = atan((u - 0.5) / f)
    """
    if fov_deg is None:
        return None
    half = math.radians(fov_deg) / 2.0
    f = 0.5 / math.tan(half)
    return math.degrees(math.atan((u - 0.5) / f))


def detect_stream(
    source: int | str = 0,
    model_path: Path | str = DEFAULT_MODEL,
    conf: float = 0.5,
    hfov_deg: float | None = 68.0,
    vfov_deg: float | None = None,
) -> Iterator[tuple["cv2.Mat", list[FaceDetection], float]]:
    """Yield (frame, detections, timestamp) for each captured frame."""
    model = YOLO(str(ensure_model(Path(model_path))))
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video source: {source!r}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = time.time()
            fh, fw = frame.shape[:2]
            results = model.predict(frame, conf=conf, verbose=False)
            faces: list[FaceDetection] = []
            if results:
                r0 = results[0]
                if r0.boxes is not None and len(r0.boxes) > 0:
                    xyxy = r0.boxes.xyxy.cpu().numpy()
                    confs = r0.boxes.conf.cpu().numpy()
                    for (x1, y1, x2, y2), c in zip(xyxy, confs):
                        cx_n = ((x1 + x2) * 0.5) / fw
                        cy_n = ((y1 + y2) * 0.5) / fh
                        faces.append(
                            FaceDetection(
                                bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                                conf=float(c),
                                center_norm=(float(cx_n), float(cy_n)),
                                azimuth_deg=angle_from_norm(cx_n, hfov_deg),
                                elevation_deg=angle_from_norm(cy_n, vfov_deg),
                            )
                        )
            yield frame, faces, ts
    finally:
        cap.release()


def draw_overlay(frame, faces: Iterable[FaceDetection]) -> None:
    for f in faces:
        x1, y1, x2, y2 = map(int, f.bbox_xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{f.conf:.2f}"
        if f.azimuth_deg is not None:
            label += f"  az={f.azimuth_deg:+.1f}°"
        cv2.putText(
            frame, label, (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default="0",
                   help="camera index (int) or path to video file (default: 0)")
    p.add_argument("--model", default=str(DEFAULT_MODEL),
                   help="path to YOLO face model .pt (auto-downloads if missing)")
    p.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    p.add_argument("--fov", type=float, default=68.0,
                   help="horizontal FOV of the camera in degrees, for azimuth output")
    p.add_argument("--vfov", type=float, default=None,
                   help="vertical FOV in degrees, for elevation output (optional)")
    p.add_argument("--no-show", action="store_true", help="don't open a preview window")
    p.add_argument("--json", action="store_true",
                   help="emit one JSON line per frame on stdout (for piping to mic code)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source: int | str = int(args.source) if args.source.isdigit() else args.source

    stream = detect_stream(
        source=source,
        model_path=args.model,
        conf=args.conf,
        hfov_deg=args.fov,
        vfov_deg=args.vfov,
    )

    try:
        for frame, faces, ts in stream:
            if args.json:
                print(
                    json.dumps({"t": ts, "faces": [f.to_dict() for f in faces]}),
                    flush=True,
                )
            if not args.no_show:
                draw_overlay(frame, faces)
                cv2.imshow("YOLO face detection", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if not args.no_show:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
