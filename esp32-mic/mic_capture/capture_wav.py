#!/usr/bin/env python3
"""Capture interleaved int16 stereo PCM streamed from PdmMicTest.ino and save a WAV.

Use with the STREAM_RAW build of the sketch (no text on the wire, just PCM).
The two channels are Left = mic with SEL->3V3, Right = mic with SEL->GND
(SEL mapping is inverted from the Adafruit datasheet on this board).

    python capture_wav.py --port /dev/cu.usbmodem101 --seconds 5

Open the result in Audacity to check the three bring-up facts:
  1. both channels carry signal,
  2. levels match for an equidistant source,
  3. a single snap lands within ~1 sample on both channels (phase coherence).
"""
import argparse
import sys
import time
import wave

import serial  # pip install pyserial


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", required=True, help="serial port, e.g. /dev/cu.usbmodem101")
    ap.add_argument("--seconds", type=float, default=5.0, help="capture duration")
    ap.add_argument("--rate", type=int, default=16000, help="must match SAMPLE_RATE in the sketch")
    ap.add_argument("--out", default="mic_test.wav", help="output WAV path")
    args = ap.parse_args()

    with serial.Serial(args.port, 115200, timeout=1) as ser:
        time.sleep(1.0)            # let the board settle after the USB-CDC reset
        ser.reset_input_buffer()   # drop any boot-time garbage

        frame = 4  # 2 channels * int16
        total = int(args.rate * frame * args.seconds)
        print(f"Recording {args.seconds:.1f}s from {args.port} ...")

        data = bytearray()
        while len(data) < total:
            chunk = ser.read(min(4096, total - len(data)))
            if not chunk:
                print("timeout waiting for data — is the STREAM_RAW build flashed?",
                      file=sys.stderr)
                break
            data.extend(chunk)

    data = data[: len(data) - (len(data) % frame)]  # whole stereo frames only
    with wave.open(args.out, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(args.rate)
        w.writeframes(data)

    frames = len(data) // frame
    print(f"Wrote {args.out}: {frames} frames ({frames / args.rate:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
