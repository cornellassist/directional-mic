# ESP32-S3 · 2-mic PDM bring-up test

Confirm two **Adafruit PDM MEMS mics (MP34DT01, #3492)** capture cleanly and
**sample-synchronously** on a **GOOUUU ESP32-S3-CAM** before building the full
6-mic array. This is the one experiment that de-risks everything: if a shared
clock gives you two phase-coherent channels, the rest of the array is just more
of the same.

Two parts:

| Folder | Side | What |
|---|---|---|
| [`PdmMicTest/`](PdmMicTest) | ESP32 firmware | Reads 2 PDM mics; live level meter, or raw PCM stream |
| [`mic_capture/`](mic_capture) | Python | Saves the raw stream to a stereo WAV |

## Wiring — 2 mics, one shared clock, one data line

The pair shares a single `CLK` **and** a single `DAT` line. `SEL` decides which
clock half each mic drives, so one becomes Left and the other Right.

> **`SEL` mapping is inverted from the Adafruit datasheet on this board** —
> verified empirically: **Left mic → `SEL` = 3V3 (high)**, **Right mic → `SEL` =
> GND (low)**. If a channel is silent, this is the first thing to swap.

| Breakout pin | Mic 1 (Left) | Mic 2 (Right) |
|---|---|---|
| `3V` | 3V3 | 3V3 |
| `GND` | GND | GND |
| `CLK` | GPIO 42 (shared) | GPIO 42 (shared) |
| `DAT` | GPIO 41 (shared) | GPIO 41 (shared) |
| `SEL` | **3V3** | **GND** |

Pins are set at the top of `PdmMicTest.ino`. Use any two **free** GPIOs on your
header — avoid camera pins 4–18, PSRAM/flash 26–37, strapping pins 0/3/45/46,
and the native-USB pins 19/20. If 42 or 41 collides with your board's onboard
LED, move it and update the sketch.

## Board Settings (Arduino IDE → Tools)

Same as [`../esp32-CAM`](../esp32-CAM/README.md); the key ones:

| Setting | Value |
|---|---|
| Board | ESP32S3 Dev Module |
| PSRAM | OPI PSRAM (N16R8 = octal) |
| Flash Size | 16MB (128Mb) |
| USB CDC On Boot | Enabled |
| Port | `/dev/cu.usbmodem*` (re-check after each mode switch) |

Requires **esp32 core v3.x** (Boards Manager → "esp32" by Espressif) for the
`ESP_I2S` library.

## Test procedure

### 1. Level meter (fastest — no PC-side code)

Flash `PdmMicTest.ino` as-is, open **Serial Monitor @ 115200**. You'll see:

```
L ####### (1420)   R ######## (1533)
```

- **Both bars move** when you tap/talk near the matching mic → wiring + SEL OK.
- **Levels track** for a source equidistant from both mics → mics are matched.
- If only one bar ever moves, swap that mic's `SEL` and recheck the solder joints.

### 2. Capture a stereo WAV (raw PCM)

Uncomment `#define STREAM_RAW` in the sketch, re-flash, **close Serial Monitor**
(only one program can hold the port), then from the repo root:

```bash
# pyserial is already a project dependency (uv add pyserial)
uv run python esp32-mic/mic_capture/capture_wav.py \
    --port /dev/cu.usbmodem2101 --seconds 10 --out mic_test.wav
```

The stream is already **decoded PCM** — the ESP32 I2S PDM-RX peripheral converts
the 1-bit PDM bitstream to 16-bit PCM in hardware, so the WAV plays in any
player with no decoding on the host side. Left channel = `SEL`→3V3 mic, Right =
`SEL`→GND mic.

### 3. Verify the capture

The mics have **no gain control**, so distance is the volume knob. At ~30 cm and
normal speech a take sits around −30 dBFS with plenty of headroom; leaning in and
talking loudly **clips** (permanent distortion — no post-processing recovers it).
A quick check with numpy/scipy (both already in the project env):

- **Both channels present and level-matched** (rms within ~10–15 %).
- **0 clipped samples** (peak below ~0 dBFS). If it clips, move back and redo.
- **Inter-channel lag of ~0–3 samples** at high correlation. This is the
  *acoustic* time-of-arrival difference between the two mic positions — the
  directional cue beamforming steers on. The shared clock removes *electronic*
  skew, not this; a 1-sample lag @16 kHz ≈ 2 cm of extra sound path.

## Bring-up result (verified July 2026)

Both mics validated end-to-end on a GOOUUU ESP32-S3-CAM, GPIO 42 (CLK) / 41 (DAT):

- ✅ Both channels live, clean, and audible individually.
- ✅ Level-matched off a shared source (rms within ~10 %).
- ✅ **Phase-coherent**: normalized correlation ~0.9 at a 1-sample lag around a
  transient — the two channels are sample-synchronous off the shared clock.
- ✅ Correct L/R split via the inverted-`SEL` wiring (Left = 3V3, Right = GND) —
  see the wiring note above; this cost some debugging.

### Why the two mics sound bad *summed* (but fine individually)

Each mic alone is a clean recording. Naively adding L+R sounds hollow/phasey
("失真"-like) — this is **comb filtering**, not a bug: two copies of the same
sound offset by the acoustic lag cancel at some frequencies and reinforce at
others. Full-clip correlation is only ~0.5–0.6 because room reverberation
arrives at the two mics differently and no single delay aligns it.

**This is exactly why beamforming exists.** You do **not** pre-sum the channels —
you feed both, separately, to the beamformer, which delay-aligns to the target
direction (and, for these low-coherence recordings, subtracts leakage) before
combining. Keep L and R as separate files; let the beamformer combine them. See
[`../video_demo/beamforming_config.yaml`](../video_demo/beamforming_config.yaml),
which notes the `video_demo` recordings have the same low cross-correlation.

## Notes / gotchas

- The meter build prints text; the raw build prints **only PCM**. Don't run
  `capture_wav.py` against the meter build — it will record the ASCII banner.
- 16 kHz stereo = 64 KB/s — trivial over the S3's native USB CDC (baud is
  nominal on USB, so 115200 is not a real bottleneck).
- `SAMPLE_RATE` in the sketch and `--rate` in the script must match.
- When splitting the stereo WAV into per-mic files for the beamformer, apply
  **one shared gain** to both channels — normalizing each channel independently
  destroys the relative L/R level the beamformer relies on.
- "Uploads fine but Serial Monitor is blank" → **USB CDC On Boot** was disabled;
  enable it and re-flash (routes `Serial` to native USB, not the UART pins).

## To-Do

- [x] Confirm 2 mics capture cleanly, level-matched, phase-coherent (July 2026)
- [x] Confirm GPIO 42/41 are free on this GOOUUU board revision
- [ ] Feed the synchronized capture (`audio_left.wav` / `audio_right.wav`) into
      the `directional_mic` beamformer
- [ ] Scale to 4 → 6 mics via multi-data-line PDM RX (ESP-IDF v5.x `slot_mask`;
      the Arduino wrapper only does 2) — see [`../BOM.md`](../BOM.md)
