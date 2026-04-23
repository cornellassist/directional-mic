# Recording iPhone + iPad simultaneously on macOS

## What works: one Aggregate Device, one stream

macOS Core Audio can combine the iPhone Continuity Microphone and the USB-attached iPad into a single input via **Audio MIDI Setup → Aggregate Device**. Opening that aggregate as one stream lets Core Audio arbitrate routing and clocks, which sidesteps every conflict we hit with independent streams.

### One-time setup

1. Open **Audio MIDI Setup** (`/System/Applications/Utilities/Audio MIDI Setup.app`).
2. `+` → **Create Aggregate Device**.
3. Check the iPhone Continuity Microphone **and** the iPad as subdevices. Put the iPhone first (1 ch) and the iPad second (2 ch) so channel order is iPhone = ch0, iPad = ch1–2.
4. Pick a clock source (either subdevice is fine) and enable drift correction on the other.

The aggregate then shows up as a normal input device (name contains "Aggregate Device"). On this machine it currently enumerates at index 9 with 3 input channels @ 48 kHz, but the index is not stable across reboots — look it up by name.

### Recording

```bash
uv run python -m directional_mic.record_aggregate \
    --seconds 5 \
    --iphone-out recording_iphone_agg.wav \
    --ipad-out recording_ipad_agg.wav
```

The script (`directional_mic/record_aggregate.py`):

- finds the aggregate by name (override with `--device N`),
- opens a single 3-channel 48 kHz PortAudio stream via `sounddevice`,
- captures `seconds × samplerate` frames,
- splits ch0 → iPhone WAV (mono) and ch1‑2 → iPad WAV (stereo),
- writes int16 WAVs and prints per-channel peaks.

Each `uv run` invocation is a fresh Python process, which keeps PortAudio state clean. `sd.rec` takes a few seconds longer than the requested duration on first open — that's Core Audio warming up the aggregate, not lost audio; the captured buffer is exactly `seconds × samplerate` frames.

### Sanity check after recording

```python
from scipy.io import wavfile
import numpy as np
for f in ["recording_iphone_agg.wav", "recording_ipad_agg.wav"]:
    sr, data = wavfile.read(f)
    if data.ndim == 1:
        data = data[:, None]
    print(f, sr, data.shape, "peak", np.max(np.abs(data), axis=0))
```

Expected: both files exactly `5.000 s` long, non-zero peaks on every channel. The iPad's two channels are usually identical (mono duplicated), and its input is hot — if you see peak 32767, lower the iPad mic level in **System Settings → Sound**.

## What didn't work (and why)

- **Two PortAudio streams in one Python process** — `-9986 "Audio Hardware Not Running"`. The iPhone Continuity plug-in does not coexist with another live Core Audio input in the same process, and once any stream open fails the host's PortAudio state stays dirty until the process exits.
- **One subprocess per device, launched in parallel (with and without stagger)** — whichever process opens the iPhone Continuity input silences the iPad system-wide (peak = 0), or the iPhone open itself hits -9986; order and head-start do not change this. It's a macOS Core Audio routing behavior, not a PortAudio bug — the Continuity plug-in appears to claim other inputs while it's active. See `directional_mic/record_dual.py` for that (now superseded) approach.
