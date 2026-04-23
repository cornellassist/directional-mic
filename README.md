# directional-mic



How to run the program. 

Open on different terminals
``` bash
uv run python -c "import sounddevice as sd; print(sd.query_devices())" #query the device number

uv run python eye_tracking/mock_gaze_server.py --pattern mouse #mouse server

uv run python -m directional_mic.runtime --inputs 0,1 --output-device 4 #The main program

uv run python -m directional_mic.monitor_mics --inputs 0,1 #Monitoring the raw inputs from the microphoen
```
How to set up the dual microphone devices.
Install the `megaphone` from app store, connect the computer and the phone using the usb cables. On the app choose the computer as the output device.

