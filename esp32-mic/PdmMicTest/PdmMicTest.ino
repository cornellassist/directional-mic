// PdmMicTest — bring-up test for two Adafruit PDM MEMS mics (MP34DT01, #3492)
// on a GOOUUU ESP32-S3-CAM (ESP32-S3-WROOM-1 N16R8).
//
// Two mics share ONE clock and ONE data line. The SEL pin picks the clock half.
// Verified empirically on this board (mapping is INVERTED from the Adafruit
// datasheet convention):
//   Left  channel mic  SEL -> 3V3  (high)
//   Right channel mic  SEL -> GND  (low)
// Sharing the clock is what makes the pair sample-synchronous (the whole point).
//
// Requires the ESP32 Arduino core v3.x (ships the ESP_I2S library; PDM stereo RX
// = exactly 2 mics, which is all this test needs).
//
// ---- Output mode -----------------------------------------------------------
//   Default            : human-readable L/R level meter over Serial @115200.
//                        Talk to / tap each mic; both bars should move.
//   #define STREAM_RAW  : stream interleaved int16 PCM (L,R,L,R ...) with NO
//                        text, for esp32-mic/mic_capture/capture_wav.py to save
//                        a stereo WAV you can open in Audacity to check phase.
// ----------------------------------------------------------------------------

#include <ESP_I2S.h>
#include <math.h>

#define STREAM_RAW   // <-- uncomment to stream raw PCM instead of the meter

// ---- Wiring: any two FREE GPIOs broken out on your header --------------------
// Avoid: camera pins 4-18, PSRAM/flash 26-37, strapping pins 0/3/45/46, and the
// native-USB pins 19/20. Verify 21 & 47 are exposed on your GOOUUU board; if a
// pin collides with the onboard LED, move it. Both mics wire to the SAME pins.
const int PDM_CLK  = 42;   // -> both mics' CLK pin
const int PDM_DATA = 41;   // -> both mics' DAT pin
// -----------------------------------------------------------------------------

const uint32_t SAMPLE_RATE = 16000;   // matches the beamformer's working rate

I2SClass I2S;
static int16_t buf[512];              // 256 stereo frames per read

static void printBar(char tag, int rms) {
  int bars = rms / 200;
  if (bars > 40) bars = 40;
  Serial.print(tag);
  Serial.print(' ');
  for (int i = 0; i < bars; i++) Serial.print('#');
  Serial.print(" (");
  Serial.print(rms);
  Serial.print(")  ");
}

void setup() {
  Serial.begin(115200);
  I2S.setPinsPdmRx(PDM_CLK, PDM_DATA);
  if (!I2S.begin(I2S_MODE_PDM_RX, SAMPLE_RATE,
                 I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO)) {
    Serial.println("ERR: I2S PDM init failed (check core v3.x + pin choice)");
    while (true) delay(1000);
  }
#ifndef STREAM_RAW
  Serial.println("PDM mic meter — tap/talk each mic; both bars should move.");
#endif
}

void loop() {
  size_t n = I2S.readBytes((char *)buf, sizeof(buf));
  if (n == 0) return;
  size_t frames = n / (2 * sizeof(int16_t));

#ifdef STREAM_RAW
  Serial.write((uint8_t *)buf, frames * 2 * sizeof(int16_t));
#else
  // Remove per-channel DC first — PDM decoding leaves a large DC bias that
  // would otherwise swamp the RMS and make a silent channel look "loud".
  long dcL = 0, dcR = 0;
  for (size_t i = 0; i < frames; i++) {
    dcL += buf[2 * i];
    dcR += buf[2 * i + 1];
  }
  int meanL = (int)(dcL / (long)frames);
  int meanR = (int)(dcR / (long)frames);

  // RMS of the AC (sound) component only.
  double sumL = 0, sumR = 0;
  for (size_t i = 0; i < frames; i++) {
    double l = buf[2 * i]     - meanL;
    double r = buf[2 * i + 1] - meanR;
    sumL += l * l;
    sumR += r * r;
  }
  int rmsL = (int)sqrt(sumL / frames);
  int rmsR = (int)sqrt(sumR / frames);
  printBar('L', rmsL);
  printBar('R', rmsR);
  Serial.println();
  delay(80);
#endif
}
