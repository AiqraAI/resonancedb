// ResonanceDB: Tap Recorder with ESP32 + MPU-6050
// By AIQRA AI
// https://github.com/AiqraAI/resonancedb

#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;
bool dmpReady = false;
float ax, ay, az;
unsigned long lastTap = 0;
bool recording = false;
int sampleCount = 0;
const int maxSamples = 2000;  // ~2 seconds at 1kHz
float recordedData[maxSamples];
int recordIndex = 0;

void setup() {
  Serial.begin(115200);
  Wire.begin();

  // Initialize MPU-6050
  if (mpu.begin() != 0) {
    Serial.println("❌ MPU6050 not found!");
    while (1);
  }
  Serial.println("✅ MPU6050 connected!");

  mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
  mpu.setFilterBandwidth(MPU6050_BAND_218_HZ);

  Serial.println("Tap the sensor to start recording...");
}

void loop() {
  mpu.getAcceleration(&ax, &ay, &az);
  float total = sqrt(ax*ax + ay*ay + az*az);  // Total acceleration
  float vibration = total - 1.0;  // Remove gravity

  // Detect tap: sudden spike in acceleration
  if (vibration > 0.5 && millis() - lastTap > 1000) {
    lastTap = millis();
    startRecording();
  }

  // Record for ~2 seconds
  if (recording) {
    recordedData[recordIndex] = vibration;
    recordIndex++;
    if (recordIndex >= maxSamples) {
      stopRecording();
    }
  }

  delay(1);  // ~1000 Hz sampling
}

void startRecording() {
  recording = true;
  recordIndex = 0;
  Serial.println("RECORDING_START");
}

void stopRecording() {
  recording = false;
  Serial.println("RECORDING_END");

  // Print data as comma-separated values
  for (int i = 0; i < maxSamples; i++) {
    Serial.print(recordedData[i], 6);  // 6 decimal places
    if (i < maxSamples - 1) Serial.print(",");
  }
  Serial.println();
  Serial.println("READY");
}
