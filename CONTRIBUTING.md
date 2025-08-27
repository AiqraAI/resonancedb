# 🤝 How to Contribute

We welcome contributions in:
- Data
- Code
- Models
- Hardware
- Docs

## 🧪 Submit Data

### Option 1: Real Sensor (ESP32 + MPU-6050)
1. Flash firmware from `/firmware/esp32-tap-recorder`
2. Record tap on a known material
3. Save as JSON using the schema in `DATA_FORMAT.md`
4. Submit via PR to `/data/`

### Option 2: Simulated Data
Use `python/simulate_tap.py` to generate synthetic samples.

### Option 3: Phone Sensors
Use apps like **Phyphox** (Android/iOS) to record taps → export CSV → convert to JSON.

## 🛠️ Improve Code
- Fix bugs
- Add signal processing features
- Optimize for microcontrollers

## 🤖 Train Models
- Use data to train classifiers
- Save as `.tflite` and submit to `/models/`

## 🧰 Share Hardware
- Design 3D-printable mounts
- Build low-power sensor nodes