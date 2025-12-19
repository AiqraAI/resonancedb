# ESP32 Tap Recorder

Simple vibration data recorder using ESP32 and MPU-6050 accelerometer.

## Hardware Requirements

- ESP32 (any variant)
- MPU-6050 Accelerometer/Gyroscope module
- Jumper wires
- Breadboard (optional)

## Wiring

| MPU-6050 | ESP32 |
|----------|-------|
| VCC | 3.3V |
| GND | GND |
| SCL | GPIO 22 |
| SDA | GPIO 21 |

## Installation

1. Install [Arduino IDE](https://www.arduino.cc/en/software)
2. Add ESP32 board support:
   - File → Preferences → Additional Board URLs
   - Add: `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
3. Install libraries:
   - Sketch → Include Library → Manage Libraries
   - Search and install: `MPU6050` by Electronic Cats
4. Open `esp32-tap-recorder.ino`
5. Select your ESP32 board and port
6. Upload!

## Usage

1. Open Serial Monitor (115200 baud)
2. Tap the surface near the sensor
3. Recording starts automatically (triggers on >0.5g spike)
4. After ~2 seconds, data is printed as CSV
5. Copy the output and save as JSON with metadata

## Output Format

```
RECORDING_START
RECORDING_END
0.001234,-0.002345,0.004567,...
READY
```

## Converting to ResonanceDB Format

```python
import json

# Paste your CSV data here
vibration_csv = "0.001234,-0.002345,..."
vibration = [float(x) for x in vibration_csv.split(",")]

sample = {
    "material": "your_material",
    "vibration": vibration,
    "sample_rate_hz": 1000,
    "excitation": "manual_tap",
    "device": "ESP32+MPU6050"
}

with open("my_sample.json", "w") as f:
    json.dump(sample, f, indent=2)
```

## Troubleshooting

- **MPU6050 not found**: Check wiring, ensure I2C address is 0x68
- **No tap detected**: Increase sensitivity by lowering threshold (0.5 → 0.3)
- **Noisy data**: Ensure secure mounting, reduce vibration sources

## License

MIT - Use freely!
