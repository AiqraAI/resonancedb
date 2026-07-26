# 📄 ResonanceDB Data Format

All contributions must follow this JSON schema.

## 🔧 Required Fields

| Field | Type | Description |
|------|------|-------------|
| `material` | string | e.g., `glass`, `oak_wood`, `aluminum` |
| `vibration` | number[] | The signal over time: accelerometer values (g) or normalized audio amplitude in [-1, 1] |
| `sample_rate_hz` | number | Samples per second (e.g., 48000 for audio, 1000 for accelerometers) |
| `excitation` | string | How vibration was created: `manual_tap`, `solenoid`, `ambient`, `simulated` |
| `source` | string | `microphone`, `phone_sensor`, `simulation`, `real` |

## 🌡️ Optional Fields

| Field | Type | Description |
|------|------|-------------|
| `temperature_c` | number | Temperature in Celsius |
| `thickness_mm` | number | Thickness of material |
| `load_g` | number | Weight on surface (grams) |
| `mounting` | string | e.g., `clamped`, `free_edge`, `on_table` |
| `device` | string | Recording device: `pixel7`, `iphone14`, `usb_mic`, `ESP32+MPU6050`, etc. Used by `resdb benchmark --group-by device` |
| `session` | string | Recording session id (one session = one material on one device in one sitting). Used by `resdb benchmark --group-by session` |
| `object` | string | Which physical object this is, e.g. `kitchen_table`. Distinct from `session`: the same object recorded on two occasions shares an `object` but has different `session` values. Used by `resdb benchmark --label-by object` to test whether a signature is stable over time and across devices |
| `striker` | string | What struck the object: `finger`, `key`, `pen`, `coin`. A hard striker excites much higher frequencies than a fingertip, so an unrecorded striker becomes a hidden variable in every comparison |

## 📂 Example

```json
{
  "material": "tempered_glass",
  "thickness_mm": 5,
  "temperature_c": 22.1,
  "load_g": 0,
  "vibration": [0.001, -0.003, 0.002, ...],
  "sample_rate_hz": 1000,
  "excitation": "manual_tap",
  "source": "real",
  "device": "ESP32+MPU6050",
  "notes": "tapped with plastic pen"
}