# 📄 ResonanceDB Data Format

All contributions must follow this JSON schema.

## 🔧 Required Fields

| Field | Type | Description |
|------|------|-------------|
| `material` | string | e.g., `glass`, `oak_wood`, `aluminum` |
| `vibration` | number[] | Acceleration values (g) over time |
| `sample_rate_hz` | number | Samples per second (e.g., 1000) |
| `excitation` | string | How vibration was created: `manual_tap`, `solenoid`, `ambient` |
| `source` | string | `real`, `simulation`, `phone_sensor` |

## 🌡️ Optional Fields

| Field | Type | Description |
|------|------|-------------|
| `temperature_c` | number | Temperature in Celsius |
| `thickness_mm` | number | Thickness of material |
| `load_g` | number | Weight on surface (grams) |
| `mounting` | string | e.g., `clamped`, `free_edge`, `on_table` |
| `device` | string | Sensor used: `ESP32+MPU6050`, `iPhone14`, etc. |

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