# 🌐 ResonanceDB

**The Open Database of How Everything Vibrates**
*By [AIQRA AI](https://aiqra.ai)*

> 🧠 _"Teaching machines to feel the world, one tap at a time."_

When you tap an object — glass, wood, metal, plastic — it vibrates and rings
in a way that is unique to its material, geometry, and condition. ResonanceDB
is an open-source project building a **public dataset of these tap
signatures**, plus the tools to capture, validate, and learn from them, so
that AI, robots, and IoT devices can identify materials and detect structural
flaws without touching them.

**The product is the dataset.** The code in this repo exists to grow that
dataset and to prove what can be learned from it. The full plan — including
what is deliberately parked — lives in [ROADMAP.md](ROADMAP.md).

---

## 📦 What's in this repo

| Path | What it is |
|------|------------|
| `resonancedb/` | The Python package: feature extraction, preprocessing, schema validation, simulation, training — and the `resdb` CLI |
| `data/` | The dataset — grows via pull requests, one folder per material |
| `docs/` | Data format spec and guides |
| `tests/` | Test suite |
| `scripts/` | Legacy phone-accelerometer (phyphox) converter |
| `firmware/` | ESP32 prototype — **parked**, see [firmware/README.md](firmware/README.md) |

---

## 🚀 Quick start

```bash
git clone https://github.com/AiqraAI/resonancedb.git
cd resonancedb
pip install -e ".[dev]"
```

Generate synthetic taps, train a model, and classify a sample — end to end in
under a minute:

```bash
resdb simulate --out-dir data/simulated
resdb validate --data data
resdb train --data data --extra all --target-length 1024
resdb predict data/simulated/glass.json
```

Other commands: `resdb tune` (k-fold hyperparameter search), `resdb evaluate`
(accuracy report + confusion matrix), `resdb inspect` (show a saved model's
embedded configuration), `resdb export` (ONNX for browser/edge inference —
needs `pip install "resonancedb[export]"`).

Use it as a library:

```python
from resonancedb import compute_feature_vector

vec = compute_feature_vector(signal, sample_rate_hz=48000, extra=True)
# [peak_freq, decay_rate, energy, spectral_centroid, spectral_bandwidth,
#  zcr, top-3 peak frequencies, autocorrelation lag]
```

---

## 🎙️ How data gets captured

**Microphone-first.** Every phone has a 44.1+ kHz microphone, and a tap's
acoustic ring is far richer than what low-rate accelerometers can capture
(glass and metal resonate in the kHz range — beyond what a ~1 kHz
accelerometer can even see). A **zero-install, browser-based capture page**
is the next milestone: open a link, allow the mic, tap the object, submit.

Until it ships, record with any audio recorder or (legacy path) a phone
accelerometer app like phyphox, and format samples per
[docs/DATA_FORMAT.md](docs/DATA_FORMAT.md):

```json
{
  "material": "oak_wood",
  "vibration": [0.001, -0.004, ...],
  "sample_rate_hz": 48000,
  "excitation": "manual_tap",
  "source": "phone_microphone",
  "temperature_c": 21.5,
  "thickness_mm": 18
}
```

➡️ **Add your sample under `data/<material>/` and open a pull request.**

---

## 📊 Why this matters

| Use Case | Impact |
|--------|--------|
| 🤖 Robotics | Robots that identify surfaces before grasping |
| 🏗️ Structural Health | Detect cracks in bridges, drones, or furniture |
| 🌐 IoT Devices | Smart tables, floors, or walls that "feel" activity |
| 🔬 Materials Science | Open data for acoustic properties of everyday materials |
| 🛠️ DIY & Education | Low-cost NDT for makers and students |

---

## 🗺️ Where this is going

1. **Phase 1** — pilot dataset + an honest **cross-device** benchmark (the
   go/no-go question: do tap signatures generalize between recording devices?)
2. **Phase 2** — zero-install browser capture page
3. **Phase 3** — PR-based community pipeline; dataset published to Hugging
   Face (`load_dataset("aiqra/resonancedb")`)
4. **Phase 4** — model zoo + in-browser demo: tap your table, get a
   prediction, no server involved

Details and acceptance criteria: [ROADMAP.md](ROADMAP.md).

---

## 🤝 Contributing

- **Data** — new materials, varied conditions (temperature, load, humidity),
  multiple recording devices. See [CONTRIBUTING.md](CONTRIBUTING.md).
- **Code** — signal processing, the capture page, evaluation tooling.
- **Science** — help design the Phase 1 cross-device benchmark protocol:
  open an issue.
- **Spread the word** — star this repo 🌟, teach it in your lab or classroom.

---

## 📄 License

Code is MIT — use freely, even commercially. We only ask: **contribute back
what you learn.** The dataset carries its own license, finalized before
Phase 3 (leaning CC-BY — see the roadmap).

---

## 🚀 Powered by AIQRA AI

This project is led by **[AIQRA AI](https://aiqra.ai)** — advancing open
physical intelligence. Want to collaborate? Contact: info@aiqra.ai

_The first tap is free. The knowledge is forever._
