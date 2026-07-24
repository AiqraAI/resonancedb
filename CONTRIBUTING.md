# 🤝 How to Contribute

We welcome contributions in:
- Data
- Code
- Models
- Docs

## 🧪 Submit Data

Data contributions are pull requests that add JSON files under
`data/<material>/` (e.g. `data/oak_wood/phone_pixel7_20260724_01.json`).

1. Record a tap, any audio recorder works today; a zero-install browser
   capture page is coming (see [ROADMAP.md](ROADMAP.md) Phase 2).
2. Format it per [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md).
3. Check it locally: `resdb validate --data data`
4. Open a PR, CI runs the same validation.

Good contributions vary the conditions: different objects of the same
material, different temperatures, thicknesses, and mounting, and, most
valuable of all, **different recording devices**, because cross-device
generalization is the project's central open question.

Simulated data (`resdb simulate`) is for testing pipelines, not for the
dataset, please don't submit it.

## 🛠️ Improve Code

```bash
pip install -e ".[dev]"
pytest
```

Areas where help is most useful:
- Signal processing and feature extraction (`resonancedb/features.py`,
  `resonancedb/preprocess.py`)
- The browser capture page (ROADMAP Phase 2, Web Audio, static site)
- Device/session-aware benchmarking (ROADMAP Phase 1)

Please add a test for any behavior change.

## 🤖 Train Models

- Train and tune with `resdb train` / `resdb tune`; saved models embed their
  feature configuration automatically.
- Export portable models with `resdb export` (ONNX).
- A public model zoo lands with ROADMAP Phase 4.

## 📣 Spread the Word

- Star the repo 🌟
- Share on social media
- Teach it in your lab or classroom
