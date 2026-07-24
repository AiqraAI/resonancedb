# ResonanceDB Roadmap

**The product is the dataset.** Every phase below serves one goal: a large,
open, well-labeled collection of vibration/acoustic tap signatures that anyone
can load with one line of code, plus the tools to capture, validate, and
learn from it. Infrastructure is kept at the minimum that goal requires
(target monthly hosting bill: $0).

Guiding decisions:

- **Microphone-first capture.** Every phone has a 44.1+ kHz mic; a tap's
  acoustic signature is far richer than a ~1 kHz accelerometer trace (glass
  and metal resonate in the kHz range). No app installs: capture runs in the
  browser.
- **GitHub is the backend.** Contributions arrive as pull requests validated
  by CI; identity, attribution, review, and moderation come for free.
- **Models run in the browser.** Tiny ONNX models via onnxruntime-web mean a
  public demo with zero servers.
- The hosted web platform (separate repo) is **parked** until real demand
  exists.

---

## Phase 0: Installable core package  ✅ (this release)

- [x] Restructure `python/` into a proper `resonancedb` package with
      `pyproject.toml`
- [x] `resdb` CLI as an entry point (simulate / validate / train / tune /
      evaluate / predict / inspect / export), no more `sys.path` hacks
- [x] Validation returns real exit codes so CI can gate on it; schema rules
      live in `resonancedb.schema` as the single source of truth
- [x] `resdb simulate` emits the standard JSON format (and simulation
      defaults no longer alias, 4 kHz sample rate)
- [x] Remove server remnants (Dockerfile, docker-compose), committed model
      binaries, generated artifacts, and stale duplicate entry points
- [x] Park the ESP32 firmware with an honest status note

**Acceptance:** fresh clone → `pip install -e .[dev] && pytest && resdb --help` works.

## Phase 1: Validate the science (go/no-go gate)

The riskiest assumption in the whole project: *tap signatures are
reproducible across devices and sessions well enough to classify materials.*

- [x] Audio support in the package: WAV loading, tap onset detection, trim
      (`resonancedb/audio.py`, `resdb ingest`)
- [x] A written capture protocol (materials, tap count, mounting, distance,
      device notes): [docs/CAPTURE_PROTOCOL.md](docs/CAPTURE_PROTOCOL.md)
- [ ] Pilot dataset: ≥5 materials × ≥30 taps × ≥2 recording devices
- [x] `resdb benchmark`: leave-one-group-out evaluation by **device/session/file**,
      never random splits
- [ ] Publish the honest number

**Gate:** cross-device accuracy meaningfully above chance → continue.
Near chance → pivot toward fixed-sensor structural monitoring.

## Phase 2: Zero-install capture page

- [ ] Static web page (`/capture`): Web Audio recording with AGC/noise
      suppression disabled → auto-trim around the transient → waveform
      preview → metadata form → download a ready-to-submit JSON/WAV
- [ ] Hosted free on GitHub Pages / Cloudflare Pages (HTTPS required for mic
      access)
- [ ] Phyphox converter stays available as a legacy path for
      contact-vibration data

## Phase 3: Community pipeline (GitHub as backend)

- [ ] Contribution = PR adding files under `data/<material>/`
- [ ] CI on data PRs: schema check, signal-quality check, duplicate detection
- [ ] Nightly job regenerates dataset stats and a leaderboard computed from
      git history
- [ ] Publish the curated dataset to Hugging Face Datasets (versioned, DOI,
      `load_dataset("aiqra/resonancedb")`)
- [ ] Finalize the data license (leaning CC-BY for maximum adoption)

## Phase 4: Model zoo + browser demo

- [ ] Train/tune on the real dataset; publish ONNX models to Hugging Face Hub
      with the cross-device benchmark attached
- [ ] Static demo page with onnxruntime-web: *tap your table, get a
      prediction*, no server, no API key
- [ ] Model cards documenting training config (the config embedded in each
      model package)

## Phase 5: Parked until demand exists

- Hosted API / web platform (separate repo: resonancedb-web), revive only
  when a concrete consumer needs server-side inference, and rebuild it on top
  of this package
- Hardware reference rig, redesigned around an I2S microphone or piezo + ADC
  (not the MPU-6050), as the calibrated instrument for serious contributors
