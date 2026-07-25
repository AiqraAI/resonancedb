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

## Phase 1: Validate the science (go/no-go gate)  DONE

The riskiest assumption in the whole project: *tap signatures are
reproducible across devices and sessions well enough to classify materials.*

- [x] Audio support in the package: WAV loading, tap onset detection, trim
      (`resonancedb/audio.py`, `resdb ingest`)
- [x] A written capture protocol: [docs/CAPTURE_PROTOCOL.md](docs/CAPTURE_PROTOCOL.md)
- [x] Pilot dataset: 411 taps, 14 objects, 4 materials, 1 device
- [x] `resdb benchmark`: leave-one-group-out evaluation, never random splits
- [x] Publish the honest number: [docs/FINDINGS.md](docs/FINDINGS.md)

**Result: a tap identifies the object, not the material.**

| Question | Accuracy | Chance |
|---|---|---|
| Which object is this tap from? | **93.0%** | 7.1% |
| Which material, on an unseen object? | 45.9% | 25% |
| Damped or ringing, on an unseen object? | 75.7% | 50% |

Geometry and mounting dominate the acoustic signature: glass objects in this
dataset span 202 to 7088 Hz, a wider range than the gap between materials.
Richer features were tested and did not help. Cross-object material ID needs
an order of magnitude more objects than one person can collect, which is an
argument for the collaborative dataset, not against it.

**Gate decision: the original premise is not supported at this scale, and the
93 percent object fingerprint points at a capability that works today.** The
plan below is reordered accordingly.

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

## Phase 4: Same-object change detection (promoted from the Phase 1 finding)

The 93 percent object fingerprint means a tap can tell one object from
another. The natural product question is therefore not "what is this made of"
but "has this object changed since last time", which holds geometry constant
and sidesteps the problem Phase 1 exposed.

- [ ] `resdb baseline` and `resdb compare`: record a reference signature for a
      specific object, then score how far a later recording deviates from it
- [ ] Establish the noise floor: how much does an unchanged object drift
      between sessions, days, temperatures, and phone positions
- [ ] Demonstrate a real detectable change (a cracked tile versus an intact
      one, a loose versus tightened fixing)
- [ ] Only then decide whether this is the product

## Phase 5: Model zoo + browser demo

- [ ] Train/tune on the real dataset; publish ONNX models to Hugging Face Hub
      with the cross-device benchmark attached
- [ ] Static demo page with onnxruntime-web: *tap your table, get a
      prediction*, no server, no API key
- [ ] Model cards documenting training config (the config embedded in each
      model package)

## Phase 6: Parked until demand exists

- Hosted API / web platform (separate repo: resonancedb-web), revive only
  when a concrete consumer needs server-side inference, and rebuild it on top
  of this package
- Hardware reference rig, redesigned around an I2S microphone or piezo + ADC
  (not the MPU-6050), as the calibrated instrument for serious contributors
