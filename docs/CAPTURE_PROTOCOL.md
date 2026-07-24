# 🎙️ Capture Protocol (Phase 1 Pilot)

This protocol exists to answer the project's riskiest question: **do tap
signatures generalize across recording devices?** Reproducibility lives in
the protocol, not the code. Follow it closely so your data is comparable
with everyone else's.

## Goal

A pilot dataset of at least:

- **5 materials**
- **30 taps per material per device**
- **2 or more recording devices** (this is the most important requirement)

## Equipment

- Two or more recording devices. Any mix works: two phones, a phone and a
  laptop, a phone and a USB microphone. Note the exact model of each.
- A striker. Use the same one for a whole session. Good options: a wooden
  pencil (eraser end removed), a plastic pen cap, a teaspoon handle. Note
  which one you used.
- The objects to tap. For the pilot, aim for one clear object per material
  class, for example: a drinking glass, a wooden table, a metal pot, a
  plastic container, a ceramic mug.

## Recording settings

- Record WAV (uncompressed) if your recorder allows it. 44.1 kHz or 48 kHz.
  Phone formats (`.m4a`, `.mp4`, `.mp3`) are also accepted: `resdb ingest`
  decodes them automatically (needs `pip install "resonancedb[media]"`,
  included in the dev install). WAV is still preferred because lossy codecs
  smear the tap transient slightly.
- **Disable voice processing if you can**: automatic gain control, noise
  suppression, and echo cancellation all mangle the tap transient. Most
  "voice memo" apps apply them; a field-recorder app with a "raw" or
  "uncompressed" mode is better.
- **Hold the phone in the air**, in your free hand, microphone end (the
  bottom edge) pointed at the tap point, 10 to 30 cm away. Keep that
  distance roughly constant within a session.
- **Never rest the phone on the object you are tapping.** Vibration then
  travels through the phone's body into the microphone and you measure the
  contact rumble instead of the object's sound. Recordings made that way
  are a different measurement condition and don't belong in the same
  dataset as air-recorded ones.
- Pick a quiet room. Fridges, fans, and traffic all raise the noise floor.

## Procedure (one session = one material on one device)

1. Place the object the way it is normally used (glass standing on a table,
   pot on a counter). Note the mounting in the metadata.
2. Start recording. Wait 2 seconds of silence.
3. Tap the object 30 times, about 1 second apart, with consistent medium
   force. Let it ring fully between taps. Do not slide or scrape.
4. Wait 2 seconds, stop recording.
5. Save as `<material>_<device>_<session>.wav`, for example
   `glass_pixel7_kitchen01.wav`.

Repeat per material, then repeat the whole set on the second device.
Tapping the same objects with both devices in the same session is ideal:
it isolates the device as the only variable.

## From recording to dataset

One command per WAV file:

```bash
resdb ingest glass_pixel7_kitchen01.wav --material glass --device pixel7 --session kitchen01
```

This detects each tap, trims it, and writes one JSON sample per tap into
`data/glass/`. Then check and evaluate:

```bash
resdb validate --data data
resdb benchmark --data data --group-by device --save-dir models
```

`resdb benchmark` holds out each device in turn: it trains on every other
device and tests on the held-out one. That number, not a random-split
accuracy, is the honest answer to the Phase 1 question.

If tap detection misses taps (quiet recording) or finds too many (noisy
room), tune it:

- `--threshold-ratio 0.15` detects quieter taps (default 0.25)
- `--min-separation 0.5` if ring-downs are being split into two taps

## Metadata that matters

Recorded per sample automatically by `resdb ingest`: `device`, `session`,
`sample_rate_hz`, `excitation`, `source`. Add by editing the JSON or noting
for later: `temperature_c`, `thickness_mm`, `mounting`, striker type in
`notes`.

## What good data looks like

- Every tap clearly audible above the noise floor
- Consistent striker and distance within a session
- Honest labels: if you are not sure whether the bowl is ceramic or glass,
  say so in `notes` or leave it out
- Variation BETWEEN sessions is welcome: different objects of the same
  material, different rooms, different strikers. That variation is exactly
  what the models need to learn.

## Go/no-go

When the pilot set is complete, run:

```bash
resdb benchmark --data data --group-by device --save-dir models
```

Mean per-group accuracy meaningfully above chance (with 5 classes, chance
is 20%) means the core hypothesis holds and Phase 2 begins. A number near
chance is also a result: it tells us to pivot before building more.
