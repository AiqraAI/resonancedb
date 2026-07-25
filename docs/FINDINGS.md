# Phase 1 Findings

**Date:** 2026-07-25
**Dataset:** 411 taps, 14 objects, 4 materials, 1 device (phone01), microphone capture
**Method:** leave-one-object-out (each object held out in turn, model trained on
the others), RandomForest on the standard feature vector, 150 Hz high-pass

## Headline

**A tap identifies the object, not the material.**

| Question | Accuracy | Chance |
|---|---|---|
| Which of 14 objects is this tap from? | **93.0%** | 7.1% |
| Which of 4 materials is this? (new object) | 45.9% | 25% |
| Is this damped (wood, concrete) or ringing (glass, metal)? | 75.7% | 50% |

The acoustic signal is rich and highly repeatable: taps on the same object are
recognized as that object 93 percent of the time, and every one of the 14
objects scores above 71 percent. But that same signal generalizes poorly to a
new object of a known material.

## Why

A tap excites the object's vibrational modes. Those modes are set by geometry
(size, shape, thickness) and boundary conditions (how the object is held or
mounted) at least as much as by material. Measured peak frequencies within a
single material in this dataset:

- glass: 202, 220, 2276, 7088 Hz across four objects (35-fold spread)
- metal: 199, 626, 1294, 2021 Hz across five objects
- wood: 168, 200, 272 Hz across three objects
- concrete: 192, 196 Hz across two walls

Glass spans a wider frequency range than the gap between materials. A model
asked to learn "glass" from three examples, then generalize to a fourth glass
object of different size and shape, has very little to work with.

Object-level effects also cross material boundaries: a thin metal tray tapped
with a finger reads as damped (3.3 percent correct in the binary test) while a
hollow wooden table reads as ringing (23.3 percent). "Metal rings, wood thuds"
is not reliable at the object level.

## What was ruled out

**Feature engineering is not the bottleneck.** A richer set (Hilbert-envelope
damping in dB/s, damping measured separately in low and high bands, spectral
flatness, normalized octave-band energy ratios) scored 36.0 percent against
39.6 percent for the existing features. Tested and rejected.

**Recording quality matters, and is fixable.** Sessions that clipped scored 0
percent (Metalspoonkey, peak level 1.00). The same pan re-recorded without
clipping scored 100 percent. `resdb ingest` now warns about clipping and
`resdb summary` flags it per session.

**More objects help, but slowly.** Going from 12 to 14 objects moved 4-class
accuracy from 43.5 to 45.9 percent. Extrapolating, useful cross-object material
identification would need many tens of objects per material, well beyond what
one person collects in a week. This is an argument for the open collaborative
dataset, not against it.

## Implications

The original premise, "tap any object and learn what it is made of", is not
supported at 14 objects with this pipeline. It is not refuted either: the
trend with more objects is positive, and the failure mode is understood
(geometry dominates). It needs a dataset an order of magnitude larger.

Meanwhile the 93 percent object-identification result points at a capability
that works **now**, with data one person can collect: comparing an object
against its own earlier baseline. That is the structural-health-monitoring
framing, and it sidesteps the geometry problem entirely because geometry is
held constant. Detecting that a tap signature has changed is a well-posed
question with a clear application (cracks, delamination, loose fixings,
moisture).

See [ROADMAP.md](../ROADMAP.md) for how this reshapes the plan.

## Reproducing

```bash
resdb summary --data data
resdb benchmark --data data --group-by session --highpass-hz 150 --extra all
```
