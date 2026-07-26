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

## How much would more data help? (measured)

Rather than guess, the learning curve was measured directly by subsampling the
existing objects. For each material, one object is held out and the model is
trained on everything else plus a varying number of *other* objects of that
same material. Every other variable is held constant, so the only thing
changing is how many examples of the material the model has seen.

Recall on a **new, unseen object** of that material:

| material | 1 other object | 2 | 3 | 4 | gain per object |
|---|---|---|---|---|---|
| wood | 33.9% | 75.6% | | | +41.7 pts |
| glass | 13.6% | 29.5% | 56.7% | | +21.6 pts |
| metal | 9.1% | 18.9% | 28.1% | 33.7% | +8.2 pts |

**Every material improves monotonically, and none has plateaued.** This is the
steep part of the curve, not the flat part. More objects per material clearly
help, and the current dataset is simply undersupplied.

The gain per object tracks how *similar* the objects within a material are:

- wood (two tables and a wardrobe, all large flat panels) learns fastest
- metal (a spoon, a pan, a tray) learns slowest, because a hand-held spoon and
  a pan share almost nothing acoustically

So the requirement is not just more objects but more objects *spanning the
variety* of the material. Adding five more spoons would teach the model
"spoon", not "metal".

A caveat on precision: with only 3 to 5 objects per material there are few
combinations to average over, so the absolute numbers are noisy and the slopes
should not be extrapolated far. The direction and consistency across three
independent materials is the reliable part.

**Revision to the conclusion above:** cross-object material identification is
undersupplied rather than unreachable, and the supply needed looks reachable.
Roughly 10 objects per material, about 40 objects in total, is a few recording
sessions rather than a research programme.

### That prediction was tested and it was wrong

The learning curve above predicted that adding objects would keep improving
recall. Two further collection rounds tested it directly and it did not hold.

**Balanced accuracy against the number of objects:**

| Objects | Taps | Balanced accuracy | Majority baseline |
|---|---|---|---|
| 14 | 411 | ~48.9% | ~36% |
| 26 | 767 | 43.4% | 50.0% |
| 32 | 945 | **40.9%** | 40.6% |

It is flat to declining, not rising, and at 32 objects the model is 0.3 points
above always guessing the commonest class.

Why the curve misled: it subsampled objects from a pool already in the dataset,
so "fewer objects" still meant objects of the same few kinds. Genuinely new
objects (a spatula, a light bulb, a chair leg) add within-class variance faster
than they add learnable signal. Subsampling an existing pool measures
something easier than real generalization, and it should not have been
extrapolated.

### Representations tested, all at or below the trivial baseline

| Representation | Balanced accuracy |
|---|---|
| Hand-crafted features (current) | 40.9% |
| Hilbert damping, per-band damping, spectral shape | 36.0% (at 26 objects) |
| **MFCCs** (the standard audio-ML representation), 80 dims | **29.6%** |
| Always guess the commonest class | 40.6% |
| Chance | 25.0% |

MFCCs largely collapsed to predicting metal (75.3% metal recall, 2.3%
concrete, 6.7% wood). Three qualitatively different representations, none of
which beats guessing.

### A confound that inflated an earlier round

At 26 objects, metal recall read 70.2% and eight metal objects scored ~100%.
That was substantially an artifact: 8 of 12 sessions using the high-impact
plastic striker were metal, so "plastic striker" predicted "metal" 67% of the
time. Non-metal plastic-struck objects were misclassified *as metal*
specifically (WoodenSlab 26/30, Wall4 29/30, GlassWindow 28/30).

Adding plastic-struck wood and glass reduced the shortcut to 44%, and metal
recall promptly fell from 70.2% to 42.6%. The earlier metal result was mostly
the striker, not the material.

## Verdict

Across 32 objects, 945 taps, three feature representations and three
collection rounds, **cross-object material classification does not work with
this approach**. It is not marginally short, it is at the level of guessing
the most common class. More data of the same kind has now been tried twice and
made the honest metric worse rather than better.

The object-identification result (93%) stands unchallenged and is the
capability worth building on.

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
