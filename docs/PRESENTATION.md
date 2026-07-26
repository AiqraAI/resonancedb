# ResonanceDB: Slide Deck Source

**Note for the slide generator:** this document is written as a slide-by-slide
script. Each `## Slide N` is one slide. "Headline" is the slide title, "On the
slide" is the visible content (keep it short), and "Speaker notes" is the
narration underneath. Numbers are real measurements, please keep them exact.

Suggested length: 17 slides, roughly a 10 minute talk.

---

## Slide 1: Title

**Headline:** ResonanceDB

**On the slide:**
- The open database of how everything vibrates
- Teaching machines to identify the physical world by tapping it
- By AIQRA AI
- github.com/AiqraAI/resonancedb

**Speaker notes:** ResonanceDB is an open-source project building a public
dataset of tap sounds, plus the tools to capture and learn from them. This deck
covers what it does, what we measured, and what we found, including one result
that changed the direction of the project.

---

## Slide 2: The idea in one sentence

**Headline:** Every object has a voice

**On the slide:**
- Tap a glass and it rings
- Tap a table and it thuds
- That sound carries information about what you hit
- Question: can a machine learn to read it?

**Speaker notes:** This is not an exotic idea. People already do it. Engineers
tap concrete to find cracks, market traders tap watermelons for ripeness,
mechanics tap parts to check for damage. All of it is done by ear, by
experienced people, and none of it is written down as data. We wanted to know
whether a phone could do the same thing.

---

## Slide 3: Why it matters

**Headline:** What this unlocks

**On the slide:**
- Robotics: know a surface before grasping it
- Structural health: find cracks and delamination in concrete
- Industry: check parts without cutting them open
- Field work: low-cost testing where lab equipment is unaffordable
- All of it non-destructive, and using hardware people already own

**Speaker notes:** The existing tools for this are expensive and need trained
operators. Impact echo equipment costs thousands. If a phone can do a useful
fraction of that, the economics change completely, especially in markets where
the specialist gear will never arrive.

---

## Slide 4: How a tap carries information

**Headline:** The physics, simply

**On the slide:**
- A tap makes an object vibrate at its natural frequencies
- Those frequencies depend on size, shape, thickness, material and mounting
- How fast the sound dies away depends on damping
- Together they form an acoustic signature

**Speaker notes:** When you strike something, you are giving it a tiny push
across all frequencies at once. The object responds by ringing at the
frequencies it naturally prefers. Hard, stiff, low-damping things like glass
and metal ring for a long time. Soft or heavy-damped things like wood and
concrete stop almost immediately. That difference is what we are trying to
read.

---

## Slide 5: How the data is collected

**Headline:** A phone, and nothing else

**On the slide:**
- Record ~30 taps on one object with any phone
- Microphone captures at 44,100 samples per second
- No app to install, no hardware to buy
- Deliberately chosen over accelerometers

**Speaker notes:** We started with an accelerometer plan and abandoned it. Phone
browsers cap motion sensors at about 60 readings per second, and glass and
metal resonate in the thousands. The microphone records at 44,100. The better
sensor turned out to be the one that needs no hardware at all.

---

## Slide 6: The pipeline

**Headline:** From a recording to a dataset

**On the slide:**
1. **Record** one object, about 30 taps
2. **Detect** each individual tap automatically
3. **Trim** half a second around each strike
4. **Extract** features: peak frequency, decay, energy, spectral shape
5. **Store** one readable JSON file per tap, with full metadata

**Speaker notes:** One command turns a recording into dataset entries. Each tap
becomes its own file carrying what it is, which device recorded it, which
session, and what it was struck with. Human readable, so anyone can inspect a
sample without special software.

---

## Slide 7: What we built

**Headline:** The toolkit

**On the slide:**
- `resdb ingest` recording to dataset samples
- `resdb summary` per-session quality report
- `resdb benchmark` honest evaluation
- `resdb train` / `tune` / `export` models, including ONNX for browsers
- Installable Python package, 30 automated tests, open source

**Speaker notes:** All of it open, all of it runnable on a laptop, with no
server and no hosting costs. The design rule was that anyone should be able to
clone the repo and reproduce every number in this deck.

---

## Slide 8: Measuring honestly

**Headline:** The evaluation had to be strict

**On the slide:**
- Wrong way: mix taps from one object into training and testing
- The model then recognises the recording, not the material
- Our way: hold out an entire object, train on the rest, test on it
- The only question that matters: does it work on something never seen before

**Speaker notes:** This matters more than any other design choice. If you
randomly split taps, the same table appears on both sides and you get a
flattering number that means nothing. We deliberately built the tool so it
refuses to do random splits. The number it reports is the hard one.

---

## Slide 9: The dataset today

**Headline:** What we have measured so far

**On the slide:**
- 411 taps
- 14 different objects
- 4 materials: wood, concrete, glass, metal
- 1 phone
- Every sample real, recorded by hand, nothing synthetic

**Speaker notes:** Small, but every entry is a genuine measurement with honest
provenance. We deliberately removed synthetic and fabricated samples from the
dataset early on, because in an open measurement corpus provenance is the whole
value.

---

## Slide 10: The headline result

**Headline:** A tap identifies the object, not the material

**On the slide:**

| Question | Accuracy | Chance |
|---|---|---|
| Which of the 14 objects is this? | **93.0%** | 7.1% |
| Which material, on an unseen object? | 45.9% | 25% |
| Damped or ringing, on an unseen object? | 75.7% | 50% |

**Speaker notes:** This was not what we set out to find. Every single object is
recognised well above chance, several of them perfectly. The acoustic signal is
rich and highly repeatable. It just encodes which specific thing you hit far
more strongly than what that thing is made of.

---

## Slide 11: Why material is hard

**Headline:** Geometry drowns out material

**On the slide:**
- Four glass objects in our data ring at 202, 220, 2276 and 7088 Hz
- That is a 35-fold spread inside a single material
- Wider than the gap between different materials
- A metal tray reads as dull; a hollow wooden table reads as ringing

**Speaker notes:** A bottle, a jar and a drinking glass are all glass, and they
sound nothing alike, because size and shape set the pitch. Asking a model to
learn "glass" from three examples and then recognise a fourth of a different
shape is a genuinely hard problem. This is the central finding, and it is a
property of physics, not a bug in the code.

---

## Slide 12: What we ruled out

**Headline:** Testing the obvious explanations

**On the slide:**
- **Better features?** Tested advanced acoustic measures. Result: 36.0% versus
  39.6%. Slightly worse. Not the bottleneck.
- **Recording quality?** Real but fixable. A clipped recording scored 0%. The
  same pan recorded properly scored 100%.
- Tooling now warns about both problems automatically

**Speaker notes:** Before concluding anything we checked whether we were simply
doing it badly. Richer signal processing did not help, which saved us weeks of
chasing it. Recording quality genuinely mattered, so the tools now catch those
mistakes while the person is still standing next to the object.

---

## Slide 13: Would more data help? We measured it

**Headline:** The learning curve is still climbing

**On the slide:**

| Material | 1 example | 2 | 3 | 4 | gain per object |
|---|---|---|---|---|---|
| Wood | 33.9% | **75.6%** | | | +41.7 pts |
| Glass | 13.6% | 29.5% | **56.7%** | | +21.6 pts |
| Metal | 9.1% | 18.9% | 28.1% | **33.7%** | +8.2 pts |

- Every material improves. None has plateaued.

**Speaker notes:** Rather than argue about whether more data would help, we
measured it by simulating a smaller dataset. Recall on a completely new object
rises steadily with every extra example. We are on the steep part of the curve.
The conclusion is that material identification is undersupplied, not
impossible.

---

## Slide 14: The five-year-old version

**Headline:** Why more examples

**On the slide:**
- Imagine you had only ever seen one dog, a tiny Chihuahua
- Then you meet a Great Dane
- You would say "that is not a dog"
- Our model has met three metal things: a spoon, a pan, a tray
- It needs to meet more metal

**Speaker notes:** This also explains why metal is our hardest class. A spoon
and a pan are the Chihuahua and the Great Dane of metal. Wood is easiest
because our wooden objects are all large flat panels that resemble each other.
Variety within a material matters as much as raw count.

---

## Slide 15: Two directions, both open

**Headline:** Where this goes

**On the slide:**
- **Keep building the dataset.** About 10 objects per material, roughly 40
  total, is a few recording sessions rather than a research programme
- **Use the fingerprint.** 93% object recognition means you can ask "has this
  object changed since last time", which holds geometry constant
- The second is the basis of crack and damage detection

**Speaker notes:** The failure to classify materials is simultaneously a strong
result for something else. If a tap reliably identifies one specific object,
you can compare an object against its own past. That is structural health
monitoring, and it sidesteps the geometry problem entirely because the geometry
never changes.

---

## Slide 16: Roadmap

**Headline:** What happens next

**On the slide:**
1. **Now:** grow the dataset, more objects spanning more variety
2. **Next:** a zero-install web page so anyone can contribute by tapping
3. **Then:** contributions arrive as pull requests, dataset published openly
4. **In parallel:** same-object change detection
5. **Open question:** does the fingerprint survive across days and devices

**Speaker notes:** That last point is the honest gap. Our 93 percent was
measured within a single recording session on one phone. Whether the signature
holds a week later on a different phone is untested, and it is the next
experiment, because most of the interesting applications depend on it.

---

## Slide 17: How to get involved

**Headline:** Contribute

**On the slide:**
- Tap things and send recordings, no equipment needed
- Improve the signal processing or build the capture page
- Help design the cross-device experiment
- Everything open source, MIT licensed
- github.com/AiqraAI/resonancedb

**Speaker notes:** The reason this is an open project rather than a private one
is straightforward. Solving material identification needs more objects than any
one person can tap, and the objects need to be genuinely varied. That is
exactly the kind of dataset a distributed group can build and a single lab
cannot.

---

## Appendix: numbers reference

For anyone checking the figures.

- Dataset: 411 taps, 14 objects, 4 materials, 1 device
- Evaluation: leave-one-object-out, RandomForest, 150 Hz high-pass filter
- Object identification: 93.0 percent, chance 7.1 percent
- Material identification on unseen object: 45.9 percent, chance 25 percent
- Damped versus ringing: 75.7 percent, chance 50 percent
- Wood versus concrete alone: 99.3 percent across 5 objects, chance 50 percent
- Alternative feature set: 36.0 percent versus 39.6 percent baseline
- Clipped session: 0 percent. Same object recorded cleanly: 100 percent
- Learning curve gains per added object: wood +41.7, glass +21.6, metal +8.2

Full detail in `docs/FINDINGS.md`, reproducible with
`resdb benchmark --data data --group-by session --highpass-hz 150 --extra all`.
