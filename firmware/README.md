# Firmware (parked)

**Status: parked until Phase 5 of the [roadmap](../ROADMAP.md).**

The `esp32-tap-recorder` sketch in this folder is an early prototype and is
**not currently functional**: it mixes two incompatible MPU-6050 libraries,
does not control its sampling rate, and the MPU-6050's ~1 kHz output limits
it to signals below ~500 Hz — well under the kHz-range resonances of glass
and metal.

The current capture path is microphone-based (44.1+ kHz, no hardware or app
install required). A hardware reference rig will return here later, designed
around an I2S microphone or a piezo + ADC front end rather than the MPU-6050,
to serve as the calibrated instrument for serious contributors.
