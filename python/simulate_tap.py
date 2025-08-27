import numpy as np
import os

def simulate_tap(frequency, damping, duration=2.0, sample_rate=1000):
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = np.exp(-damping * t) * np.sin(2 * np.pi * frequency * t)
    return signal

# Materials: frequency and damping based on real-world trends
materials = {
    "glass": {"frequency": 800, "damping": 0.4},
    "wood": {"frequency": 300, "damping": 1.5},
    "metal": {"frequency": 1000, "damping": 0.3},
    "plastic": {"frequency": 500, "damping": 1.0}
}

sample_rate = 1000
data_dir = "../data/simulated"
os.makedirs(data_dir, exist_ok=True)

for name, props in materials.items():
    signal = simulate_tap(props["frequency"], props["damping"], sample_rate=sample_rate)
    np.savez(
        f"{data_dir}/{name}.npz",
        vibration=signal,
        sample_rate_hz=sample_rate,
        material=name,
        source="simulation",
        damping=props["damping"],
        dominant_frequency=props["frequency"]
    )
    print(f"✅ Generated simulated/{name}.npz")