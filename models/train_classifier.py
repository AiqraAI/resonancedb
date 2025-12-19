# models/train_classifier.py
import numpy as np
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from python.features import compute_feature_vector
import joblib

def load_data(
    data_dir,
    *,
    extra=False,
    top_k_peaks: int = 3,
    detrend: bool | None = None,
    window: str | None = None,
    target_length: int | None = None,
    resample_rate_hz: float | None = None,
):
    features = []
    labels = []
    data_path = Path(data_dir)

    print(f"🔍 Looking for data in: {data_path.absolute()}")

    # Check if folder exists
    if not data_path.exists():
        print(f"❌ Folder does not exist: {data_path}")
        return np.array([]), np.array([])

    # Find all .json files
    json_files = list(data_path.rglob("*.json"))
    print(f"📁 Found {len(json_files)} JSON file(s)")

    if len(json_files) == 0:
        print("💡 Try creating test files in data/ (e.g., test_glass.json)")
        return np.array([]), np.array([])

    for file_path in json_files:
        print(f"\n📄 Processing: {file_path.name}")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Check required keys
            required = ['material', 'vibration', 'sample_rate_hz']
            missing = [k for k in required if k not in data]
            if missing:
                print(f"  ❌ Missing keys: {missing}")
                continue

            material = data['material']
            vibration = data['vibration']
            sample_rate = data['sample_rate_hz']

            if not isinstance(vibration, (list, tuple)) or len(vibration) == 0:
                print("  ❌ 'vibration' is empty or not a list")
                continue

            if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
                print("  ❌ 'sample_rate_hz' must be a positive number")
                continue

            if not isinstance(material, str) or not material.strip():
                print("  ❌ 'material' is not a valid string")
                continue

            # Convert to numpy array
            signal = np.array(vibration)

            # Unified feature extraction with preprocess and optional extras
            # Default behavior: if detrend/window are None, use standard defaults (True/'hann')
            detrend_flag = True if detrend is None else bool(detrend)
            window_name = "hann" if (window is None or window == "hann") else None
            vec = compute_feature_vector(
                signal,
                sample_rate,
                detrend=detrend_flag,
                window=window_name,
                target_length=target_length,
                resample_rate_hz=resample_rate_hz,
                extra=extra,
                top_k_peaks=top_k_peaks,
            )
            peak_freq, decay_rate, energy = vec[:3].tolist()
            # Use full vector to keep model features consistent when extras are enabled
            features.append(vec.tolist())
            labels.append(material)

            print(f"  ✅ Extracted: {material} | {peak_freq:.1f} Hz | decay={decay_rate:.3f} | energy={energy:.3f}")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue

    if len(features) == 0:
        print("\n❌ No valid data loaded. Cannot train model.")
        return np.array([]), np.array([])

    print(f"\n✅ Loaded {len(features)} sample(s): {set(labels)}")
    return np.array(features), np.array(labels)


# --- MAIN ---
if __name__ == "__main__":
    # Allow training via CLI or direct execution
    def _train_default():
        X, y = load_data("data")  # Looks inside the current folder

        if len(X) == 0:
            print("🛑 Training aborted: no data to learn from.")
            exit(1)

        # Split and train
        try:
            # Only stratify if we have enough samples per class
            stratify = None
            if len(set(y)) > 1:  # Multiple classes exist
                class_counts = np.bincount([list(set(y)).index(label) for label in y])
                if min(class_counts) >= 2:  # At least 2 samples per class
                    stratify = y

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=stratify
            )
        except ValueError as e:
            print(f"❌ Split failed: {e}")
            print("💡 Try adding more samples (at least 2 different materials)")
            exit(1)

        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n✅ Model trained!")
        print(f"📊 Test Accuracy: {acc * 100:.1f}%")
        print(f"🎯 Classes: {sorted(set(y))}")

        # Save the trained model
        joblib.dump(clf, 'models/material_model.pkl')
        print("✅ Model saved to models/material_model.pkl")

    _train_default()