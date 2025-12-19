import argparse
import sys
from pathlib import Path
import numpy as np
import json
import joblib

# Ensure project root is on the import path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Avoid importing simulate_tap at module load to prevent side effects
from models.train_classifier import load_data
from python.features import compute_feature_vector
from scripts.validate_data import validate_sample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


def cmd_simulate(args: argparse.Namespace) -> None:
    from python.simulate_tap import simulate_tap, materials as SIM_MATERIALS
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = args.sample_rate
    duration = args.duration

    mats = SIM_MATERIALS
    if args.material is not None:
        sel = args.material.lower()
        if sel not in mats:
            print(f"❌ Unknown material '{args.material}'. Available: {list(mats.keys())}")
            return
        mats = {sel: mats[sel]}

    for name, props in mats.items():
        signal = simulate_tap(props["frequency"], props["damping"], duration=duration, sample_rate=sample_rate)
        file_path = out_dir / f"{name}.npz"
        np.savez(
            file_path,
            vibration=signal,
            sample_rate_hz=sample_rate,
            material=name,
            source="simulation",
            damping=props["damping"],
            dominant_frequency=props["frequency"],
        )
        print(f"✅ Generated {file_path}")


def _parse_extra_flag(extra_flag: str | None):
    if extra_flag is None:
        return False
    flag = extra_flag.strip().lower()
    if flag in ("", "none", "false"):
        return False
    if flag in ("all", "true"):
        return True
    # comma-separated list
    names = [s.strip() for s in flag.split(",") if s.strip()]
    return names if names else False


def _parse_list_flag(flag: str | None, cast=float):
    if flag is None:
        return None
    raw = flag.strip()
    if not raw:
        return None
    items = []
    for part in raw.split(","):
        p = part.strip().lower()
        if p in ("none", "null"):
            items.append(None)
        else:
            try:
                items.append(cast(part))
            except Exception:
                items.append(part)
    return items


def train_model(
    data_dir: Path,
    out_model: Path,
    test_size: float = 0.3,
    random_state: int = 42,
    extra_flag: str | None = None,
    top_k_peaks: int = 3,
    detrend: bool | None = None,
    window: bool | None = None,
    target_length: int | None = None,
    resample_rate_hz: float | None = None,
) -> bool:
    extra = _parse_extra_flag(extra_flag)
    # Map window toggle to window name
    window_name = "hann" if (window is True or window is None) else None
    detrend_flag = True if (detrend is None) else bool(detrend)

    X, y = load_data(
        str(data_dir),
        extra=extra,
        top_k_peaks=top_k_peaks,
        detrend=detrend_flag,
        window=window_name,
        target_length=target_length,
        resample_rate_hz=resample_rate_hz,
    )
    if len(X) == 0:
        print("🛑 Training aborted: no data to learn from.")
        return False

    try:
        stratify = None
        if len(set(y)) > 1:
            # Safe stratification only if enough samples per class
            class_counts = {}
            for label in y:
                class_counts[label] = class_counts.get(label, 0) + 1
            if min(class_counts.values()) >= 2:
                stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
    except ValueError as e:
        print(f"❌ Split failed: {e}")
        print("💡 Try adding more samples (at least 2 different materials)")
        return False

    clf = RandomForestClassifier(n_estimators=10, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model trained!")
    print(f"📊 Test Accuracy: {acc * 100:.1f}%")
    print(f"🎯 Classes: {sorted(set(y))}")

    # Persist model with feature configuration metadata for consistent prediction
    package = {
        "model": clf,
        "config": {
            "preprocess": {
                "detrend": detrend_flag,
                "window": window_name is not None,
                "target_length": target_length,
                "resample_rate_hz": resample_rate_hz,
            },
            "extra": extra,
            "top_k_peaks": top_k_peaks,
            "input_dim": int(X_train.shape[1]) if hasattr(X_train, "shape") else int(len(X[0])),
        },
    }
    joblib.dump(package, str(out_model))
    print(f"✅ Model saved to {out_model}")
    return True


def cmd_train(args: argparse.Namespace) -> None:
    data_dir = Path(args.data)
    out_model = Path(args.out)
    train_model(
        data_dir,
        out_model,
        test_size=args.test_size,
        random_state=args.random_state,
        extra_flag=args.extra,
        top_k_peaks=args.top_k_peaks,
        detrend=args.detrend,
        window=args.window,
        target_length=args.target_length,
        resample_rate_hz=args.resample_rate_hz,
    )


def cmd_predict(args: argparse.Namespace) -> None:
    # Load model (and config if present)
    model_path = Path(args.model) if args.model else (ROOT / "models" / "material_model.pkl")
    loaded = joblib.load(str(model_path))
    if isinstance(loaded, dict) and "model" in loaded:
        model = loaded["model"]
        saved_cfg = loaded.get("config", {})
    else:
        model = loaded
        saved_cfg = {}

    # Load data
    with open(args.file, 'r') as f:
        data = json.load(f)

    signal = np.array(data['vibration'])
    sample_rate = data['sample_rate_hz']

    # Determine effective configuration (CLI overrides saved when provided)
    saved_pre = saved_cfg.get("preprocess", {})
    extra = _parse_extra_flag(args.extra) if args.extra is not None else saved_cfg.get("extra", False)
    top_k = args.top_k_peaks if args.top_k_peaks is not None else int(saved_cfg.get("top_k_peaks", 3))
    # Booleans default to None to allow fallback to saved config
    detrend = args.detrend if args.detrend is not None else bool(saved_pre.get("detrend", True))
    window_flag = args.window if args.window is not None else bool(saved_pre.get("window", True))
    window_name = "hann" if window_flag else None
    target_length = args.target_length if args.target_length is not None else saved_pre.get("target_length", None)
    resample_rate_hz = args.resample_rate_hz if args.resample_rate_hz is not None else saved_pre.get("resample_rate_hz", None)

    # Compute features with effective extras and preprocess
    vec = compute_feature_vector(
        signal, sample_rate,
        detrend=detrend,
        window=window_name,
        target_length=target_length,
        resample_rate_hz=resample_rate_hz,
        extra=extra,
        top_k_peaks=top_k,
    )

    # Match model input dimensionality
    n_in = getattr(model, 'n_features_in_', len(vec))
    if len(vec) != n_in:
        # Attempt to use saved configuration for consistency
        print(f"⚠️ Feature length {len(vec)} != model expects {n_in}. Using saved model configuration.")
        if saved_cfg:
            pre2 = saved_cfg.get("preprocess", {})
            extra2 = saved_cfg.get("extra", False)
            topk2 = int(saved_cfg.get("top_k_peaks", 3))
            window2 = "hann" if pre2.get("window", True) else None
            vec = compute_feature_vector(
                signal, sample_rate,
                detrend=bool(pre2.get("detrend", True)),
                window=window2,
                target_length=pre2.get("target_length", None),
                resample_rate_hz=pre2.get("resample_rate_hz", None),
                extra=extra2,
                top_k_peaks=topk2,
            )
        else:
            # Fallback to base features if no saved config
            vec = compute_feature_vector(signal, sample_rate, extra=False)

    features = np.array([vec])
    prediction = model.predict(features)[0]
    print(f"🔮 This is {prediction.upper()}!")
    
    # Optionally print extras if requested
    if extra:
        print(f"📊 Used {len(vec)} features (including extras)")
    return prediction


def cmd_inspect(args: argparse.Namespace) -> None:
    model_path = Path(args.model) if args.model else (ROOT / "models" / "material_model.pkl")
    loaded = joblib.load(str(model_path))
    if isinstance(loaded, dict) and "model" in loaded:
        cfg = loaded.get("config", {})
        print("📦 Model package detected")
        print(f"📁 Path: {model_path}")
        print("🧩 Input dim:", cfg.get("input_dim", getattr(loaded["model"], "n_features_in_", "?")))
        print("🧪 Extras:", cfg.get("extra", False))
        print("🔢 Top-K peaks:", cfg.get("top_k_peaks", 3))
        pre = cfg.get("preprocess", {})
        print("🧰 Preprocess:")
        print(f"   • detrend: {pre.get('detrend', True)}")
        print(f"   • window: {pre.get('window', True)}")
        print(f"   • target_length: {pre.get('target_length', None)}")
        print(f"   • resample_rate_hz: {pre.get('resample_rate_hz', None)}")
        if "best_params" in cfg:
            print("🏆 Best params:", cfg.get("best_params"))
        if "best_cv_accuracy" in cfg:
            bacc = float(cfg.get("best_cv_accuracy"))
            folds = cfg.get("cv_folds", "?")
            print(f"📊 Best CV accuracy: {bacc * 100:.1f}% ({folds}-fold)")
    else:
        print("📦 Raw sklearn model (no embedded config)")
        print(f"📁 Path: {model_path}")
        print("🧩 Input dim:", getattr(loaded, "n_features_in_", "?"))


def cmd_evaluate(args: argparse.Namespace) -> None:
    data_dir = Path(args.data)
    save_dir = Path(args.save_dir) if args.save_dir else (ROOT / "models")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load model and config
    model_path = Path(args.model) if args.model else (ROOT / "models" / "material_model.pkl")
    loaded = joblib.load(str(model_path))
    if isinstance(loaded, dict) and "model" in loaded:
        model = loaded["model"]
        saved_cfg = loaded.get("config", {})
    else:
        model = loaded
        saved_cfg = {}

    # Effective configuration: default to saved, allow CLI override
    saved_pre = saved_cfg.get("preprocess", {})
    extra = _parse_extra_flag(args.extra) if args.extra is not None else saved_cfg.get("extra", False)
    top_k = args.top_k_peaks if args.top_k_peaks is not None else int(saved_cfg.get("top_k_peaks", 3))
    detrend = args.detrend if args.detrend is not None else bool(saved_pre.get("detrend", True))
    window_flag = args.window if args.window is not None else bool(saved_pre.get("window", True))
    window_name = "hann" if window_flag else None
    target_length = args.target_length if args.target_length is not None else saved_pre.get("target_length", None)
    resample_rate_hz = args.resample_rate_hz if args.resample_rate_hz is not None else saved_pre.get("resample_rate_hz", None)

    # Load dataset features with effective config
    X, y = load_data(
        str(data_dir),
        extra=extra,
        top_k_peaks=top_k,
        detrend=detrend,
        window=window_name,
        target_length=target_length,
        resample_rate_hz=resample_rate_hz,
    )
    if len(X) == 0:
        print("🛑 Evaluation aborted: no data to evaluate.")
        return

    # Predict and compute metrics
    y_pred = model.predict(X)
    acc = float(accuracy_score(y, y_pred))
    report = classification_report(y, y_pred, output_dict=True)
    labels_sorted = sorted(list(set(y)))
    cm = confusion_matrix(y, y_pred, labels=labels_sorted).tolist()

    # Save report JSON
    out_json = save_dir / "eval_report.json"
    payload = {
        "model_path": str(model_path),
        "data_dir": str(data_dir),
        "accuracy": acc,
        "labels": labels_sorted,
        "classification_report": report,
        "confusion_matrix": cm,
        "config_used": {
            "preprocess": {
                "detrend": detrend,
                "window": window_flag,
                "target_length": target_length,
                "resample_rate_hz": resample_rate_hz,
            },
            "extra": extra,
            "top_k_peaks": top_k,
        },
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"✅ Saved evaluation report: {out_json}")

    # Optionally save confusion matrix image
    try:
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        ax.imshow(np.array(cm), cmap="Blues")
        ax.set_xticks(range(len(labels_sorted)))
        ax.set_yticks(range(len(labels_sorted)))
        ax.set_xticklabels(labels_sorted, rotation=45, ha="right")
        ax.set_yticklabels(labels_sorted)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(len(labels_sorted)):
            for j in range(len(labels_sorted)):
                ax.text(j, i, cm[i][j], va="center", ha="center", color="black")
        fig.tight_layout()
        out_png = save_dir / "confusion_matrix.png"
        fig.savefig(out_png)
        plt.close(fig)
        print(f"🖼️ Saved confusion matrix image: {out_png}")
    except Exception as e:
        print(f"ℹ️ Skipped confusion matrix image: {e}")


def cmd_validate(args: argparse.Namespace) -> None:
    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"📁 ERROR: '{data_dir}' does not exist!")
        return

    valid = True
    has_json = False
    for json_file in data_dir.rglob("*.json"):
        has_json = True
        if not validate_sample(json_file):
            valid = False

    if not has_json:
        print(f"🟡 No JSON files found in {data_dir}")
        print("💡 Create a file like 'data/test_glass.json' to get started.")
    print("\n" + ("✅ ALL FILES VALID" if valid and has_json else "❌ SOME FILES INVALID"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="resdb", description="ResonanceDB CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # simulate
    ps = sub.add_parser("simulate", help="Generate synthetic vibration npz files")
    ps.add_argument("--out-dir", default=str(ROOT / "data" / "simulated"))
    ps.add_argument("--material", default=None, help="Material to generate (default: all)")
    ps.add_argument("--sample-rate", type=int, default=1000)
    ps.add_argument("--duration", type=float, default=2.0)
    ps.set_defaults(func=cmd_simulate)

    # train
    pt = sub.add_parser("train", help="Train classifier from JSON data")
    pt.add_argument("--data", default=str(ROOT / "data"))
    pt.add_argument("--out", default=str(ROOT / "models" / "material_model.pkl"))
    pt.add_argument("--test-size", type=float, default=0.3)
    pt.add_argument("--random-state", type=int, default=42)
    pt.add_argument("--extra", default=None, help="Extra features: 'all' or comma-separated names")
    pt.add_argument("--top-k-peaks", type=int, default=3, help="Top-K peaks when extras include 'top_peaks'")
    # Preprocess flags
    gpt = pt.add_argument_group("Preprocess")
    gpt.add_argument("--target-length", type=int, default=None, help="Normalize signal length to this value")
    gpt.add_argument("--resample-rate-hz", type=float, default=None, help="Resample signal to this rate (Hz)")
    gpt.add_argument("--detrend", dest="detrend", action="store_true", default=None, help="Enable mean detrending")
    gpt.add_argument("--no-detrend", dest="detrend", action="store_false", help="Disable mean detrending")
    gpt.add_argument("--window", dest="window", action="store_true", default=None, help="Apply Hann window")
    gpt.add_argument("--no-window", dest="window", action="store_false", help="Disable windowing")
    pt.set_defaults(func=cmd_train)

    # predict
    pp = sub.add_parser("predict", help="Predict material from a JSON file")
    pp.add_argument("file", help="Path to JSON data file")
    pp.add_argument("--model", default=str(ROOT / "models" / "material_model.pkl"))
    pp.add_argument("--extra", default=None, help="Extra features: 'all' or comma-separated names")
    pp.add_argument("--top-k-peaks", type=int, default=3, help="Top-K peaks when extras include 'top_peaks'")
    # Preprocess flags
    gpp = pp.add_argument_group("Preprocess")
    gpp.add_argument("--target-length", type=int, default=None, help="Normalize signal length to this value")
    gpp.add_argument("--resample-rate-hz", type=float, default=None, help="Resample signal to this rate (Hz)")
    gpp.add_argument("--detrend", dest="detrend", action="store_true", default=None, help="Enable mean detrending")
    gpp.add_argument("--no-detrend", dest="detrend", action="store_false", help="Disable mean detrending")
    gpp.add_argument("--window", dest="window", action="store_true", default=None, help="Apply Hann window")
    gpp.add_argument("--no-window", dest="window", action="store_false", help="Disable windowing")
    pp.set_defaults(func=cmd_predict)

    # inspect
    pi = sub.add_parser("inspect", help="Inspect saved model configuration")
    pi.add_argument("--model", default=str(ROOT / "models" / "material_model.pkl"))
    pi.set_defaults(func=cmd_inspect)

    # evaluate
    pe = sub.add_parser("evaluate", help="Evaluate model on a dataset")
    pe.add_argument("--model", default=str(ROOT / "models" / "material_model.pkl"))
    pe.add_argument("--data", default=str(ROOT / "data"))
    pe.add_argument("--save-dir", default=str(ROOT / "models"))
    pe.add_argument("--extra", default=None, help="Extra features: 'all' or comma-separated names")
    pe.add_argument("--top-k-peaks", type=int, default=3, help="Top-K peaks when extras include 'top_peaks'")
    gpe = pe.add_argument_group("Preprocess")
    gpe.add_argument("--target-length", type=int, default=None, help="Normalize signal length to this value")
    gpe.add_argument("--resample-rate-hz", type=float, default=None, help="Resample signal to this rate (Hz)")
    gpe.add_argument("--detrend", dest="detrend", action="store_true", default=None, help="Enable mean detrending")
    gpe.add_argument("--no-detrend", dest="detrend", action="store_false", help="Disable mean detrending")
    gpe.add_argument("--window", dest="window", action="store_true", default=None, help="Apply Hann window")
    gpe.add_argument("--no-window", dest="window", action="store_false", help="Disable windowing")
    pe.set_defaults(func=cmd_evaluate)

    # tune
    ptune = sub.add_parser("tune", help="Hyperparameter tuning with k-fold CV")
    ptune.add_argument("--data", default=str(ROOT / "data"))
    ptune.add_argument("--out", default=str(ROOT / "models" / "material_model.pkl"))
    ptune.add_argument("--cv", type=int, default=2, help="Number of CV folds (stratified)")
    ptune.add_argument("--random-state", type=int, default=42)
    # Grid flags (comma-separated)
    ptune.add_argument("--n-estimators", default="10,50", help="Grid for n_estimators")
    ptune.add_argument("--max-depth", default="None,5", help="Grid for max_depth")
    ptune.add_argument("--max-features", default="sqrt", help="Grid for max_features")
    ptune.add_argument("--extra", default=None, help="Extra features: 'all' or comma-separated names")
    ptune.add_argument("--top-k-peaks", type=int, default=3, help="Top-K peaks when extras include 'top_peaks'")
    gtune = ptune.add_argument_group("Preprocess")
    gtune.add_argument("--target-length", type=int, default=None, help="Normalize signal length to this value")
    gtune.add_argument("--resample-rate-hz", type=float, default=None, help="Resample signal to this rate (Hz)")
    gtune.add_argument("--detrend", dest="detrend", action="store_true", default=None, help="Enable mean detrending")
    gtune.add_argument("--no-detrend", dest="detrend", action="store_false", help="Disable mean detrending")
    gtune.add_argument("--window", dest="window", action="store_true", default=None, help="Apply Hann window")
    gtune.add_argument("--no-window", dest="window", action="store_false", help="Disable windowing")
    ptune.set_defaults(func=cmd_tune)
    
    # export
    pexp = sub.add_parser("export", help="Export trained model to portable formats")
    pexp.add_argument("--model", default=str(ROOT / "models" / "material_model.pkl"))
    pexp.add_argument("--out", default=None, help="Output path (defaults to .onnx next to model)")
    pexp.add_argument("--format", choices=["onnx", "tflite"], default="onnx")
    pexp.add_argument("--verify", dest="verify", action="store_true", default=True, help="Run a quick ONNXRuntime check")
    pexp.add_argument("--no-verify", dest="verify", action="store_false")
    pexp.set_defaults(func=cmd_export)

    # validate
    pv = sub.add_parser("validate", help="Validate data files against schema rules")
    pv.add_argument("--data", default=str(ROOT / "data"))
    pv.set_defaults(func=cmd_validate)

    return p


def cmd_tune(args: argparse.Namespace) -> None:
    data_dir = Path(args.data)
    out_model = Path(args.out)

    extra = _parse_extra_flag(args.extra)
    detrend_flag = True if (args.detrend is None) else bool(args.detrend)
    window_name = "hann" if (args.window is True or args.window is None) else None

    X, y = load_data(
        str(data_dir),
        extra=extra,
        top_k_peaks=args.top_k_peaks,
        detrend=detrend_flag,
        window=window_name,
        target_length=args.target_length,
        resample_rate_hz=args.resample_rate_hz,
    )
    if len(X) == 0:
        print("🛑 Tuning aborted: no data to learn from.")
        return

    # Determine stratified CV folds safely
    labels = list(y)
    uniques = sorted(set(labels))
    counts = {lab: labels.count(lab) for lab in uniques}
    cv_folds = int(args.cv)
    if len(uniques) < 2:
        print("⚠️ Only one class present; using CV=2 without stratification may fail. Add more classes.")
    if min(counts.values()) < cv_folds:
        cv_folds = max(2, min(counts.values()))
        print(f"ℹ️ Reduced CV folds to {cv_folds} due to limited samples per class.")

    try:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=args.random_state)
    except Exception:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=args.random_state)

    # Parse grids
    n_estimators_grid = _parse_list_flag(args.n_estimators, int) or [10, 50]
    max_depth_grid = _parse_list_flag(args.max_depth, lambda s: None if s.strip().lower() == "none" else int(s)) or [None, 5]
    max_features_grid = _parse_list_flag(args.max_features, str) or ["sqrt"]

    param_grid = {
        "n_estimators": n_estimators_grid,
        "max_depth": max_depth_grid,
        "max_features": max_features_grid,
    }

    base = RandomForestClassifier(random_state=args.random_state)
    gs = GridSearchCV(base, param_grid=param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    gs.fit(X, y)

    best_model = gs.best_estimator_
    best_acc = float(gs.best_score_)
    print("✅ Tuning complete")
    print("🏆 Best params:", gs.best_params_)
    print(f"📊 Best CV accuracy: {best_acc * 100:.1f}% ({cv_folds}-fold)")

    # Persist tuned model with metadata
    package = {
        "model": best_model,
        "config": {
            "preprocess": {
                "detrend": detrend_flag,
                "window": window_name is not None,
                "target_length": args.target_length,
                "resample_rate_hz": args.resample_rate_hz,
            },
            "extra": extra,
            "top_k_peaks": args.top_k_peaks,
            "input_dim": int(X.shape[1]) if hasattr(X, "shape") else int(len(X[0])),
            "best_params": gs.best_params_,
            "cv_folds": cv_folds,
            "best_cv_accuracy": best_acc,
        },
    }
    joblib.dump(package, str(out_model))
    print(f"✅ Tuned model saved to {out_model}")


def cmd_export(args: argparse.Namespace) -> None:
    """Export a trained model package to ONNX (or TFLite stub)."""
    model_path = Path(args.model) if args.model else (ROOT / "models" / "material_model.pkl")
    out_path = Path(args.out) if args.out else Path(model_path).with_suffix(".onnx")

    loaded = joblib.load(str(model_path))
    if isinstance(loaded, dict) and "model" in loaded:
        model = loaded["model"]
        cfg = loaded.get("config", {})
    else:
        model = loaded
        cfg = {}

    input_dim = int(cfg.get("input_dim", getattr(model, "n_features_in_", 0) or 0))
    if input_dim <= 0:
        print("⚠️ Unable to determine input dimension; assuming 10.")
        input_dim = 10

    fmt = (args.format or "onnx").lower()
    if fmt == "onnx":
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            import onnx
        except ImportError:
            print("❌ ONNX export requires 'skl2onnx' and 'onnx'. Install via: pip install skl2onnx onnx")
            return

        initial_types = [("input", FloatTensorType([None, input_dim]))]
        options = {type(model): {"zipmap": False}}
        try:
            model_onnx = convert_sklearn(model, initial_types=initial_types, options=options)
        except Exception as e:
            print(f"❌ ONNX conversion failed: {e}")
            return

        # Attach minimal metadata for portability
        try:
            meta_items = {
                "classes": json.dumps(list(getattr(model, "classes_", []))),
                "input_dim": str(input_dim),
                "extra": json.dumps(cfg.get("extra", False)),
                "top_k_peaks": str(cfg.get("top_k_peaks", 3)),
                "preprocess": json.dumps(cfg.get("preprocess", {})),
            }
            if "best_params" in cfg:
                meta_items["best_params"] = json.dumps(cfg["best_params"])
            if "best_cv_accuracy" in cfg:
                meta_items["best_cv_accuracy"] = str(cfg["best_cv_accuracy"]) 
            if "cv_folds" in cfg:
                meta_items["cv_folds"] = str(cfg["cv_folds"]) 
            for k, v in meta_items.items():
                prop = model_onnx.metadata_props.add()
                prop.key = f"resdb_{k}"
                prop.value = v
        except Exception:
            # Non-fatal; continue without metadata
            pass

        with open(out_path, "wb") as f:
            f.write(model_onnx.SerializeToString())
        print(f"✅ ONNX model saved to {out_path}")

        if args.verify:
            try:
                import onnxruntime as ort
            except ImportError:
                print("ℹ️ Skipping verification: 'onnxruntime' not installed.")
            else:
                try:
                    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
                    inp_name = sess.get_inputs()[0].name
                    dummy = np.zeros((1, input_dim), dtype=np.float32)
                    _outputs = sess.run(None, {inp_name: dummy})
                    print("🔍 ONNX runtime inference succeeded.")
                except Exception as e:
                    print(f"⚠️ ONNXRuntime verification failed: {e}")
    elif fmt == "tflite":
        print("⚠️ TFLite export is not supported for scikit-learn RandomForest.")
        print("💡 Consider ONNX export, or retrain with TensorFlow/Keras for TFLite.")
    else:
        print(f"❌ Unknown export format '{args.format}'. Use 'onnx' or 'tflite'.")


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()