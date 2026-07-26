"""resdb, the ResonanceDB command line.

Subcommands: simulate, validate, train, tune, evaluate, predict, inspect, export.

Heavy dependencies (scikit-learn, matplotlib, skl2onnx) are imported lazily so
cheap commands like `validate` start instantly and work without the optional
extras installed.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .dataset import load_data
from .features import compute_feature_vector
from .schema import validate_tree
from .simulate import DEFAULT_SAMPLE_RATE, MATERIALS, generate_dataset

DEFAULT_MODEL = Path("models") / "material_model.pkl"


# ---------------------------------------------------------------- helpers

def _parse_extra_flag(extra_flag: str | None):
    if extra_flag is None:
        return False
    flag = extra_flag.strip().lower()
    if flag in ("", "none", "false"):
        return False
    if flag in ("all", "true"):
        return True
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


def _load_model_package(model_path: Path):
    """Load a saved model file; returns (model, config_dict)."""
    import joblib

    loaded = joblib.load(str(model_path))
    if isinstance(loaded, dict) and "model" in loaded:
        return loaded["model"], loaded.get("config", {})
    return loaded, {}


def _window_name(window_flag: bool | None) -> str | None:
    """Map the CLI tri-state (--window/--no-window/unset) to a window name."""
    return "hann" if (window_flag is True or window_flag is None) else None


def _effective_predict_config(args, saved_cfg: dict):
    """Resolve feature configuration: CLI flags override the saved config."""
    saved_pre = saved_cfg.get("preprocess", {})
    extra = _parse_extra_flag(args.extra) if args.extra is not None else saved_cfg.get("extra", False)
    top_k = args.top_k_peaks if args.top_k_peaks is not None else int(saved_cfg.get("top_k_peaks", 3))
    detrend = args.detrend if args.detrend is not None else bool(saved_pre.get("detrend", True))
    window_flag = args.window if args.window is not None else bool(saved_pre.get("window", True))
    target_length = args.target_length if args.target_length is not None else saved_pre.get("target_length")
    resample = args.resample_rate_hz if args.resample_rate_hz is not None else saved_pre.get("resample_rate_hz")
    highpass = args.highpass_hz if args.highpass_hz is not None else saved_pre.get("highpass_hz")
    return {
        "detrend": detrend,
        "window": "hann" if window_flag else None,
        "target_length": target_length,
        "resample_rate_hz": resample,
        "highpass_hz": highpass,
        "extra": extra,
        "top_k_peaks": top_k,
    }


# ---------------------------------------------------------------- commands

def cmd_simulate(args: argparse.Namespace) -> int:
    mats = MATERIALS
    if args.material is not None:
        sel = args.material.lower()
        if sel not in mats:
            print(f"[FAIL] Unknown material '{args.material}'. Available: {list(mats)}")
            return 1
        mats = {sel: mats[sel]}

    try:
        written = generate_dataset(args.out_dir, sample_rate=args.sample_rate,
                                   duration=args.duration, materials=mats)
    except ValueError as e:
        print(f"[FAIL] {e}")
        return 1
    for path in written:
        print(f"[OK] Generated {path}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """Convert a WAV recording into one schema-valid JSON sample per tap."""
    from .audio import wav_to_samples
    from .schema import validate_sample_dict

    wav_path = Path(args.wav)
    if not wav_path.exists():
        print(f"[FAIL] WAV file not found: {wav_path}")
        return 1

    try:
        samples = wav_to_samples(
            wav_path,
            args.material.lower(),
            device=args.device,
            session=args.session,
            obj=args.object,
            excitation=args.excitation,
            source=args.source,
            striker=args.striker,
            threshold_ratio=args.threshold_ratio,
            min_separation_s=args.min_separation,
            duration_s=args.duration,
        )
    except Exception as e:
        print(f"[FAIL] Could not read {wav_path}: {e}")
        return 1

    if not samples:
        print(f"[FAIL] No taps detected in {wav_path.name}. "
              "Try lowering --threshold-ratio, or check the recording.")
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else Path("data") / args.material.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    session = samples[0]["session"]

    # The object must be part of the filename. Without it, two different
    # objects of the same material recorded on the same device in the same
    # session collide, and --force silently replaces one with the other.
    obj_part = f"{args.object}_" if args.object else ""

    # Guard against silently overwriting a different recording. Re-using the
    # same labels for a second recording would replace the first one's taps
    # and leave a mix of two recordings under one label.
    prefix = f"{args.material.lower()}_{args.device}_{obj_part}{session}_tap"
    existing = sorted(out_dir.glob(f"{prefix}*.json"))
    if existing and not args.force:
        print(f"[FAIL] {len(existing)} sample(s) already exist for "
              f"material={args.material.lower()} device={args.device} "
              f"session={session} in {out_dir}.")
        print("       Ingesting here would overwrite another recording's taps.")
        print("       Use a different --session for this recording, or pass "
              "--force to replace the existing ones.")
        return 1

    written = 0
    for i, sample in enumerate(samples, start=1):
        errors = validate_sample_dict(sample)
        if errors:
            print(f"[SKIP] tap {i}: {'; '.join(errors)}")
            continue
        out_path = out_dir / f"{args.material.lower()}_{args.device}_{obj_part}{session}_tap{i:02d}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(sample, f)
        print(f"[OK] {out_path} ({len(sample['vibration'])} samples "
              f"@ {sample['sample_rate_hz']} Hz)")
        written += 1

    print(f"\n{written} tap(s) written from {wav_path.name}")

    # Clipping destroys exactly the spectral detail that distinguishes hard
    # ringing materials from each other, and it cannot be undone afterwards.
    # Warn now, while re-recording is still cheap.
    clipped = sum(1 for s in samples
                  if max(abs(v) for v in s["vibration"]) >= 0.99)
    if clipped:
        pct = 100 * clipped / len(samples)
        print(f"[WARN] {clipped} of {len(samples)} taps ({pct:.0f}%) are clipped "
              "(they hit full scale).")
        print("       Clipping adds false harmonics and erases the detail that "
              "separates ringing materials.")
        print("       Re-record further from the object or tapping more gently, "
              "then ingest with --force.")
    return 0 if written else 1


def cmd_summary(args: argparse.Namespace) -> int:
    """Per-session overview: tap counts, frequencies, and recording quality."""
    from .summary import format_summary, summarize_dataset

    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"[FAIL] '{data_dir}' does not exist")
        return 1

    report = summarize_dataset(
        data_dir,
        # 0 means "no filtering" rather than an invalid cutoff
        highpass_hz=args.highpass_hz or None,
        include_simulated=args.include_simulated,
    )
    print(format_summary(report))

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n[OK] Saved summary: {out}")
    return 0 if report["sessions"] else 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Leave-one-group-out evaluation: the honest cross-device number."""
    from .benchmark import leave_one_group_out

    extra = _parse_extra_flag(args.extra) if args.extra is not None else True
    top_k = 3 if args.top_k_peaks is None else int(args.top_k_peaks)
    detrend = True if args.detrend is None else bool(args.detrend)

    X, y, meta = load_data(
        str(Path(args.data)),
        extra=extra,
        top_k_peaks=top_k,
        detrend=detrend,
        window=_window_name(args.window),
        target_length=args.target_length,
        resample_rate_hz=args.resample_rate_hz,
        highpass_hz=args.highpass_hz,
        return_meta=True,
    )
    if len(X) == 0:
        print("[FAIL] Benchmark aborted: no data.")
        return 1

    # Simulated samples are for pipeline testing; mixing them into a real
    # benchmark corrupts the number the benchmark exists to produce.
    if not args.include_simulated:
        keep = [i for i, m in enumerate(meta) if m.get("source") != "simulation"]
        dropped = len(meta) - len(keep)
        if dropped:
            print(f"[INFO] Excluding {dropped} simulated sample(s) "
                  "(use --include-simulated to keep them)")
        X = X[keep]
        y = y[keep]
        meta = [meta[i] for i in keep]
    if len(X) == 0:
        print("[FAIL] Benchmark aborted: only simulated data found.")
        return 1

    if len(set(y.tolist())) < 2:
        print(f"[FAIL] Benchmark needs at least 2 materials; found only "
              f"{sorted(set(y.tolist()))}. A single-class benchmark would "
              "score 100% while proving nothing. Record more materials first.")
        return 1

    if args.label_by == "object":
        y = np.array([m.get("object") or "unknown" for m in meta])

    groups = [m.get(args.group_by) or "unknown" for m in meta]
    try:
        report = leave_one_group_out(X, y, groups, random_state=args.random_state)
    except ValueError as e:
        print(f"[FAIL] {e}")
        return 1

    print(f"\nLeave-one-{args.group_by}-out benchmark "
          f"({report['n_samples']} samples, {report['n_groups']} groups, "
          f"classes: {report['classes']})")
    for group, stats in report["groups"].items():
        if stats["unseen_classes"]:
            names = ", ".join(str(c) for c in stats["unseen_classes"])
            print(f"  {group}: not evaluable (n={stats['n']}), no {names} "
                  "anywhere else to train on")
        else:
            print(f"  {group}: {stats['accuracy'] * 100:.1f}% (n={stats['n']})")

    chance = report.get("chance_accuracy")
    chance_txt = f"  (chance {chance * 100:.0f}%)" if chance else ""
    if report["unevaluable_groups"]:
        print(f"\n  {len(report['unevaluable_groups'])} group(s) could not be "
              "evaluated: their material appears in no other group, so nothing "
              "could be learned from it.")
        print("  Record that material a second time, on another object or "
              "device, to include it.")
        if report["mean_evaluable_group_accuracy"] is not None:
            print(f"\n  Mean accuracy over the {report['n_evaluable_groups']} "
                  f"evaluable group(s): "
                  f"{report['mean_evaluable_group_accuracy'] * 100:.1f}%{chance_txt}")
            print(f"  Pooled over evaluable groups:    "
                  f"{report['pooled_evaluable_accuracy'] * 100:.1f}%")
        print(f"\n  Counting unevaluable groups as 0%, mean would be "
              f"{report['mean_group_accuracy'] * 100:.1f}%")
    else:
        print(f"\n  Mean per-group accuracy: "
              f"{report['mean_group_accuracy'] * 100:.1f}%{chance_txt}")
        print(f"  Pooled accuracy:         {report['pooled_accuracy'] * 100:.1f}%")

    if report.get("per_class_recall"):
        print("\n  Per-class recall:")
        for c, v in report["per_class_recall"].items():
            print(f"    {c:12s} {v * 100:5.1f}%")
        bal = report["balanced_accuracy"]
        base = report.get("majority_baseline")
        print(f"\n  Balanced accuracy (mean over classes): {bal * 100:.1f}%")
        if base is not None:
            print(f"  Always guessing the commonest class:   {base * 100:.1f}%")
            headline = report.get("mean_evaluable_group_accuracy")
            if headline is None:
                headline = report["mean_group_accuracy"]
            if headline <= base + 0.05:
                print("\n  [WARN] The model barely beats that trivial baseline. "
                      "Plain accuracy is being flattered by class imbalance; "
                      "read the balanced accuracy instead.")

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_json = save_dir / "benchmark_report.json"
        payload = dict(report)
        payload["group_by"] = args.group_by
        payload["data_dir"] = str(args.data)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[OK] Saved benchmark report: {out_json}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"[FAIL] '{data_dir}' does not exist")
        return 1

    valid, invalid = validate_tree(data_dir)
    total = valid + invalid
    if total == 0:
        print(f"[WARN] No JSON files found in {data_dir}")
        return 1
    print(f"\n{valid}/{total} files valid")
    return 0 if invalid == 0 else 1


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
    highpass_hz: float | None = None,
) -> bool:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    import joblib

    extra = _parse_extra_flag(extra_flag)
    window_name = _window_name(window)
    detrend_flag = True if detrend is None else bool(detrend)
    top_k_peaks = 3 if top_k_peaks is None else int(top_k_peaks)

    X, y = load_data(
        str(data_dir),
        extra=extra,
        top_k_peaks=top_k_peaks,
        detrend=detrend_flag,
        window=window_name,
        target_length=target_length,
        resample_rate_hz=resample_rate_hz,
        highpass_hz=highpass_hz,
    )
    if len(X) == 0:
        print("[FAIL] Training aborted: no data to learn from.")
        return False

    try:
        stratify = None
        if len(set(y)) > 1:
            class_counts = {}
            for label in y:
                class_counts[label] = class_counts.get(label, 0) + 1
            if min(class_counts.values()) >= 2:
                stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
    except ValueError as e:
        print(f"[FAIL] Split failed: {e}")
        print("       Try adding more samples (at least 2 different materials)")
        return False

    clf = RandomForestClassifier(n_estimators=10, random_state=random_state)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"\n[OK] Model trained. Test accuracy: {acc * 100:.1f}%")
    print(f"     Classes: {sorted(set(y))}")

    package = {
        "model": clf,
        "config": {
            "preprocess": {
                "detrend": detrend_flag,
                "window": window_name is not None,
                "target_length": target_length,
                "resample_rate_hz": resample_rate_hz,
                "highpass_hz": highpass_hz,
            },
            "extra": extra,
            "top_k_peaks": top_k_peaks,
            "input_dim": int(X.shape[1]),
        },
    }
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, str(out_model))
    print(f"[OK] Model saved to {out_model}")
    return True


def cmd_train(args: argparse.Namespace) -> int:
    ok = train_model(
        Path(args.data),
        Path(args.out),
        test_size=args.test_size,
        random_state=args.random_state,
        extra_flag=args.extra,
        top_k_peaks=args.top_k_peaks,
        detrend=args.detrend,
        window=args.window,
        target_length=args.target_length,
        resample_rate_hz=args.resample_rate_hz,
        highpass_hz=args.highpass_hz,
    )
    return 0 if ok else 1


def cmd_tune(args: argparse.Namespace) -> int:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    import joblib

    extra = _parse_extra_flag(args.extra)
    detrend_flag = True if (args.detrend is None) else bool(args.detrend)
    window_name = _window_name(args.window)
    top_k_peaks = 3 if args.top_k_peaks is None else int(args.top_k_peaks)

    X, y = load_data(
        str(Path(args.data)),
        extra=extra,
        top_k_peaks=top_k_peaks,
        detrend=detrend_flag,
        window=window_name,
        target_length=args.target_length,
        resample_rate_hz=args.resample_rate_hz,
        highpass_hz=args.highpass_hz,
    )
    if len(X) == 0:
        print("[FAIL] Tuning aborted: no data to learn from.")
        return 1

    labels = list(y)
    uniques = sorted(set(labels))
    counts = {lab: labels.count(lab) for lab in uniques}
    cv_folds = int(args.cv)
    if len(uniques) < 2:
        print("[WARN] Only one class present; add more classes.")
    if min(counts.values()) < cv_folds:
        cv_folds = max(2, min(counts.values()))
        print(f"[WARN] Reduced CV folds to {cv_folds} due to limited samples per class.")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=args.random_state)

    param_grid = {
        "n_estimators": _parse_list_flag(args.n_estimators, int) or [10, 50],
        "max_depth": _parse_list_flag(
            args.max_depth, lambda s: None if s.strip().lower() == "none" else int(s)
        ) or [None, 5],
        "max_features": _parse_list_flag(args.max_features, str) or ["sqrt"],
    }

    base = RandomForestClassifier(random_state=args.random_state)
    gs = GridSearchCV(base, param_grid=param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    gs.fit(X, y)

    best_acc = float(gs.best_score_)
    print("[OK] Tuning complete")
    print(f"     Best params: {gs.best_params_}")
    print(f"     Best CV accuracy: {best_acc * 100:.1f}% ({cv_folds}-fold)")

    out_model = Path(args.out)
    package = {
        "model": gs.best_estimator_,
        "config": {
            "preprocess": {
                "detrend": detrend_flag,
                "window": window_name is not None,
                "target_length": args.target_length,
                "resample_rate_hz": args.resample_rate_hz,
                "highpass_hz": args.highpass_hz,
            },
            "extra": extra,
            "top_k_peaks": top_k_peaks,
            "input_dim": int(X.shape[1]),
            "best_params": gs.best_params_,
            "cv_folds": cv_folds,
            "best_cv_accuracy": best_acc,
        },
    }
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, str(out_model))
    print(f"[OK] Tuned model saved to {out_model}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    model_path = Path(args.model) if args.model else DEFAULT_MODEL
    try:
        model, saved_cfg = _load_model_package(model_path)
    except FileNotFoundError:
        print(f"[FAIL] Model not found: {model_path}. Run 'resdb train' first.")
        return 1

    try:
        with open(args.file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[FAIL] Data file not found: {args.file}")
        return 1

    signal = np.array(data["vibration"])
    sample_rate = data["sample_rate_hz"]

    cfg = _effective_predict_config(args, saved_cfg)
    vec = compute_feature_vector(signal, sample_rate, **cfg)

    n_in = getattr(model, "n_features_in_", len(vec))
    if len(vec) != n_in and saved_cfg:
        # CLI overrides produced a shape the model can't take; fall back to
        # the configuration the model was trained with.
        print(f"[WARN] Feature length {len(vec)} != model expects {n_in}; "
              "using saved model configuration.")
        pre = saved_cfg.get("preprocess", {})
        vec = compute_feature_vector(
            signal, sample_rate,
            detrend=bool(pre.get("detrend", True)),
            window="hann" if pre.get("window", True) else None,
            target_length=pre.get("target_length"),
            resample_rate_hz=pre.get("resample_rate_hz"),
            highpass_hz=pre.get("highpass_hz"),
            extra=saved_cfg.get("extra", False),
            top_k_peaks=int(saved_cfg.get("top_k_peaks", 3)),
        )

    prediction = model.predict(np.array([vec]))[0]
    print(f"Prediction: {prediction}")
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(np.array([vec]))[0]
        pairs = sorted(zip(model.classes_, proba), key=lambda t: -t[1])
        print("Probabilities: " + ", ".join(f"{c}={p:.2f}" for c, p in pairs))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    model_path = Path(args.model) if args.model else DEFAULT_MODEL
    try:
        model, cfg = _load_model_package(model_path)
    except FileNotFoundError:
        print(f"[FAIL] Model not found: {model_path}")
        return 1

    print(f"Path: {model_path}")
    if cfg:
        print("Format: model package (with config)")
        print(f"Input dim: {cfg.get('input_dim', getattr(model, 'n_features_in_', '?'))}")
        print(f"Extras: {cfg.get('extra', False)}")
        print(f"Top-K peaks: {cfg.get('top_k_peaks', 3)}")
        pre = cfg.get("preprocess", {})
        print("Preprocess:")
        print(f"  detrend: {pre.get('detrend', True)}")
        print(f"  window: {pre.get('window', True)}")
        print(f"  target_length: {pre.get('target_length')}")
        print(f"  resample_rate_hz: {pre.get('resample_rate_hz')}")
        print(f"  highpass_hz: {pre.get('highpass_hz')}")
        if "best_params" in cfg:
            print(f"Best params: {cfg['best_params']}")
        if "best_cv_accuracy" in cfg:
            print(f"Best CV accuracy: {float(cfg['best_cv_accuracy']) * 100:.1f}% "
                  f"({cfg.get('cv_folds', '?')}-fold)")
    else:
        print("Format: raw sklearn model (no embedded config)")
        print(f"Input dim: {getattr(model, 'n_features_in_', '?')}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    model_path = Path(args.model) if args.model else DEFAULT_MODEL
    try:
        model, saved_cfg = _load_model_package(model_path)
    except FileNotFoundError:
        print(f"[FAIL] Model not found: {model_path}")
        return 1

    cfg = _effective_predict_config(args, saved_cfg)
    X, y = load_data(
        str(Path(args.data)),
        extra=cfg["extra"],
        top_k_peaks=cfg["top_k_peaks"],
        detrend=cfg["detrend"],
        window=cfg["window"],
        target_length=cfg["target_length"],
        resample_rate_hz=cfg["resample_rate_hz"],
        highpass_hz=cfg["highpass_hz"],
    )
    if len(X) == 0:
        print("[FAIL] Evaluation aborted: no data to evaluate.")
        return 1

    y_pred = model.predict(X)
    acc = float(accuracy_score(y, y_pred))
    labels_sorted = sorted(set(y))
    cm = confusion_matrix(y, y_pred, labels=labels_sorted).tolist()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_json = save_dir / "eval_report.json"
    payload = {
        "model_path": str(model_path),
        "data_dir": str(args.data),
        "accuracy": acc,
        "labels": labels_sorted,
        "classification_report": classification_report(y, y_pred, output_dict=True),
        "confusion_matrix": cm,
        "config_used": cfg,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] Accuracy: {acc * 100:.1f}%")
    print(f"[OK] Saved evaluation report: {out_json}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

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
        print(f"[OK] Saved confusion matrix image: {out_png}")
    except ImportError:
        print("[INFO] matplotlib not installed; skipped confusion matrix image "
              "(pip install 'resonancedb[plots]')")
    except Exception as e:
        print(f"[INFO] Skipped confusion matrix image: {e}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    model_path = Path(args.model) if args.model else DEFAULT_MODEL
    out_path = Path(args.out) if args.out else model_path.with_suffix(".onnx")

    try:
        model, cfg = _load_model_package(model_path)
    except FileNotFoundError:
        print(f"[FAIL] Model not found: {model_path}")
        return 1

    input_dim = int(cfg.get("input_dim", getattr(model, "n_features_in_", 0) or 0))
    if input_dim <= 0:
        print("[FAIL] Unable to determine input dimension from model.")
        return 1

    fmt = (args.format or "onnx").lower()
    if fmt != "onnx":
        print(f"[FAIL] Unsupported export format '{args.format}'. "
              "TFLite is not supported for scikit-learn models; use 'onnx'.")
        return 1

    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        print("[FAIL] ONNX export requires the export extra: "
              "pip install 'resonancedb[export]'")
        return 1

    initial_types = [("input", FloatTensorType([None, input_dim]))]
    options = {type(model): {"zipmap": False}}
    try:
        model_onnx = convert_sklearn(model, initial_types=initial_types, options=options)
    except Exception as e:
        print(f"[FAIL] ONNX conversion failed: {e}")
        return 1

    meta_items = {
        "classes": json.dumps([str(c) for c in getattr(model, "classes_", [])]),
        "input_dim": str(input_dim),
        "extra": json.dumps(cfg.get("extra", False)),
        "top_k_peaks": str(cfg.get("top_k_peaks", 3)),
        "preprocess": json.dumps(cfg.get("preprocess", {})),
    }
    for key in ("best_params", "best_cv_accuracy", "cv_folds"):
        if key in cfg:
            meta_items[key] = json.dumps(cfg[key])
    for k, v in meta_items.items():
        prop = model_onnx.metadata_props.add()
        prop.key = f"resdb_{k}"
        prop.value = v

    with open(out_path, "wb") as f:
        f.write(model_onnx.SerializeToString())
    print(f"[OK] ONNX model saved to {out_path}")

    if args.verify:
        try:
            import onnxruntime as ort
        except ImportError:
            print("[INFO] Skipping verification: onnxruntime not installed.")
            return 0
        try:
            sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
            inp_name = sess.get_inputs()[0].name
            sess.run(None, {inp_name: np.zeros((1, input_dim), dtype=np.float32)})
            print("[OK] ONNX runtime inference check passed.")
        except Exception as e:
            print(f"[WARN] ONNXRuntime verification failed: {e}")
            return 1
    return 0


# ---------------------------------------------------------------- parser

def _add_preprocess_flags(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Preprocess")
    g.add_argument("--target-length", type=int, default=None,
                   help="Normalize signal length to this value")
    g.add_argument("--resample-rate-hz", type=float, default=None,
                   help="Resample signal to this rate (Hz)")
    g.add_argument("--highpass-hz", type=float, default=None,
                   help="Remove content below this frequency (Hz). Recommended "
                        "for microphone recordings, where sub-100 Hz rumble "
                        "otherwise dominates every feature. Try 150.")
    g.add_argument("--detrend", dest="detrend", action="store_true", default=None,
                   help="Enable mean detrending")
    g.add_argument("--no-detrend", dest="detrend", action="store_false",
                   help="Disable mean detrending")
    g.add_argument("--window", dest="window", action="store_true", default=None,
                   help="Apply Hann window")
    g.add_argument("--no-window", dest="window", action="store_false",
                   help="Disable windowing")


def _add_feature_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--extra", default=None,
                        help="Extra features: 'all' or comma-separated names")
    parser.add_argument("--top-k-peaks", type=int, default=None,
                        help="Top-K peaks when extras include 'top_peaks'")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="resdb", description="ResonanceDB CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("simulate", help="Generate synthetic vibration JSON samples")
    ps.add_argument("--out-dir", default="data/simulated")
    ps.add_argument("--material", default=None, help="Material to generate (default: all)")
    ps.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    ps.add_argument("--duration", type=float, default=2.0)
    ps.set_defaults(func=cmd_simulate)

    pv = sub.add_parser("validate", help="Validate data files against the schema")
    pv.add_argument("--data", default="data")
    pv.set_defaults(func=cmd_validate)

    ping = sub.add_parser(
        "ingest",
        help="Convert a WAV recording into JSON samples (one per detected tap)",
    )
    ping.add_argument("wav", help="Path to a WAV recording containing one or more taps")
    ping.add_argument("--material", required=True, help="Material label, e.g. oak_wood")
    ping.add_argument("--device", required=True,
                      help="Recording device, e.g. pixel7, iphone14, usb_mic")
    ping.add_argument("--session", default=None,
                      help="Session id, one recording occasion (default: WAV filename stem)")
    ping.add_argument("--object", default=None,
                      help="Which physical object this is, e.g. kitchen_table. "
                           "Distinct from --session: record the same object on "
                           "two occasions with the same --object and different "
                           "--session to test whether its signature is stable.")
    ping.add_argument("--out-dir", default=None,
                      help="Output directory (default: data/<material>/)")
    ping.add_argument("--excitation", default="manual_tap")
    ping.add_argument("--source", default="microphone")
    ping.add_argument("--striker", default=None,
                      help="What you tapped with, e.g. finger, key, pen, coin. "
                           "Strongly recommended: a hard striker excites much "
                           "higher frequencies than a fingertip, so leaving it "
                           "unrecorded hides a variable that affects every "
                           "comparison.")
    ping.add_argument("--threshold-ratio", type=float, default=0.25,
                      help="Tap detection threshold as a fraction of the loudest event")
    ping.add_argument("--min-separation", type=float, default=0.25,
                      help="Minimum seconds between distinct taps")
    ping.add_argument("--duration", type=float, default=0.5,
                      help="Seconds to keep per tap")
    ping.add_argument("--force", action="store_true", default=False,
                      help="Overwrite existing samples with the same "
                           "material/device/session labels")
    ping.set_defaults(func=cmd_ingest)

    psum = sub.add_parser(
        "summary",
        help="Per-session overview: tap counts, frequencies, recording quality",
    )
    psum.add_argument("--data", default="data")
    psum.add_argument("--highpass-hz", type=float, default=150.0,
                      help="High-pass cutoff used for the reported frequencies "
                           "(default 150; pass 0 to disable)")
    psum.add_argument("--include-simulated", action="store_true", default=False,
                      help="Also list samples with source=simulation")
    psum.add_argument("--json", default=None,
                      help="Also write the summary to this JSON path")
    psum.set_defaults(func=cmd_summary)

    pb = sub.add_parser(
        "benchmark",
        help="Leave-one-group-out evaluation (group by device/session/file)",
    )
    pb.add_argument("--data", default="data")
    pb.add_argument("--group-by", choices=["device", "session", "file"],
                    default="device",
                    help="What counts as a held-out group (default: device)")
    pb.add_argument("--label-by", choices=["material", "object"], default="material",
                    help="What the model predicts. 'object' asks whether an "
                         "object's signature is recognisable in a different "
                         "session or on a different device (default: material)")
    pb.add_argument("--random-state", type=int, default=42)
    pb.add_argument("--save-dir", default=None,
                    help="Directory to write benchmark_report.json")
    pb.add_argument("--include-simulated", action="store_true", default=False,
                    help="Also benchmark samples with source=simulation (off by default)")
    _add_feature_flags(pb)
    _add_preprocess_flags(pb)
    pb.set_defaults(func=cmd_benchmark)

    pt = sub.add_parser("train", help="Train classifier from JSON data")
    pt.add_argument("--data", default="data")
    pt.add_argument("--out", default=str(DEFAULT_MODEL))
    pt.add_argument("--test-size", type=float, default=0.3)
    pt.add_argument("--random-state", type=int, default=42)
    _add_feature_flags(pt)
    _add_preprocess_flags(pt)
    pt.set_defaults(func=cmd_train)

    ptune = sub.add_parser("tune", help="Hyperparameter tuning with k-fold CV")
    ptune.add_argument("--data", default="data")
    ptune.add_argument("--out", default=str(DEFAULT_MODEL))
    ptune.add_argument("--cv", type=int, default=2, help="Number of CV folds (stratified)")
    ptune.add_argument("--random-state", type=int, default=42)
    ptune.add_argument("--n-estimators", default="10,50", help="Grid for n_estimators")
    ptune.add_argument("--max-depth", default="None,5", help="Grid for max_depth")
    ptune.add_argument("--max-features", default="sqrt", help="Grid for max_features")
    _add_feature_flags(ptune)
    _add_preprocess_flags(ptune)
    ptune.set_defaults(func=cmd_tune)

    pe = sub.add_parser("evaluate", help="Evaluate model on a dataset")
    pe.add_argument("--model", default=str(DEFAULT_MODEL))
    pe.add_argument("--data", default="data")
    pe.add_argument("--save-dir", default="models")
    _add_feature_flags(pe)
    _add_preprocess_flags(pe)
    pe.set_defaults(func=cmd_evaluate)

    pp = sub.add_parser("predict", help="Predict material from a JSON file")
    pp.add_argument("file", help="Path to JSON data file")
    pp.add_argument("--model", default=str(DEFAULT_MODEL))
    _add_feature_flags(pp)
    _add_preprocess_flags(pp)
    pp.set_defaults(func=cmd_predict)

    pi = sub.add_parser("inspect", help="Inspect saved model configuration")
    pi.add_argument("--model", default=str(DEFAULT_MODEL))
    pi.set_defaults(func=cmd_inspect)

    pexp = sub.add_parser("export", help="Export trained model to ONNX")
    pexp.add_argument("--model", default=str(DEFAULT_MODEL))
    pexp.add_argument("--out", default=None, help="Output path (defaults to .onnx next to model)")
    pexp.add_argument("--format", choices=["onnx"], default="onnx")
    pexp.add_argument("--verify", dest="verify", action="store_true", default=True,
                      help="Run a quick ONNXRuntime check")
    pexp.add_argument("--no-verify", dest="verify", action="store_false")
    pexp.set_defaults(func=cmd_export)

    return p


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
