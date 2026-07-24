"""Group-aware benchmarking: the honest number.

Random train/test splits flatter this project: taps from the same recording
session land on both sides of the split, so the model gets credit for
memorizing the session, not for recognizing the material. The product's
actual promise is generalization to a device it has never seen.

This module evaluates with leave-one-group-out: for each group (device,
session, or file), train on everything else and test on the held-out group.
"""

import numpy as np


def leave_one_group_out(
    X,
    y,
    groups: list[str],
    *,
    random_state: int = 42,
    n_estimators: int = 50,
) -> dict:
    """Leave-one-group-out evaluation with a RandomForest baseline.

    Returns a report dict:
      groups: {group: {"n": int, "accuracy": float, "unseen_classes": [...]}}
      mean_group_accuracy: unweighted mean over groups
      pooled_accuracy: accuracy over all held-out predictions pooled together
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    X = np.asarray(X)
    y = np.asarray(y)
    groups_arr = np.asarray(groups)

    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        raise ValueError(
            "Need at least 2 groups for leave-one-group-out evaluation; "
            f"got {unique_groups}. Record with a second device, or check that "
            "your samples carry 'device'/'session' metadata."
        )

    report: dict = {"groups": {}}
    all_true: list = []
    all_pred: list = []

    for group in unique_groups:
        test_mask = groups_arr == group
        train_mask = ~test_mask

        clf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
        clf.fit(X[train_mask], y[train_mask])
        y_pred = clf.predict(X[test_mask])
        y_true = y[test_mask]

        # Classes present only in the held-out group can never be predicted
        # correctly; they are reported, not hidden, because that gap is real.
        train_classes = set(y[train_mask])
        unseen = sorted(set(y_true) - train_classes)

        acc = float(accuracy_score(y_true, y_pred))
        report["groups"][group] = {
            "n": int(test_mask.sum()),
            "accuracy": acc,
            "unseen_classes": unseen,
        }
        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())

    group_accs = [g["accuracy"] for g in report["groups"].values()]
    report["mean_group_accuracy"] = float(np.mean(group_accs))
    report["pooled_accuracy"] = float(accuracy_score(all_true, all_pred))
    report["n_samples"] = int(len(y))
    report["n_groups"] = len(unique_groups)
    report["classes"] = sorted(set(y.tolist()))
    return report
