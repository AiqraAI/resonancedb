import numpy as np
import pytest

from resonancedb.benchmark import leave_one_group_out


def make_separable_dataset():
    """Two well-separated classes recorded on two 'devices'.

    Device B's features carry a small offset, as a second microphone would.
    """
    rng = np.random.default_rng(42)
    X, y, groups = [], [], []
    for device, offset in [("deviceA", 0.0), ("deviceB", 0.3)]:
        for label, center in [("glass", 10.0), ("wood", 2.0)]:
            for _ in range(15):
                X.append([center + offset + rng.normal(0, 0.2),
                          center * 2 + rng.normal(0, 0.2)])
                y.append(label)
                groups.append(device)
    return np.array(X), np.array(y), groups


def test_leave_one_group_out_separable():
    X, y, groups = make_separable_dataset()
    report = leave_one_group_out(X, y, groups)

    assert report["n_groups"] == 2
    assert report["n_samples"] == 60
    assert set(report["groups"]) == {"deviceA", "deviceB"}
    for stats in report["groups"].values():
        assert stats["n"] == 30
        assert stats["accuracy"] == 1.0
        assert stats["unseen_classes"] == []
    assert report["mean_group_accuracy"] == 1.0
    assert report["pooled_accuracy"] == 1.0


def test_single_group_rejected():
    X = np.zeros((4, 2))
    y = np.array(["a", "a", "b", "b"])
    with pytest.raises(ValueError, match="at least 2 groups"):
        leave_one_group_out(X, y, ["only", "only", "only", "only"])


def test_unseen_class_reported():
    """A class that exists only in the held-out group must be surfaced."""
    rng = np.random.default_rng(0)
    X, y, groups = [], [], []
    for _ in range(10):
        X.append([1.0 + rng.normal(0, 0.1)]); y.append("wood"); groups.append("devA")
        X.append([1.0 + rng.normal(0, 0.1)]); y.append("wood"); groups.append("devB")
    # metal exists ONLY on devB
    for _ in range(5):
        X.append([5.0 + rng.normal(0, 0.1)]); y.append("metal"); groups.append("devB")

    report = leave_one_group_out(np.array(X), np.array(y), groups)
    assert report["groups"]["devB"]["unseen_classes"] == ["metal"]
    assert report["groups"]["devA"]["unseen_classes"] == []
