import numpy as np

from tower_sim.windowing import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    assert_no_future_leakage,
    validate_split_disjoint,
)


def test_feature_name_lists_exclude_future_label_columns() -> None:
    bad_tokens = ("future_min_d", "ttc_label")
    bad_prefixes = ("risk_arm_arm", "risk_arm_hook", "risk_hook_hook", "future_risk", "label_risk")

    assert not any(token in name for name in NODE_FEATURE_NAMES for token in bad_tokens)
    assert not any(token in name for name in EDGE_FEATURE_NAMES for token in bad_tokens)
    assert not any(name.startswith(prefix) for name in NODE_FEATURE_NAMES for prefix in bad_prefixes)
    assert not any(name.startswith(prefix) for name in EDGE_FEATURE_NAMES for prefix in bad_prefixes)
    assert_no_future_leakage(NODE_FEATURE_NAMES, EDGE_FEATURE_NAMES)


def test_split_scenario_ids_must_be_disjoint() -> None:
    train_ids = np.array([0, 1, 2])
    val_ids = np.array([3])
    test_ids = np.array([4, 5])

    validate_split_disjoint(train_ids, val_ids, test_ids)


def test_split_scenario_ids_overlap_is_rejected() -> None:
    train_ids = np.array([0, 1, 2])
    val_ids = np.array([2, 3])
    test_ids = np.array([4])

    try:
        validate_split_disjoint(train_ids, val_ids, test_ids)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("Expected overlapping scenario ids to be rejected")
