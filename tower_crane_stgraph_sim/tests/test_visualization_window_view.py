from pathlib import Path

import numpy as np

from tower_sim.visualization.window_view import (
    assert_window_inputs_safe,
    bundle_from_arrays,
    filter_window_samples,
    get_window_sample,
    load_window_bundle,
    summarize_window_bundle,
)


def test_window_bundle_summarizes_debug_small_npz() -> None:
    path = Path(__file__).resolve().parents[1] / "outputs" / "debug_small" / "train_windows.npz"
    bundle = load_window_bundle(path, split="train")

    summary = summarize_window_bundle(bundle)

    assert summary["num_samples"] > 0
    assert "node_features" in summary["shapes"]
    assert summary["leakage_offenders"] == []
    assert_window_inputs_safe(bundle)

    sample = get_window_sample(bundle, 0)
    assert sample.split == "train"
    assert sample.shapes["node_features"] == tuple(bundle.arrays["node_features"][0].shape)
    assert "node_mask" in sample.masks


def test_window_filter_can_find_positive_risk_samples_and_risk_types() -> None:
    arrays = {
        "node_features": np.zeros((2, 2, 2, 2), dtype=np.float32),
        "edge_features": np.zeros((2, 2, 2, 2, 1), dtype=np.float32),
        "node_mask": np.ones((2, 2), dtype=bool),
        "edge_mask": np.ones((2, 2, 2), dtype=bool),
        "time_mask": np.ones((2, 2), dtype=bool),
        "y_traj": np.zeros((2, 1, 2, 4), dtype=np.float32),
        "y_risk": np.zeros((2, 2, 2, 4), dtype=np.float32),
        "y_min_distance": np.zeros((2, 2, 2, 4), dtype=np.float32),
        "node_feature_names": np.array(["r", "brake_flag"]),
        "edge_feature_names": np.array(["d_arm_arm"]),
        "y_risk_feature_names": np.array(["risk_arm_arm", "risk_arm_hook_i_to_j", "risk_arm_hook_j_to_i", "risk_hook_hook"]),
        "scenario_ids": np.array(["scenario_000001", "scenario_000002"]),
        "scenario_indices": np.array([1, 2]),
        "window_start_steps": np.array([0, 10]),
    }
    arrays["y_risk"][1, 0, 1, 0] = 1
    arrays["node_features"][1, 0, 0, 1] = 1
    bundle = bundle_from_arrays(arrays, split="train")

    assert filter_window_samples(bundle, positive_only=True) == [1]
    assert filter_window_samples(bundle, risk_type="arm_arm") == [1]
    assert filter_window_samples(bundle, contains_brake=True) == [1]
    assert filter_window_samples(bundle, scenario_id="scenario_000001") == [0]


def test_window_input_leakage_is_rejected() -> None:
    bundle = bundle_from_arrays(
        {
            "node_features": np.zeros((1, 1, 1, 1)),
            "edge_features": np.zeros((1, 1, 1, 1, 1)),
            "node_feature_names": np.array(["future_min_d_arm_arm"]),
            "edge_feature_names": np.array(["d_arm_arm"]),
        }
    )

    try:
        assert_window_inputs_safe(bundle)
    except ValueError as exc:
        assert "future_min_d" in str(exc)
    else:
        raise AssertionError("Expected future-label leakage to be rejected")
