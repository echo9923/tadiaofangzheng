from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd

from tower_sim.io_utils import dataset_paths, split_scenario_ids
from tower_sim.windowing import make_windows


def test_dataset_paths_use_formal_dataset_layout() -> None:
    root = Path("dataset_root")
    paths = dataset_paths(root)

    assert paths.root == root
    assert paths.tables == root / "tables"
    assert paths.windows == root / "windows"
    assert paths.quality == root / "quality"
    assert paths.plots == root / "quality" / "plots"
    assert paths.config_used == root / "config_used.yaml"
    assert paths.metadata == root / "metadata.json"
    assert paths.data_dictionary == root / "data_dictionary.md"


def test_split_scenario_ids_can_reserve_generalization_split() -> None:
    assignments = split_scenario_ids(
        list(range(10)),
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        add_generalization_test=True,
    )

    assert "generalization" in set(assignments.values())
    assigned_ids_by_split = {
        split: {sid for sid, assigned_split in assignments.items() if assigned_split == split}
        for split in set(assignments.values())
    }
    for split_a, ids_a in assigned_ids_by_split.items():
        for split_b, ids_b in assigned_ids_by_split.items():
            if split_a != split_b:
                assert ids_a.isdisjoint(ids_b)


def test_make_windows_writes_generalization_npz_when_enabled() -> None:
    # Avoid relying on pytest's default temp root on locked-down Windows hosts.
    output_dir = Path("test_artifacts") / f"generalization_windows_{uuid.uuid4().hex}"
    output_dir.mkdir(parents=True)
    scenario_table = pd.DataFrame(
        [
            {"scenario_id": 0, "split": "generalization"},
        ]
    )
    crane_static = pd.DataFrame(
        [
            {
                "scenario_id": 0,
                "crane_id": crane_id,
                "tower_height": 30.0,
                "jib_length": 20.0,
                "max_radius": 20.0,
                "priority": 1,
            }
            for crane_id in [0, 1]
        ]
    )
    rows = []
    for step in range(5):
        for crane_id in [0, 1]:
            rows.append(
                {
                    "scenario_id": 0,
                    "timestamp": float(step),
                    "step": step,
                    "crane_id": crane_id,
                    "theta": 0.0,
                    "r": 10.0,
                    "h": 20.0,
                    "theta_dot": 0.0,
                    "r_dot": 0.0,
                    "h_dot": 0.0,
                    "theta_ddot": 0.0,
                    "r_ddot": 0.0,
                    "h_ddot": 0.0,
                    "load_weight": 0.0,
                    "command_theta": 0.0,
                    "command_r": 0.0,
                    "command_h": 0.0,
                    "brake_flag": 0,
                    "emergency_flag": 0,
                    "task_id": 0,
                    "task_stage": "transport_to_dropoff",
                    "theta_missing": 0,
                    "r_missing": 0,
                    "h_missing": 0,
                }
            )
    state_obs = pd.DataFrame(rows)
    state_true = state_obs.drop(columns=["theta_missing", "r_missing", "h_missing"]).copy()
    edge_rows = []
    label_rows = []
    for step in range(5):
        for i, j in [(0, 1), (1, 0)]:
            edge_rows.append(
                {
                    "scenario_id": 0,
                    "timestamp": float(step),
                    "step": step,
                    "crane_i": i,
                    "crane_j": j,
                    "d_arm_arm": 10.0,
                    "d_arm_hook_i_to_j": 10.0,
                    "d_arm_hook_j_to_i": 10.0,
                    "d_hook_hook": 10.0,
                    "delta_theta": 0.0,
                    "delta_theta_dot": 0.0,
                    "delta_r": 0.0,
                    "delta_h": 0.0,
                    "delta_tower_height": 0.0,
                    "base_distance": 10.0,
                    "overlap_ratio": 1.0,
                    "relative_approach_speed": 0.0,
                    "relative_approach_speed_arm_arm": 0.0,
                    "relative_approach_speed_arm_hook": 0.0,
                    "relative_approach_speed_hook_hook": 0.0,
                    "ttc_est_arm_arm": -1.0,
                    "ttc_est_arm_hook": -1.0,
                    "ttc_est_hook_hook": -1.0,
                    "same_height_risk_zone": 0,
                }
            )
            label_rows.append(
                {
                    "scenario_id": 0,
                    "timestamp": float(step),
                    "step": step,
                    "horizon_s": 2.0,
                    "crane_i": i,
                    "crane_j": j,
                    "future_min_d_arm_arm": 10.0,
                    "future_min_d_arm_hook_i_to_j": 10.0,
                    "future_min_d_arm_hook_j_to_i": 10.0,
                    "future_min_d_hook_hook": 10.0,
                    "risk_arm_arm": 0,
                    "risk_arm_hook_i_to_j": 0,
                    "risk_arm_hook_j_to_i": 0,
                    "risk_hook_hook": 0,
                    "ttc_label_arm_arm": -1.0,
                    "ttc_label_arm_hook": -1.0,
                    "ttc_label_hook_hook": -1.0,
                }
            )
    config = {
        "simulation": {"dt": 1.0},
        "windowing": {
            "input_window_s": 2.0,
            "prediction_horizon_s": 2.0,
            "stride_s": 1.0,
            "max_cranes": 2,
        },
        "split": {"add_generalization_test": True},
    }

    counts = make_windows(
        state_obs,
        state_true,
        crane_static,
        pd.DataFrame(edge_rows),
        pd.DataFrame(label_rows),
        scenario_table,
        config,
        output_dir,
    )

    assert counts["generalization"] > 0
    assert (output_dir / "generalization_windows.npz").exists()
    with np.load(output_dir / "generalization_windows.npz", allow_pickle=False) as data:
        assert set(data["scenario_ids"].tolist()) == {0}
    try:
        shutil.rmtree(output_dir)
    except PermissionError:
        pass
