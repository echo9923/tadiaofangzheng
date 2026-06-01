import math

import pandas as pd

import tower_sim.labels as labels_module
from tower_sim.labels import compute_future_labels
from tower_sim.geometry import (
    point_point_distance,
    reconstruct_geometry_from_rows,
    segment_point_distance,
    segment_segment_distance,
)


def test_future_labels_detect_manual_two_crane_arm_risk() -> None:
    crane_static = pd.DataFrame(
        [
            {
                "scenario_id": 0,
                "crane_id": 0,
                "base_x": 0.0,
                "base_y": 0.0,
                "base_z": 0.0,
                "tower_height": 30.0,
                "jib_length": 20.0,
                "min_radius": 2.0,
                "max_radius": 20.0,
                "safety_radius_arm": 2.0,
                "safety_radius_hook": 2.0,
                "max_theta_dot": 1.0,
                "max_r_dot": 1.0,
                "max_h_dot": 1.0,
                "max_theta_acc": 1.0,
                "max_r_acc": 1.0,
                "max_h_acc": 1.0,
                "response_tau": 1.0,
                "load_capacity": 1000.0,
                "priority": 1,
            },
            {
                "scenario_id": 0,
                "crane_id": 1,
                "base_x": 10.0,
                "base_y": -10.0,
                "base_z": 0.0,
                "tower_height": 30.0,
                "jib_length": 20.0,
                "min_radius": 2.0,
                "max_radius": 20.0,
                "safety_radius_arm": 2.0,
                "safety_radius_hook": 2.0,
                "max_theta_dot": 1.0,
                "max_r_dot": 1.0,
                "max_h_dot": 1.0,
                "max_theta_acc": 1.0,
                "max_r_acc": 1.0,
                "max_h_acc": 1.0,
                "response_tau": 1.0,
                "load_capacity": 1000.0,
                "priority": 0,
            },
        ]
    )
    state_true = pd.DataFrame(
        [
            {
                "scenario_id": 0,
                "timestamp": float(step),
                "step": step,
                "crane_id": crane_id,
                "theta": theta,
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
            }
            for step in range(3)
            for crane_id, theta in [(0, 0.0), (1, 1.5707963267948966)]
        ]
    )

    labels = compute_future_labels(
        state_true,
        crane_static,
        dt=1.0,
        horizons_s=[2.0],
        thresholds={
            "d_safe_arm_arm_m": 1.0,
            "d_safe_arm_hook_m": 1.0,
            "d_safe_hook_hook_m": 1.0,
        },
    )

    first = labels[(labels["step"] == 0) & (labels["crane_i_index"] == 0) & (labels["crane_j_index"] == 1)].iloc[0]
    assert first["future_min_d_arm_arm"] == 0.0
    assert first["risk_arm_arm"] == 1
    assert first["ttc_label_arm_arm"] == 1.0


def test_future_labels_set_ttc_minus_one_when_no_risk() -> None:
    crane_static = pd.DataFrame(
        [
            {
                "scenario_id": 1,
                "crane_id": 0,
                "base_x": 0.0,
                "base_y": 0.0,
                "base_z": 0.0,
                "tower_height": 30.0,
                "jib_length": 10.0,
                "min_radius": 2.0,
                "max_radius": 10.0,
                "safety_radius_arm": 2.0,
                "safety_radius_hook": 2.0,
                "max_theta_dot": 1.0,
                "max_r_dot": 1.0,
                "max_h_dot": 1.0,
                "max_theta_acc": 1.0,
                "max_r_acc": 1.0,
                "max_h_acc": 1.0,
                "response_tau": 1.0,
                "load_capacity": 1000.0,
                "priority": 1,
            },
            {
                "scenario_id": 1,
                "crane_id": 1,
                "base_x": 100.0,
                "base_y": 100.0,
                "base_z": 0.0,
                "tower_height": 30.0,
                "jib_length": 10.0,
                "min_radius": 2.0,
                "max_radius": 10.0,
                "safety_radius_arm": 2.0,
                "safety_radius_hook": 2.0,
                "max_theta_dot": 1.0,
                "max_r_dot": 1.0,
                "max_h_dot": 1.0,
                "max_theta_acc": 1.0,
                "max_r_acc": 1.0,
                "max_h_acc": 1.0,
                "response_tau": 1.0,
                "load_capacity": 1000.0,
                "priority": 0,
            },
        ]
    )
    state_true = pd.DataFrame(
        [
            {
                "scenario_id": 1,
                "timestamp": float(step),
                "step": step,
                "crane_id": crane_id,
                "theta": 0.0,
                "r": 5.0,
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
            }
            for step in range(3)
            for crane_id in [0, 1]
        ]
    )

    labels = compute_future_labels(
        state_true,
        crane_static,
        dt=1.0,
        horizons_s=[2.0],
        thresholds={
            "d_safe_arm_arm_m": 1.0,
            "d_safe_arm_hook_m": 1.0,
            "d_safe_hook_hook_m": 1.0,
        },
    )

    first = labels[(labels["step"] == 0) & (labels["crane_i_index"] == 0) & (labels["crane_j_index"] == 1)].iloc[0]
    assert first["risk_arm_arm"] == 0
    assert first["risk_arm_hook_i_to_j"] == 0
    assert first["risk_hook_hook"] == 0
    assert first["ttc_label_arm_arm"] == -1.0


def _static_rows_for_two_arm_cranes(scenario_id: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario_id": scenario_id,
                "crane_id": 0,
                "base_x": 0.0,
                "base_y": 0.0,
                "base_z": 0.0,
                "tower_height": 30.0,
                "jib_length": 20.0,
                "min_radius": 2.0,
                "max_radius": 20.0,
                "safety_radius_arm": 2.0,
                "safety_radius_hook": 2.0,
                "max_theta_dot": 1.0,
                "max_r_dot": 1.0,
                "max_h_dot": 1.0,
                "max_theta_acc": 1.0,
                "max_r_acc": 1.0,
                "max_h_acc": 1.0,
                "response_tau": 1.0,
                "load_capacity": 1000.0,
                "priority": 1,
            },
            {
                "scenario_id": scenario_id,
                "crane_id": 1,
                "base_x": 10.0,
                "base_y": -10.0,
                "base_z": 0.0,
                "tower_height": 30.0,
                "jib_length": 20.0,
                "min_radius": 2.0,
                "max_radius": 20.0,
                "safety_radius_arm": 2.0,
                "safety_radius_hook": 2.0,
                "max_theta_dot": 1.0,
                "max_r_dot": 1.0,
                "max_h_dot": 1.0,
                "max_theta_acc": 1.0,
                "max_r_acc": 1.0,
                "max_h_acc": 1.0,
                "response_tau": 1.0,
                "load_capacity": 1000.0,
                "priority": 0,
            },
        ]
    )


def _state_rows_for_label_steps(
    scenario_id: int,
    crane_1_theta_by_step: dict[int, float],
) -> pd.DataFrame:
    rows = []
    for step in sorted(crane_1_theta_by_step):
        for crane_id, theta in [(0, 0.0), (1, crane_1_theta_by_step[step])]:
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "timestamp": float(step),
                    "step": step,
                    "crane_id": crane_id,
                    "theta": theta,
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
                }
            )
    return pd.DataFrame(rows)


def test_future_labels_exclude_current_step_from_future_min_distance() -> None:
    crane_static = _static_rows_for_two_arm_cranes(scenario_id=2)
    state_true = _state_rows_for_label_steps(
        2,
        {
            0: math.pi / 2.0,
            1: 0.0,
            2: 0.0,
        },
    )

    labels = compute_future_labels(
        state_true,
        crane_static,
        dt=1.0,
        horizons_s=[2.0],
        thresholds={
            "d_safe_arm_arm_m": 1.0,
            "d_safe_arm_hook_m": 1.0,
            "d_safe_hook_hook_m": 1.0,
        },
    )

    first = labels[(labels["step"] == 0) & (labels["crane_i_index"] == 0) & (labels["crane_j_index"] == 1)].iloc[0]
    assert first["future_min_d_arm_arm"] > 1.0
    assert first["risk_arm_arm"] == 0
    assert first["ttc_label_arm_arm"] == -1.0


def test_future_labels_report_positive_ttc_when_future_step_enters_risk() -> None:
    crane_static = _static_rows_for_two_arm_cranes(scenario_id=3)
    state_true = _state_rows_for_label_steps(
        3,
        {
            0: 0.0,
            1: 0.0,
            2: math.pi / 2.0,
            3: math.pi / 2.0,
        },
    )

    labels = compute_future_labels(
        state_true,
        crane_static,
        dt=1.0,
        horizons_s=[3.0],
        thresholds={
            "d_safe_arm_arm_m": 1.0,
            "d_safe_arm_hook_m": 1.0,
            "d_safe_hook_hook_m": 1.0,
        },
    )

    first = labels[(labels["step"] == 0) & (labels["crane_i_index"] == 0) & (labels["crane_j_index"] == 1)].iloc[0]
    assert first["future_min_d_arm_arm"] == 0.0
    assert first["risk_arm_arm"] == 1
    assert first["ttc_label_arm_arm"] == 2.0


def test_future_labels_skip_steps_without_complete_future_horizon() -> None:
    crane_static = _static_rows_for_two_arm_cranes(scenario_id=4)
    state_true = _state_rows_for_label_steps(
        4,
        {
            0: 0.0,
            1: 0.0,
            2: 0.0,
        },
    )

    labels = compute_future_labels(
        state_true,
        crane_static,
        dt=1.0,
        horizons_s=[2.0],
        thresholds={
            "d_safe_arm_arm_m": 1.0,
            "d_safe_arm_hook_m": 1.0,
            "d_safe_hook_hook_m": 1.0,
        },
    )

    assert set(labels["step"].unique()) == {0}
    distance_columns = [
        "future_min_d_arm_arm",
        "future_min_d_arm_hook_i_to_j",
        "future_min_d_arm_hook_j_to_i",
        "future_min_d_hook_hook",
    ]
    assert not labels[distance_columns].isin([math.inf, -math.inf]).any().any()


def _reference_future_labels(
    state_true: pd.DataFrame,
    crane_static: pd.DataFrame,
    dt: float,
    horizons_s: list[float],
    thresholds: dict[str, float],
) -> pd.DataFrame:
    rows = []
    states_by_step = {
        step: {int(row["crane_index"]): row.to_dict() for _, row in group.iterrows()}
        for step, group in state_true.groupby("step")
    }
    static_rows = {int(row["crane_index"]): row.to_dict() for _, row in crane_static.iterrows()}
    sorted_steps = sorted(states_by_step)
    for step in sorted_steps:
        for horizon_s in horizons_s:
            horizon_steps = max(1, int(round(horizon_s / dt)))
            future_steps = [future for future in sorted_steps if step < future <= step + horizon_steps]
            if len(future_steps) < horizon_steps:
                continue
            for crane_i in sorted(static_rows):
                for crane_j in sorted(static_rows):
                    if crane_i == crane_j:
                        continue
                    distances = []
                    for future in future_steps:
                        geom_i = reconstruct_geometry_from_rows(static_rows[crane_i], states_by_step[future][crane_i])
                        geom_j = reconstruct_geometry_from_rows(static_rows[crane_j], states_by_step[future][crane_j])
                        distances.append(
                            (
                                segment_segment_distance(geom_i.root, geom_i.tip, geom_j.root, geom_j.tip),
                                segment_point_distance(geom_i.root, geom_i.tip, geom_j.hook),
                                segment_point_distance(geom_j.root, geom_j.tip, geom_i.hook),
                                point_point_distance(geom_i.hook, geom_j.hook),
                            )
                        )
                    min_distances = tuple(min(values[index] for values in distances) for index in range(4))

                    def first_ttc(index: int, threshold: float) -> float:
                        for offset, values in enumerate(distances, start=1):
                            if values[index] < threshold:
                                return offset * dt
                        return -1.0

                    arm_hook_ttc = -1.0
                    for offset, values in enumerate(distances, start=1):
                        if min(values[1], values[2]) < thresholds["d_safe_arm_hook_m"]:
                            arm_hook_ttc = offset * dt
                            break
                    rows.append(
                        {
                            "scenario_id": "scenario_000000",
                            "scenario_uid": "scenario_000000",
                            "scenario_index": 0,
                            "timestamp": float(step),
                            "step": step,
                            "horizon_s": horizon_s,
                            "crane_i": f"crane_{crane_i:02d}",
                            "crane_j": f"crane_{crane_j:02d}",
                            "crane_i_uid": f"crane_{crane_i:02d}",
                            "crane_j_uid": f"crane_{crane_j:02d}",
                            "crane_i_index": crane_i,
                            "crane_j_index": crane_j,
                            "future_min_d_arm_arm": min_distances[0],
                            "future_min_d_arm_hook_i_to_j": min_distances[1],
                            "future_min_d_arm_hook_j_to_i": min_distances[2],
                            "future_min_d_hook_hook": min_distances[3],
                            "risk_arm_arm": int(min_distances[0] < thresholds["d_safe_arm_arm_m"]),
                            "risk_arm_hook_i_to_j": int(min_distances[1] < thresholds["d_safe_arm_hook_m"]),
                            "risk_arm_hook_j_to_i": int(min_distances[2] < thresholds["d_safe_arm_hook_m"]),
                            "risk_hook_hook": int(min_distances[3] < thresholds["d_safe_hook_hook_m"]),
                            "ttc_label_arm_arm": first_ttc(0, thresholds["d_safe_arm_arm_m"]),
                            "ttc_label_arm_hook": arm_hook_ttc,
                            "ttc_label_hook_hook": first_ttc(3, thresholds["d_safe_hook_hook_m"]),
                        }
                    )
    return pd.DataFrame(rows)


def test_future_labels_match_reference_for_multiple_cranes_and_horizons() -> None:
    crane_static = pd.DataFrame(
        [
            {
                "scenario_id": "scenario_000000",
                "scenario_index": 0,
                "crane_id": f"crane_{crane_id:02d}",
                "crane_index": crane_id,
                "base_x": float(crane_id * 12.0),
                "base_y": 0.0,
                "base_z": 0.0,
                "tower_height": 30.0,
                "jib_length": 20.0,
                "min_radius": 2.0,
                "max_radius": 20.0,
            }
            for crane_id in [0, 1, 2]
        ]
    )
    rows = []
    for step in range(5):
        for crane_id in [0, 1, 2]:
            rows.append(
                {
                    "scenario_id": "scenario_000000",
                    "scenario_index": 0,
                    "timestamp": float(step),
                    "step": step,
                    "crane_id": f"crane_{crane_id:02d}",
                    "crane_index": crane_id,
                    "theta": (step * 0.2) + crane_id * 0.7,
                    "r": 8.0 + crane_id,
                    "h": 12.0 + step,
                }
            )
    state_true = pd.DataFrame(rows)
    thresholds = {
        "d_safe_arm_arm_m": 4.0,
        "d_safe_arm_hook_m": 4.0,
        "d_safe_hook_hook_m": 4.0,
    }

    actual = compute_future_labels(state_true, crane_static, dt=1.0, horizons_s=[1.0, 2.0], thresholds=thresholds)
    expected = _reference_future_labels(state_true, crane_static, dt=1.0, horizons_s=[1.0, 2.0], thresholds=thresholds)

    sort_cols = ["step", "horizon_s", "crane_i_index", "crane_j_index"]
    actual = actual.sort_values(sort_cols).reset_index(drop=True)
    expected = expected.sort_values(sort_cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[expected.columns], expected, check_dtype=False, atol=1e-9, rtol=1e-9)


def test_future_labels_geometry_reconstruction_scales_with_state_rows(monkeypatch) -> None:
    crane_static = _static_rows_for_two_arm_cranes(scenario_id=5)
    state_true = _state_rows_for_label_steps(
        5,
        {step: 0.1 * step for step in range(8)},
    )
    calls = {"count": 0}

    def counted_reconstruct(static_row, state_row):
        calls["count"] += 1
        return reconstruct_geometry_from_rows(static_row, state_row)

    monkeypatch.setattr(labels_module, "reconstruct_geometry_from_rows", counted_reconstruct)

    compute_future_labels(
        state_true,
        crane_static,
        dt=1.0,
        horizons_s=[1.0, 2.0, 3.0],
        thresholds={
            "d_safe_arm_arm_m": 1.0,
            "d_safe_arm_hook_m": 1.0,
            "d_safe_hook_hook_m": 1.0,
        },
    )

    assert calls["count"] <= len(state_true)
