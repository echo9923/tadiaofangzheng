import pandas as pd

from tower_sim.labels import compute_future_labels


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

    first = labels[(labels["step"] == 0) & (labels["crane_i"] == 0) & (labels["crane_j"] == 1)].iloc[0]
    assert first["future_min_d_arm_arm"] == 0.0
    assert first["risk_arm_arm"] == 1
    assert first["ttc_label_arm_arm"] == 0.0


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

    first = labels[(labels["step"] == 0) & (labels["crane_i"] == 0) & (labels["crane_j"] == 1)].iloc[0]
    assert first["risk_arm_arm"] == 0
    assert first["risk_arm_hook_i_to_j"] == 0
    assert first["risk_hook_hook"] == 0
    assert first["ttc_label_arm_arm"] == -1.0
