from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from tower_sim.quality import enforce_quality_gates, generate_quality_report


def test_quality_gate_raises_on_required_nan_when_enabled() -> None:
    stats = {"has_nan_required": True, "future_leakage": False}
    config = {
        "quality_control": {
            "fail_on_nan_in_required_fields": True,
            "fail_on_future_leakage": True,
        }
    }

    try:
        enforce_quality_gates(stats, config)
    except ValueError as exc:
        assert "NaN" in str(exc)
    else:
        raise AssertionError("Expected quality gate to reject NaN")


def test_quality_gate_raises_on_future_leakage_when_enabled() -> None:
    stats = {"has_nan_required": False, "future_leakage": True}
    config = {
        "quality_control": {
            "fail_on_nan_in_required_fields": True,
            "fail_on_future_leakage": True,
        }
    }

    try:
        enforce_quality_gates(stats, config)
    except ValueError as exc:
        assert "Future leakage" in str(exc)
    else:
        raise AssertionError("Expected quality gate to reject future leakage")


def test_quality_gate_raises_on_risk_ratio_outside_target_range() -> None:
    stats = {
        "has_nan_required": False,
        "future_leakage": False,
        "risk_any_ratio": 0.75,
        "risk_ratio_in_target_range": False,
    }
    config = {
        "quality_control": {
            "target_risk_ratio_range": [0.02, 0.50],
            "fail_on_risk_ratio_out_of_range": True,
        }
    }

    try:
        enforce_quality_gates(stats, config)
    except ValueError as exc:
        assert "risk_any_ratio" in str(exc)
    else:
        raise AssertionError("Expected quality gate to reject risk ratio outside target range")


def test_quality_gate_can_warn_only_for_risk_ratio_outside_target_range() -> None:
    stats = {
        "has_nan_required": False,
        "future_leakage": False,
        "risk_any_ratio": 0.75,
        "risk_ratio_in_target_range": False,
        "split_disjoint": True,
    }
    config = {
        "quality_control": {
            "target_risk_ratio_range": [0.02, 0.50],
            "fail_on_risk_ratio_out_of_range": False,
        }
    }

    enforce_quality_gates(stats, config)


def test_quality_gate_raises_on_integrity_failures() -> None:
    config = {"quality_control": {"target_risk_ratio_range": [0.0, 1.0]}}
    failing_keys = [
        "r_out_of_bounds",
        "h_out_of_bounds",
        "speed_out_of_bounds",
        "acc_out_of_bounds",
        "edge_future_label_has_inf",
    ]

    for key in failing_keys:
        stats = {
            "has_nan_required": False,
            "future_leakage": False,
            "risk_any_ratio": 0.1,
            "risk_ratio_in_target_range": True,
            "split_disjoint": True,
            key: True,
        }
        try:
            enforce_quality_gates(stats, config)
        except ValueError as exc:
            assert key in str(exc)
        else:
            raise AssertionError(f"Expected quality gate to reject {key}")

    try:
        enforce_quality_gates(
            {
                "has_nan_required": False,
                "future_leakage": False,
                "risk_any_ratio": 0.1,
                "risk_ratio_in_target_range": True,
                "split_disjoint": False,
            },
            config,
        )
    except ValueError as exc:
        assert "split_disjoint" in str(exc)
    else:
        raise AssertionError("Expected quality gate to reject non-disjoint split")


def test_quality_report_checks_generalization_split_disjointness() -> None:
    output_dir = Path("test_artifacts") / f"quality_split_{uuid.uuid4().hex}"
    output_dir.mkdir(parents=True)
    scenario_table = pd.DataFrame(
        [
            {"scenario_id": "scenario_000001", "scenario_index": 1, "scenario_uid": "scenario_000001", "num_cranes": 2, "duration_s": 1.0, "split": "train"},
            {"scenario_id": "scenario_000001", "scenario_index": 1, "scenario_uid": "scenario_000001", "num_cranes": 2, "duration_s": 1.0, "split": "generalization"},
        ]
    )
    crane_static = pd.DataFrame()
    state_true = pd.DataFrame()
    edge_current = pd.DataFrame()
    edge_future_label = pd.DataFrame()

    try:
        generate_quality_report(
            output_dir,
            scenario_table,
            crane_static,
            pd.DataFrame(),
            state_true,
            state_true,
            edge_current,
            edge_future_label,
            {
                "simulation": {"dt": 1.0},
                "project": {"random_seed": 1},
                "quality_control": {
                    "target_risk_ratio_range": [0.0, 1.0],
                    "generate_summary_plots": False,
                },
            },
            geometry_table=pd.DataFrame(),
        )
    except ValueError as exc:
        assert "split_disjoint" in str(exc)
    else:
        raise AssertionError("Expected quality gate to reject overlap with generalization split")


def test_generate_quality_report_skips_plots_when_disabled() -> None:
    output_dir = Path("test_artifacts") / f"quality_no_plots_{uuid.uuid4().hex}"
    output_dir.mkdir(parents=True)
    scenario_table = pd.DataFrame(
        [
            {
                "scenario_id": 0,
                "scenario_uid": "scenario_000000",
                "num_cranes": 2,
                "duration_s": 1.0,
                "split": "train",
            }
        ]
    )

    stats = generate_quality_report(
        output_dir,
        scenario_table,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        {
            "simulation": {"dt": 1.0},
            "project": {"random_seed": 1},
            "quality_control": {
                "target_risk_ratio_range": [0.0, 1.0],
                "generate_summary_plots": False,
            },
        },
        geometry_table=pd.DataFrame(),
    )

    assert stats["split_disjoint"] is True
    assert not (output_dir / "plots").exists()
    assert "Plots disabled" in (output_dir / "quality_report.md").read_text(encoding="utf-8")


def test_quality_report_allows_configured_emergency_braking_acceleration() -> None:
    output_dir = Path("test_artifacts") / f"quality_emergency_{uuid.uuid4().hex}"
    output_dir.mkdir(parents=True)
    scenario_table = pd.DataFrame(
        [
            {
                "scenario_id": 0,
                "scenario_uid": "scenario_000000",
                "num_cranes": 2,
                "duration_s": 1.0,
                "split": "train",
            }
        ]
    )
    crane_static = pd.DataFrame(
        [
            {
                "scenario_id": 0,
                "scenario_uid": "scenario_000000",
                "crane_id": crane_id,
                "crane_uid": f"crane_0{crane_id}",
                "base_x": crane_id * 30.0,
                "base_y": 0.0,
                "min_radius": 2.0,
                "max_radius": 10.0,
                "tower_height": 30.0,
                "jib_length": 12.0,
                "max_theta_dot": 2.0,
                "max_r_dot": 2.0,
                "max_h_dot": 2.0,
                "max_theta_acc": 1.0,
                "max_r_acc": 1.0,
                "max_h_acc": 1.0,
                "load_capacity": 10000.0,
            }
            for crane_id in [0, 1]
        ]
    )
    state_rows = []
    geometry_rows = []
    for crane_id in [0, 1]:
        state_rows.append(
            {
                "scenario_id": 0,
                "scenario_uid": "scenario_000000",
                "step": 0,
                "timestamp": 0.0,
                "crane_id": crane_id,
                "crane_uid": f"crane_0{crane_id}",
                "theta": 0.0,
                "r": 5.0,
                "h": 10.0,
                "theta_dot": 0.0,
                "r_dot": 0.0,
                "h_dot": 0.0,
                "theta_ddot": 1.5 if crane_id == 0 else 0.0,
                "r_ddot": -1.5 if crane_id == 0 else 0.0,
                "h_ddot": 1.5 if crane_id == 0 else 0.0,
                "load_weight": 0.0,
                "command_theta": 0.0,
                "command_r": 0.0,
                "command_h": 0.0,
                "brake_flag": 1 if crane_id == 0 else 0,
                "emergency_flag": 1 if crane_id == 0 else 0,
                "task_id": 0,
                "task_uid": "task_00_0000",
                "task_stage": "transport_to_dropoff",
            }
        )
        geometry_rows.append(
            {
                "scenario_id": 0,
                "scenario_uid": "scenario_000000",
                "step": 0,
                "crane_id": crane_id,
                "crane_uid": f"crane_0{crane_id}",
                "root_x": crane_id * 30.0,
                "root_y": 0.0,
                "tip_x": crane_id * 30.0 + 5.0,
                "tip_y": 0.0,
                "hook_x": crane_id * 30.0 + 5.0,
                "hook_y": 0.0,
            }
        )
    edge_current = pd.DataFrame(
        [
            {
                "scenario_id": 0,
                "scenario_uid": "scenario_000000",
                "step": 0,
                "crane_i": 0,
                "crane_j": 1,
                "d_arm_arm": 20.0,
                "d_arm_hook_i_to_j": 20.0,
                "d_arm_hook_j_to_i": 20.0,
                "d_hook_hook": 20.0,
            }
        ]
    )
    edge_future_label = pd.DataFrame(
        [
            {
                "scenario_id": 0,
                "scenario_uid": "scenario_000000",
                "step": 0,
                "horizon_s": 1.0,
                "crane_i": 0,
                "crane_j": 1,
                "future_min_d_arm_arm": 20.0,
                "future_min_d_arm_hook_i_to_j": 20.0,
                "future_min_d_arm_hook_j_to_i": 20.0,
                "future_min_d_hook_hook": 20.0,
                "risk_arm_arm": 0,
                "risk_arm_hook_i_to_j": 0,
                "risk_arm_hook_j_to_i": 0,
                "risk_hook_hook": 0,
            }
        ]
    )

    stats = generate_quality_report(
        output_dir,
        scenario_table,
        crane_static,
        pd.DataFrame([{"scenario_id": 0, "task_id": 0}]),
        pd.DataFrame(state_rows),
        pd.DataFrame(state_rows),
        edge_current,
        edge_future_label,
        {
            "simulation": {"dt": 1.0},
            "project": {"random_seed": 1},
            "dynamics": {"emergency_brake_scale": 2.0},
            "quality_control": {"target_risk_ratio_range": [0.0, 1.0]},
        },
        geometry_table=pd.DataFrame(geometry_rows),
    )

    assert stats["acc_out_of_bounds"] is False


def test_quality_report_allows_configured_normal_braking_acceleration() -> None:
    output_dir = Path("test_artifacts") / f"quality_normal_brake_{uuid.uuid4().hex}"
    output_dir.mkdir(parents=True)
    scenario_table = pd.DataFrame(
        [
            {
                "scenario_id": 0,
                "scenario_uid": "scenario_000000",
                "num_cranes": 2,
                "duration_s": 1.0,
                "split": "train",
            }
        ]
    )
    crane_static = pd.DataFrame(
        [
            {
                "scenario_id": 0,
                "scenario_uid": "scenario_000000",
                "crane_id": crane_id,
                "crane_uid": f"crane_0{crane_id}",
                "base_x": crane_id * 30.0,
                "base_y": 0.0,
                "min_radius": 2.0,
                "max_radius": 10.0,
                "tower_height": 30.0,
                "jib_length": 12.0,
                "max_theta_dot": 2.0,
                "max_r_dot": 2.0,
                "max_h_dot": 2.0,
                "max_theta_acc": 1.0,
                "max_r_acc": 1.0,
                "max_h_acc": 1.0,
                "load_capacity": 10000.0,
            }
            for crane_id in [0, 1]
        ]
    )
    state_rows = []
    geometry_rows = []
    for crane_id in [0, 1]:
        state_rows.append(
            {
                "scenario_id": 0,
                "scenario_uid": "scenario_000000",
                "step": 0,
                "timestamp": 0.0,
                "crane_id": crane_id,
                "crane_uid": f"crane_0{crane_id}",
                "theta": 0.0,
                "r": 5.0,
                "h": 10.0,
                "theta_dot": 0.0,
                "r_dot": 0.0,
                "h_dot": 0.0,
                "theta_ddot": 1.5 if crane_id == 0 else 0.0,
                "r_ddot": -1.5 if crane_id == 0 else 0.0,
                "h_ddot": 1.5 if crane_id == 0 else 0.0,
                "load_weight": 0.0,
                "command_theta": 0.0,
                "command_r": 0.0,
                "command_h": 0.0,
                "brake_flag": 1 if crane_id == 0 else 0,
                "emergency_flag": 0,
                "task_id": 0,
                "task_uid": "task_00_0000",
                "task_stage": "transport_to_dropoff",
            }
        )
        geometry_rows.append(
            {
                "scenario_id": 0,
                "scenario_uid": "scenario_000000",
                "step": 0,
                "crane_id": crane_id,
                "crane_uid": f"crane_0{crane_id}",
                "root_x": crane_id * 30.0,
                "root_y": 0.0,
                "tip_x": crane_id * 30.0 + 5.0,
                "tip_y": 0.0,
                "hook_x": crane_id * 30.0 + 5.0,
                "hook_y": 0.0,
            }
        )
    edge_current = pd.DataFrame(
        [
            {
                "scenario_id": 0,
                "scenario_uid": "scenario_000000",
                "step": 0,
                "crane_i": 0,
                "crane_j": 1,
                "d_arm_arm": 20.0,
                "d_arm_hook_i_to_j": 20.0,
                "d_arm_hook_j_to_i": 20.0,
                "d_hook_hook": 20.0,
            }
        ]
    )
    edge_future_label = pd.DataFrame(
        [
            {
                "scenario_id": 0,
                "scenario_uid": "scenario_000000",
                "step": 0,
                "horizon_s": 1.0,
                "crane_i": 0,
                "crane_j": 1,
                "future_min_d_arm_arm": 20.0,
                "future_min_d_arm_hook_i_to_j": 20.0,
                "future_min_d_arm_hook_j_to_i": 20.0,
                "future_min_d_hook_hook": 20.0,
                "risk_arm_arm": 0,
                "risk_arm_hook_i_to_j": 0,
                "risk_arm_hook_j_to_i": 0,
                "risk_hook_hook": 0,
            }
        ]
    )

    stats = generate_quality_report(
        output_dir,
        scenario_table,
        crane_static,
        pd.DataFrame([{"scenario_id": 0, "task_id": 0}]),
        pd.DataFrame(state_rows),
        pd.DataFrame(state_rows),
        edge_current,
        edge_future_label,
        {
            "simulation": {"dt": 1.0},
            "project": {"random_seed": 1},
            "dynamics": {"normal_brake_scale": 2.0},
            "quality_control": {"target_risk_ratio_range": [0.0, 1.0]},
        },
        geometry_table=pd.DataFrame(geometry_rows),
    )

    assert stats["acc_out_of_bounds"] is False


def test_generate_quality_report_writes_required_csv_artifacts() -> None:
    output_dir = Path("test_artifacts") / f"quality_{uuid.uuid4().hex}"
    output_dir.mkdir(parents=True)
    scenario_table = pd.DataFrame(
        [
            {
                "scenario_id": 0,
                "scenario_uid": "scenario_000000",
                "num_cranes": 2,
                "duration_s": 3.0,
                "split": "train",
            }
        ]
    )
    crane_static = pd.DataFrame(
        [
            {
                "scenario_id": 0,
                "scenario_uid": "scenario_000000",
                "crane_id": crane_id,
                "crane_uid": f"crane_0{crane_id}",
                "base_x": crane_id * 30.0,
                "base_y": 0.0,
                "min_radius": 2.0,
                "max_radius": 10.0,
                "tower_height": 30.0,
                "jib_length": 12.0,
                "max_theta_dot": 1.0,
                "max_r_dot": 1.0,
                "max_h_dot": 1.0,
                "max_theta_acc": 1.0,
                "max_r_acc": 1.0,
                "max_h_acc": 1.0,
            }
            for crane_id in [0, 1]
        ]
    )
    state_rows = []
    geometry_rows = []
    for step in range(3):
        for crane_id in [0, 1]:
            state_rows.append(
                {
                    "scenario_id": 0,
                    "scenario_uid": "scenario_000000",
                    "step": step,
                    "timestamp": float(step),
                    "crane_id": crane_id,
                    "crane_uid": f"crane_0{crane_id}",
                    "theta": 0.0,
                    "r": 5.0,
                    "h": 10.0,
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
                    "task_uid": "task_00_0000",
                    "task_stage": "transport_to_dropoff",
                }
            )
            geometry_rows.append(
                {
                    "scenario_id": 0,
                    "scenario_uid": "scenario_000000",
                    "step": step,
                    "crane_id": crane_id,
                    "crane_uid": f"crane_0{crane_id}",
                    "root_x": crane_id * 30.0,
                    "root_y": 0.0,
                    "tip_x": crane_id * 30.0 + 5.0,
                    "tip_y": 0.0,
                    "hook_x": crane_id * 30.0 + 5.0,
                    "hook_y": 0.0,
                }
            )
    state_true = pd.DataFrame(state_rows)
    state_obs = state_true.copy()
    edge_current = pd.DataFrame(
        [
            {
                "scenario_id": 0,
                "scenario_uid": "scenario_000000",
                "step": 0,
                "crane_i": 0,
                "crane_j": 1,
                "d_arm_arm": 20.0,
                "d_arm_hook_i_to_j": 20.0,
                "d_arm_hook_j_to_i": 20.0,
                "d_hook_hook": 20.0,
            }
        ]
    )
    edge_future_label = pd.DataFrame(
        [
            {
                "scenario_id": 0,
                "scenario_uid": "scenario_000000",
                "step": 0,
                "horizon_s": 2.0,
                "crane_i": 0,
                "crane_j": 1,
                "future_min_d_arm_arm": 20.0,
                "future_min_d_arm_hook_i_to_j": 20.0,
                "future_min_d_arm_hook_j_to_i": 20.0,
                "future_min_d_hook_hook": 20.0,
                "risk_arm_arm": 0,
                "risk_arm_hook_i_to_j": 0,
                "risk_arm_hook_j_to_i": 0,
                "risk_hook_hook": 0,
            }
        ]
    )
    config = {
        "simulation": {"dt": 1.0},
        "project": {"random_seed": 1},
        "quality_control": {"target_risk_ratio_range": [0.0, 1.0]},
    }

    stats = generate_quality_report(
        output_dir,
        scenario_table,
        crane_static,
        pd.DataFrame([{"scenario_id": 0, "task_id": 0}]),
        state_true,
        state_obs,
        edge_current,
        edge_future_label,
        config,
        geometry_table=pd.DataFrame(geometry_rows),
    )

    assert (output_dir / "quality_report.md").exists()
    assert (output_dir / "risk_ratio_by_scenario.csv").exists()
    assert (output_dir / "feature_summary.csv").exists()
    risk_by_scenario = pd.read_csv(output_dir / "risk_ratio_by_scenario.csv")
    feature_summary = pd.read_csv(output_dir / "feature_summary.csv")
    assert risk_by_scenario.loc[0, "scenario_uid"] == "scenario_000000"
    assert "risk_any_ratio" in risk_by_scenario.columns
    assert {"source", "feature", "mean"}.issubset(feature_summary.columns)
    assert stats["risk_ratio_in_target_range"] is True
    assert stats["edge_future_label_has_inf"] is False
