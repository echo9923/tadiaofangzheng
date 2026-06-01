from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tower_sim.config import load_config
from tower_sim.dataclasses import CraneStatic
from tower_sim.layout import has_radius_overlap
from tower_sim.simulation import _configured_save_formats, _generate_validated_cranes, _num_cranes_for_scene, simulate_dataset
from tower_sim.tasks import generate_tasks


def _crane(crane_id: int) -> CraneStatic:
    return CraneStatic(
        scenario_id=0,
        crane_id=crane_id,
        base_x=0.0 if crane_id == 0 else 40.0,
        base_y=0.0,
        base_z=0.0,
        tower_height=100.0,
        jib_length=50.0,
        min_radius=4.0,
        max_radius=45.0,
        safety_radius_arm=3.0,
        safety_radius_hook=3.0,
        max_theta_dot=0.2,
        max_r_dot=1.0,
        max_h_dot=1.0,
        max_theta_acc=0.1,
        max_r_acc=0.5,
        max_h_acc=0.5,
        response_tau=1.0,
        load_capacity=10000.0,
        priority=2 - crane_id,
    )


def test_two_crane_crossing_respects_minimum_three_cranes_when_configured() -> None:
    rng = np.random.default_rng(7)

    sampled = [_num_cranes_for_scene("two_crane_crossing", 3, 4, rng) for _ in range(20)]

    assert min(sampled) >= 3
    assert max(sampled) <= 4


def test_transport_height_ratio_accepts_configured_range() -> None:
    config = {
        "simulation": {"scenario_duration_s": 120.0},
        "task_generation": {
            "tasks_per_crane_range": [1, 1],
            "pickup_height_range_m": [2.0, 2.0],
            "dropoff_height_range_m": [3.0, 3.0],
            "transport_height_ratio": [0.55, 0.85],
            "load_weight_ratio_range": [0.5, 0.5],
        }
    }

    tasks = generate_tasks(0, "no_overlap_safe", [_crane(0)], config, np.random.default_rng(11))

    assert len(tasks) == 1
    assert 55.0 <= tasks[0].transport_h <= 85.0


def test_task_start_times_are_distributed_over_scenario_duration_and_sorted_per_crane() -> None:
    config = {
        "simulation": {"scenario_duration_s": 900.0},
        "task_generation": {
            "tasks_per_crane_range": [4, 4],
            "pickup_height_range_m": [2.0, 2.0],
            "dropoff_height_range_m": [3.0, 3.0],
            "transport_height_ratio": [0.55, 0.85],
            "load_weight_ratio_range": [0.5, 0.5],
        },
    }

    tasks = generate_tasks(0, "overlap_no_conflict", [_crane(0), _crane(1)], config, np.random.default_rng(11))

    start_times_by_crane = {}
    for task in tasks:
        start_times_by_crane.setdefault(task.crane_index, []).append(task.start_time)
        assert 0.0 <= task.start_time <= 900.0 * 0.75
    assert set(start_times_by_crane) == {0, 1}
    for start_times in start_times_by_crane.values():
        assert start_times == sorted(start_times)
    assert any(start_time > 80.0 for start_times in start_times_by_crane.values() for start_time in start_times)


def test_default_config_is_formal_dataset_configuration() -> None:
    config = load_config(Path("configs") / "default.yaml")

    assert config["simulation"]["num_scenarios"] == 100
    assert config["simulation"]["scenario_duration_s"] == 900.0
    assert config["simulation"]["dt"] == 0.2
    assert config["simulation"]["num_cranes_range"] == [3, 6]
    assert config["simulation"]["save_format"] == ["csv", "parquet", "npz"]
    assert config["task_generation"]["tasks_per_crane_range"] == [4, 12]
    assert config["dynamics"]["hook_clearance_m"] == 2.0
    assert config["quality_control"]["target_risk_ratio_range"] == [0.10, 0.30]
    assert config["quality_control"]["fail_on_risk_ratio_out_of_range"] is True


def test_debug_configs_use_full_yaml_schema() -> None:
    for config_name in ["debug_small.yaml", "debug_fast.yaml"]:
        config = load_config(Path("configs") / config_name)

        assert config["simulation"]["num_cranes_range"][0] >= 3
        assert isinstance(config["task_generation"]["stage_tolerance"], dict)
        assert isinstance(config["task_generation"]["transport_height_ratio"], list)
        assert config["controller"]["command_smoothing"] is True
        assert "command_smoothing_alpha" in config["controller"]
        assert config["dynamics"]["hook_clearance_m"] == 2.0
        assert config["quality_control"]["target_risk_ratio_range"] == [0.0, 0.80]
        assert config["quality_control"]["fail_on_risk_ratio_out_of_range"] is False


def test_configured_save_formats_normalize_and_reject_unknown_values() -> None:
    assert _configured_save_formats({"simulation": {"save_format": "csv"}}) == {"csv"}
    assert _configured_save_formats({"simulation": {"save_format": ["CSV", "parquet", "npz"]}}) == {
        "csv",
        "parquet",
        "npz",
    }

    try:
        _configured_save_formats({"simulation": {"save_format": ["csv", "json"]}})
    except ValueError as exc:
        assert "save_format" in str(exc)
    else:
        raise AssertionError("Expected unknown save_format to be rejected")

    try:
        _configured_save_formats({"simulation": {"save_format": ["npz"]}})
    except ValueError as exc:
        assert "csv or parquet" in str(exc)
    else:
        raise AssertionError("Expected save_format without a table format to be rejected")


def test_simulate_dataset_outputs_formal_ids_and_window_scenario_uids() -> None:
    config = load_config(Path("configs") / "debug_fast.yaml")
    config["project"]["output_dir"] = "test_artifacts/id_schema"
    config["project"]["run_id"] = "run_schema"
    config["simulation"]["num_scenarios"] = 1
    config["simulation"]["scenario_duration_s"] = 12.0
    config["simulation"]["dt"] = 1.0
    config["simulation"]["save_format"] = ["csv", "npz"]
    config["quality_control"]["target_risk_ratio_range"] = [0.0, 1.0]

    output_dir, _ = simulate_dataset(config)

    scenario_table = pd.read_csv(output_dir / "tables" / "scenario_table.csv")
    crane_static = pd.read_csv(output_dir / "tables" / "crane_static.csv")
    task_table = pd.read_csv(output_dir / "tables" / "task_table.csv")
    state_true = pd.read_csv(output_dir / "tables" / "state_true.csv")
    state_obs = pd.read_csv(output_dir / "tables" / "state_obs.csv")
    geometry_table = pd.read_csv(output_dir / "tables" / "geometry_table.csv")
    edge_current = pd.read_csv(output_dir / "tables" / "edge_current.csv")
    edge_future_label = pd.read_csv(output_dir / "tables" / "edge_future_label.csv")

    assert scenario_table.loc[0, "scenario_id"] == "scenario_000000"
    assert scenario_table.loc[0, "scenario_uid"] == "scenario_000000"
    assert scenario_table.loc[0, "scenario_index"] == 0
    assert crane_static["crane_id"].map(lambda value: isinstance(value, str) and value.startswith("crane_")).all()
    assert task_table["task_id"].map(lambda value: isinstance(value, str) and value.startswith("task_")).all()
    assert {"crane_uid", "crane_index"}.issubset(crane_static.columns)
    assert {"task_uid", "task_index"}.issubset(task_table.columns)
    assert {"scenario_uid", "crane_uid", "task_uid", "scenario_index", "crane_index", "task_index"}.issubset(state_true.columns)
    assert {"scenario_uid", "crane_uid", "task_uid", "scenario_index", "crane_index", "task_index"}.issubset(state_obs.columns)
    assert {"scenario_uid", "crane_uid", "scenario_index", "crane_index"}.issubset(geometry_table.columns)
    assert {"scenario_uid", "crane_i_uid", "crane_j_uid", "scenario_index", "crane_i_index", "crane_j_index"}.issubset(edge_current.columns)
    assert edge_current["crane_i"].map(lambda value: isinstance(value, str) and value.startswith("crane_")).all()
    assert edge_current["crane_j"].map(lambda value: isinstance(value, str) and value.startswith("crane_")).all()
    assert edge_future_label["crane_i"].map(lambda value: isinstance(value, str) and value.startswith("crane_")).all()
    assert edge_future_label["crane_j"].map(lambda value: isinstance(value, str) and value.startswith("crane_")).all()

    with np.load(output_dir / "windows" / "train_windows.npz", allow_pickle=False) as data:
        assert data["scenario_ids"].dtype.kind in {"U", "S"}
        assert set(data["scenario_ids"].tolist()) == {"scenario_000000"}
        assert "scenario_indices" in data.files
        assert data["scenario_indices"].dtype.kind in {"i", "u"}
        assert "scenario_business_ids" in data.files
        assert "scenario_uids" in data.files
        assert data["scenario_business_ids"].dtype.kind in {"U", "S"}


def test_simulate_dataset_records_parquet_write_status_in_metadata(monkeypatch) -> None:
    config = load_config(Path("configs") / "debug_fast.yaml")
    config["project"]["output_dir"] = "test_artifacts/parquet_metadata"
    config["project"]["run_id"] = "run_parquet_metadata"
    config["simulation"]["num_scenarios"] = 1
    config["simulation"]["scenario_duration_s"] = 8.0
    config["simulation"]["dt"] = 1.0
    config["simulation"]["save_format"] = ["csv", "parquet", "npz"]
    config["quality_control"]["target_risk_ratio_range"] = [0.0, 1.0]

    def fake_write_optional_parquet(df, path, required=False):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("fake parquet", encoding="utf-8")
        return {"written": True, "path": str(path), "error": None}

    monkeypatch.setattr("tower_sim.simulation.write_optional_parquet", fake_write_optional_parquet)

    output_dir, _ = simulate_dataset(config)

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["save_formats"] == ["csv", "npz", "parquet"]
    assert metadata["parquet_written"] is True
    assert metadata["parquet_tables"]["state_true"]["written"] is True
    assert metadata["parquet_tables"]["edge_future_label"]["error"] is None


def test_simulate_dataset_records_parquet_failures_without_raising(monkeypatch) -> None:
    config = load_config(Path("configs") / "debug_fast.yaml")
    config["project"]["output_dir"] = "test_artifacts/parquet_metadata_failure"
    config["project"]["run_id"] = "run_parquet_metadata_failure"
    config["simulation"]["num_scenarios"] = 1
    config["simulation"]["scenario_duration_s"] = 8.0
    config["simulation"]["dt"] = 1.0
    config["simulation"]["save_format"] = ["csv", "parquet", "npz"]
    config["quality_control"]["target_risk_ratio_range"] = [0.0, 1.0]

    def fake_write_optional_parquet(df, path, required=False):
        if required:
            raise RuntimeError(f"Failed to write parquet: {path}")
        return {"written": False, "path": str(path), "error": "pyarrow not installed"}

    monkeypatch.setattr("tower_sim.simulation.write_optional_parquet", fake_write_optional_parquet)

    output_dir, _ = simulate_dataset(config)

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["parquet_written"] is False
    assert metadata["parquet_tables"]["state_true"]["written"] is False
    assert metadata["parquet_tables"]["state_true"]["error"] == "pyarrow not installed"


def test_no_overlap_safe_scenarios_have_no_radius_overlap_or_are_relabelled() -> None:
    config = load_config(Path("configs") / "debug_fast.yaml")
    config["project"]["output_dir"] = "test_artifacts/no_overlap"
    config["project"]["run_id"] = "run_no_overlap"
    config["simulation"]["num_scenarios"] = 1
    config["simulation"]["scenario_duration_s"] = 8.0
    config["simulation"]["dt"] = 1.0
    config["simulation"]["num_cranes_range"] = [3, 3]
    config["quality_control"]["target_risk_ratio_range"] = [0.0, 1.0]
    config["layout"]["site_size_m"] = [80.0, 80.0]
    config["crane_static"]["jib_length_range_m"] = [60.0, 60.0]
    config["crane_static"]["max_radius_margin_m"] = 0.0
    config["layout"]["overlap_scene_ratio"] = {
        "no_overlap_safe": 1.0,
        "overlap_no_conflict": 0.0,
        "two_crane_crossing": 0.0,
        "multi_crane_avoidance": 0.0,
        "delayed_or_failed_avoidance": 0.0,
    }

    output_dir, _ = simulate_dataset(config)

    scenario_table = pd.read_csv(output_dir / "tables" / "scenario_table.csv")
    crane_static = pd.read_csv(output_dir / "tables" / "crane_static.csv")
    if scenario_table.loc[0, "scene_type"] == "no_overlap_safe":
        statics = [
            CraneStatic(
                scenario_id=str(row["scenario_id"]),
                scenario_index=int(row["scenario_index"]),
                crane_id=str(row["crane_id"]),
                crane_index=int(row["crane_index"]),
                base_x=float(row["base_x"]),
                base_y=float(row["base_y"]),
                base_z=float(row["base_z"]),
                tower_height=float(row["tower_height"]),
                jib_length=float(row["jib_length"]),
                min_radius=float(row["min_radius"]),
                max_radius=float(row["max_radius"]),
                safety_radius_arm=float(row["safety_radius_arm"]),
                safety_radius_hook=float(row["safety_radius_hook"]),
                max_theta_dot=float(row["max_theta_dot"]),
                max_r_dot=float(row["max_r_dot"]),
                max_h_dot=float(row["max_h_dot"]),
                max_theta_acc=float(row["max_theta_acc"]),
                max_r_acc=float(row["max_r_acc"]),
                max_h_acc=float(row["max_h_acc"]),
                response_tau=float(row["response_tau"]),
                load_capacity=float(row["load_capacity"]),
                priority=int(row["priority"]),
            )
            for _, row in crane_static.iterrows()
        ]
        assert not has_radius_overlap(statics)
    else:
        assert scenario_table.loc[0, "scene_type"] == "overlap_no_conflict"
        assert scenario_table.loc[0, "layout_relabel_reason"] == "no_overlap_infeasible"


def test_no_overlap_layout_validation_only_generates_static_layout(monkeypatch) -> None:
    config = load_config(Path("configs") / "debug_fast.yaml")
    config["simulation"]["num_cranes_range"] = [3, 3]
    calls = {"count": 0}

    overlapping = [replace(_crane(crane_id), base_x=0.0, base_y=0.0, max_radius=50.0) for crane_id in range(3)]
    separated = [replace(_crane(crane_id), base_x=crane_id * 120.0, base_y=0.0, max_radius=40.0) for crane_id in range(3)]

    def fake_generate_cranes(scenario_id, scene_type, num_cranes, config, rng):
        calls["count"] += 1
        return overlapping if calls["count"] == 1 else separated

    monkeypatch.setattr("tower_sim.simulation.generate_cranes", fake_generate_cranes)

    scene_type, reason, cranes = _generate_validated_cranes(
        scenario_id=0,
        scene_type="no_overlap_safe",
        num_cranes=3,
        config=config,
        rng=np.random.default_rng(5),
    )

    assert scene_type == "no_overlap_safe"
    assert reason == ""
    assert cranes is separated
    assert calls["count"] == 2
