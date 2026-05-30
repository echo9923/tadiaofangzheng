from pathlib import Path

import numpy as np

from tower_sim.config import load_config
from tower_sim.dataclasses import CraneStatic
from tower_sim.simulation import _num_cranes_for_scene
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


def test_default_config_is_formal_dataset_configuration() -> None:
    config = load_config(Path("configs") / "default.yaml")

    assert config["simulation"]["scenario_duration_s"] == 900.0
    assert config["simulation"]["dt"] == 0.2
    assert config["simulation"]["num_cranes_range"] == [3, 6]
    assert config["simulation"]["save_format"] == ["csv", "parquet", "npz"]


def test_debug_configs_use_full_yaml_schema() -> None:
    for config_name in ["debug_small.yaml", "debug_fast.yaml"]:
        config = load_config(Path("configs") / config_name)

        assert config["simulation"]["num_cranes_range"][0] >= 3
        assert isinstance(config["task_generation"]["stage_tolerance"], dict)
        assert isinstance(config["task_generation"]["transport_height_ratio"], list)
        assert config["controller"]["command_smoothing"] is True
        assert "command_smoothing_alpha" in config["controller"]
