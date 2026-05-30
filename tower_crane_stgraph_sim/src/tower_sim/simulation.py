from __future__ import annotations

import math
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tower_sim.config import get_command_smoothing, get_stage_tolerance, save_config, scenario_seed
from tower_sim.controller import (
    advance_task_stage,
    choose_active_task,
    make_nominal_command,
    smooth_command,
    target_for_stage,
)
from tower_sim.dataclasses import Command, CraneState, CraneStatic
from tower_sim.dynamics import update_state
from tower_sim.geometry import reconstruct_geometry
from tower_sim.interaction import apply_avoidance, compute_edges_for_step
from tower_sim.io_utils import dataset_paths, ensure_dataset_dirs, make_run_root, split_scenario_ids, write_csv, write_optional_parquet
from tower_sim.labels import compute_future_labels
from tower_sim.layout import generate_cranes, sample_scene_type
from tower_sim.quality import generate_quality_report
from tower_sim.sensors import generate_observations
from tower_sim.tasks import generate_tasks
from tower_sim.windowing import make_windows
from tower_sim.windowing import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    Y_MIN_DISTANCE_FEATURE_NAMES,
    Y_RISK_FEATURE_NAMES,
    Y_TRAJ_FEATURE_NAMES,
)


SCENARIO_COLUMNS = [
    "scenario_id",
    "scene_type",
    "num_cranes",
    "duration_s",
    "dt",
    "site_length_m",
    "site_width_m",
    "seed",
    "split",
]

STATE_COLUMNS = [
    "scenario_id",
    "timestamp",
    "step",
    "crane_id",
    "theta",
    "r",
    "h",
    "theta_dot",
    "r_dot",
    "h_dot",
    "theta_ddot",
    "r_ddot",
    "h_ddot",
    "load_weight",
    "command_theta",
    "command_r",
    "command_h",
    "brake_flag",
    "emergency_flag",
    "task_id",
    "task_stage",
]

GEOMETRY_COLUMNS = [
    "scenario_id",
    "timestamp",
    "step",
    "crane_id",
    "root_x",
    "root_y",
    "root_z",
    "tip_x",
    "tip_y",
    "tip_z",
    "hook_x",
    "hook_y",
    "hook_z",
]

EDGE_COLUMNS = [
    "scenario_id",
    "timestamp",
    "step",
    "crane_i",
    "crane_j",
    "d_arm_arm",
    "d_arm_hook_i_to_j",
    "d_arm_hook_j_to_i",
    "d_hook_hook",
    "delta_theta",
    "delta_theta_dot",
    "delta_r",
    "delta_h",
    "delta_tower_height",
    "base_distance",
    "overlap_ratio",
    "relative_approach_speed",
    "relative_approach_speed_arm_arm",
    "relative_approach_speed_arm_hook",
    "relative_approach_speed_hook_hook",
    "ttc_est_arm_arm",
    "ttc_est_arm_hook",
    "ttc_est_hook_hook",
    "same_height_risk_zone",
]


def _progress(iterable, **kwargs):
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kwargs)
    except Exception:
        return iterable


def _copy_state_with_stage(state: CraneState) -> CraneState:
    return CraneState(
        theta=state.theta,
        r=state.r,
        h=state.h,
        theta_dot=state.theta_dot,
        r_dot=state.r_dot,
        h_dot=state.h_dot,
        theta_ddot=state.theta_ddot,
        r_ddot=state.r_ddot,
        h_ddot=state.h_ddot,
        load_weight=state.load_weight,
        task_id=state.task_id,
        task_stage=state.task_stage,
    )


def _initial_state(crane: CraneStatic, rng: np.random.Generator) -> CraneState:
    h = float(np.clip(crane.tower_height * 0.55, 2.0, crane.tower_height - 2.0))
    return CraneState(
        theta=float(rng.uniform(-math.pi, math.pi)),
        r=float(rng.uniform(crane.min_radius, crane.max_radius)),
        h=h,
        theta_dot=0.0,
        r_dot=0.0,
        h_dot=0.0,
        theta_ddot=0.0,
        r_ddot=0.0,
        h_ddot=0.0,
        load_weight=0.0,
        task_id=0,
        task_stage="move_to_pickup",
    )


def _state_row(
    scenario_id: int,
    timestamp: float,
    step: int,
    crane_id: int,
    state: CraneState,
    command: Command,
) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "timestamp": timestamp,
        "step": step,
        "crane_id": crane_id,
        "theta": state.theta,
        "r": state.r,
        "h": state.h,
        "theta_dot": state.theta_dot,
        "r_dot": state.r_dot,
        "h_dot": state.h_dot,
        "theta_ddot": state.theta_ddot,
        "r_ddot": state.r_ddot,
        "h_ddot": state.h_ddot,
        "load_weight": state.load_weight,
        "command_theta": command.theta_dot_cmd,
        "command_r": command.r_dot_cmd,
        "command_h": command.h_dot_cmd,
        "brake_flag": command.brake_flag,
        "emergency_flag": command.emergency_flag,
        "task_id": state.task_id,
        "task_stage": state.task_stage,
    }


def _geometry_row(scenario_id: int, timestamp: float, step: int, crane_id: int, static: CraneStatic, state: CraneState) -> dict[str, Any]:
    geom = reconstruct_geometry(static, state)
    return {
        "scenario_id": scenario_id,
        "timestamp": timestamp,
        "step": step,
        "crane_id": crane_id,
        "root_x": geom.root[0],
        "root_y": geom.root[1],
        "root_z": geom.root[2],
        "tip_x": geom.tip[0],
        "tip_y": geom.tip[1],
        "tip_z": geom.tip[2],
        "hook_x": geom.hook[0],
        "hook_y": geom.hook[1],
        "hook_z": geom.hook[2],
    }


def _build_data_dictionary(output_path: Path) -> None:
    content = f"""# Data Dictionary

All distances are in meters, time is in seconds, angular values are in radians, and angular differences are wrapped to [-pi, pi]. Safety thresholds are simulation parameters, not normative construction-code values.

## scenario_table.csv

`scenario_id`, `scene_type`, `num_cranes`, `duration_s`, `dt`, `site_length_m`, `site_width_m`, `seed`, `split`.

## crane_static.csv

`scenario_id`, `crane_id`, `base_x`, `base_y`, `base_z`, `tower_height`, `jib_length`, `min_radius`, `max_radius`, `safety_radius_arm`, `safety_radius_hook`, `max_theta_dot`, `max_r_dot`, `max_h_dot`, `max_theta_acc`, `max_r_acc`, `max_h_acc`, `response_tau`, `load_capacity`, `priority`.

These are static node attributes and crane motion limits. They may be used as input features where appropriate.

## task_table.csv

`scenario_id`, `crane_id`, `task_id`, `start_time`, `pickup_theta`, `pickup_r`, `pickup_h`, `dropoff_theta`, `dropoff_r`, `dropoff_h`, `transport_h`, `load_weight`, `priority`.

This table explains the simulated task plan. It is not required as a direct model input because executed commands and states are already stored in state tables.

## state_true.csv

`scenario_id`, `timestamp`, `step`, `crane_id`, `theta`, `r`, `h`, `theta_dot`, `r_dot`, `h_dot`, `theta_ddot`, `r_ddot`, `h_ddot`, `load_weight`, `command_theta`, `command_r`, `command_h`, `brake_flag`, `emergency_flag`, `task_id`, `task_stage`.

Ground-truth dynamic states and issued commands. This table is used for geometry reconstruction and labels. Use `state_obs.csv` for model input.

## state_obs.csv

`scenario_id`, `timestamp`, `step`, `crane_id`, `theta`, `r`, `h`, `theta_dot`, `r_dot`, `h_dot`, `theta_ddot`, `r_ddot`, `h_ddot`, `load_weight`, `command_theta`, `command_r`, `command_h`, `brake_flag`, `emergency_flag`, `task_id`, `task_stage`, `theta_missing`, `r_missing`, `h_missing`, `obs_delay_steps`, `is_outlier`.

Noisy/delayed/dropout observations derived from `state_true.csv`. This table is the node-input source. Missing masks and outlier markers are included.

## geometry_table.csv

`scenario_id`, `timestamp`, `step`, `crane_id`, `root_x`, `root_y`, `root_z`, `tip_x`, `tip_y`, `tip_z`, `hook_x`, `hook_y`, `hook_z`.

Per-step reconstructed jib root, jib tip, and hook coordinates for every crane.

## edge_current.csv

`scenario_id`, `timestamp`, `step`, `crane_i`, `crane_j`, `d_arm_arm`, `d_arm_hook_i_to_j`, `d_arm_hook_j_to_i`, `d_hook_hook`, `delta_theta`, `delta_theta_dot`, `delta_r`, `delta_h`, `delta_tower_height`, `base_distance`, `overlap_ratio`, `relative_approach_speed`, `relative_approach_speed_arm_arm`, `relative_approach_speed_arm_hook`, `relative_approach_speed_hook_hook`, `ttc_est_arm_arm`, `ttc_est_arm_hook`, `ttc_est_hook_hook`, `same_height_risk_zone`.

Current-time physical-prior edge features computed from current geometry and current relative motion. These fields are allowed as model inputs. `ttc_est_*` is current-state-only and is not a future label.

## edge_future_label.csv

`scenario_id`, `timestamp`, `step`, `horizon_s`, `crane_i`, `crane_j`, `future_min_d_arm_arm`, `future_min_d_arm_hook_i_to_j`, `future_min_d_arm_hook_j_to_i`, `future_min_d_hook_hook`, `risk_arm_arm`, `risk_arm_hook_i_to_j`, `risk_arm_hook_j_to_i`, `risk_hook_hook`, `ttc_label_arm_arm`, `ttc_label_arm_hook`, `ttc_label_hook_hook`.

Future minimum distances, risk labels, and label TTC values computed from `state_true.csv`. These fields are labels only and must not be used as input features.

## *_windows.npz

Sliding-window tensors. `node_features` and `edge_features` are inputs. `y_traj`, `y_risk`, and `y_min_distance` are labels.

- `node_features`: {NODE_FEATURE_NAMES}
- `edge_features`: {EDGE_FEATURE_NAMES}
- `y_traj`: {Y_TRAJ_FEATURE_NAMES}
- `y_risk`: {Y_RISK_FEATURE_NAMES}
- `y_min_distance`: {Y_MIN_DISTANCE_FEATURE_NAMES}

Train/validation/test splits are by `scenario_id`; never mix windows from one scenario across splits. When `split.add_generalization_test` is enabled, `windows/generalization_windows.npz` is generated from held-out scenario ids.
"""
    output_path.write_text(content, encoding="utf-8")


def _configured_save_formats(config: dict[str, Any]) -> set[str]:
    value = config.get("simulation", {}).get("save_format", "csv")
    if isinstance(value, str):
        return {value.lower()}
    if isinstance(value, (list, tuple)):
        return {str(item).lower() for item in value}
    return {"csv"}


def _write_table(df: pd.DataFrame, tables_dir: Path, name: str, save_formats: set[str]) -> None:
    if "csv" in save_formats:
        write_csv(df, tables_dir / f"{name}.csv")
    if "parquet" in save_formats:
        write_optional_parquet(df, tables_dir / f"{name}.parquet")


def _num_cranes_for_scene(scene_type: str, num_min: int, num_max: int, rng: np.random.Generator) -> int:
    sampled = int(rng.integers(num_min, num_max + 1))
    if scene_type == "two_crane_crossing":
        return max(3, sampled)
    return sampled


def _write_readme_copy(output_path: Path) -> None:
    source = Path(__file__).resolve().parents[2] / "README.md"
    if source.exists():
        output_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _simulate_one_scenario(
    scenario_id: int,
    scene_type: str,
    num_cranes: int,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dt = float(config["simulation"]["dt"])
    duration = float(config["simulation"]["scenario_duration_s"])
    steps = int(round(duration / dt)) + 1
    cranes = generate_cranes(scenario_id, scene_type, num_cranes, config, rng)
    tasks = generate_tasks(scenario_id, scene_type, cranes, config, rng)
    tasks_by_crane = {crane.crane_id: [task for task in tasks if task.crane_id == crane.crane_id] for crane in cranes}
    states = {crane.crane_id: _initial_state(crane, rng) for crane in cranes}
    if scene_type in {"two_crane_crossing", "multi_crane_avoidance", "delayed_or_failed_avoidance"}:
        for crane in cranes:
            first_tasks = tasks_by_crane.get(crane.crane_id, [])
            if not first_tasks:
                continue
            task = first_tasks[0]
            states[crane.crane_id] = CraneState(
                theta=task.pickup_theta,
                r=task.pickup_r,
                h=task.transport_h,
                theta_dot=0.0,
                r_dot=0.0,
                h_dot=0.0,
                theta_ddot=0.0,
                r_ddot=0.0,
                h_ddot=0.0,
                load_weight=task.load_weight,
                task_id=task.task_id,
                task_stage="transport_to_dropoff",
            )
    prev_commands: dict[int, Command] = {}
    prev_pair_min: dict[tuple[int, int], dict[str, float]] = {}
    avoidance_delay_counters: dict[int, int] = {}
    state_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []

    static_by_id = {crane.crane_id: crane for crane in cranes}
    stage_tolerance = get_stage_tolerance(config["task_generation"])
    command_smoothing = get_command_smoothing(config["controller"])
    for step in range(steps):
        timestamp = step * dt
        commands: dict[int, Command] = {}
        staged_states: dict[int, CraneState] = {}
        for crane_id, state in states.items():
            current_state = _copy_state_with_stage(state)
            task = choose_active_task(tasks_by_crane[crane_id], current_state, timestamp)
            current_state = advance_task_stage(current_state, task, stage_tolerance)
            target = target_for_stage(task, current_state)
            command = make_nominal_command(
                current_state,
                static_by_id[crane_id],
                target,
                float(config["controller"]["k_theta"]),
                float(config["controller"]["k_r"]),
                float(config["controller"]["k_h"]),
            )
            command = smooth_command(command, prev_commands.get(crane_id), command_smoothing)
            staged_states[crane_id] = current_state
            commands[crane_id] = command

        commands = apply_avoidance(commands, cranes, staged_states, config, rng, dt, avoidance_delay_counters)
        next_states: dict[int, CraneState] = {}
        for crane_id, current_state in staged_states.items():
            next_state = update_state(
                current_state,
                static_by_id[crane_id],
                commands[crane_id],
                dt=dt,
                h_clearance=2.0,
                h_min=0.0,
                min_acc_scale=float(config["load_effect"].get("min_acc_scale", 0.5))
                if bool(config["load_effect"].get("enabled", True))
                else 1.0,
            )
            next_states[crane_id] = next_state
            state_rows.append(_state_row(scenario_id, timestamp, step, crane_id, next_state, commands[crane_id]))
            geometry_rows.append(_geometry_row(scenario_id, timestamp, step, crane_id, static_by_id[crane_id], next_state))

        geometries = {cid: reconstruct_geometry(static_by_id[cid], state) for cid, state in next_states.items()}
        pair_rows, prev_pair_min = compute_edges_for_step(
            cranes,
            next_states,
            geometries,
            prev_pair_min,
            dt=dt,
            thresholds=config["risk_thresholds"],
        )
        for row in pair_rows:
            edge_rows.append({"scenario_id": scenario_id, "timestamp": timestamp, "step": step, **row})
        states = next_states
        prev_commands = commands
    return [asdict(c) for c in cranes], [asdict(t) for t in tasks], state_rows, geometry_rows, edge_rows


def simulate_dataset(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Generate all simulation tables, windows, reports, and plots."""

    run_root = make_run_root(config["project"]["output_dir"], config["project"].get("run_id"))
    paths = ensure_dataset_dirs(dataset_paths(run_root))
    output_dir = paths.root
    save_config(config, paths.config_used)

    base_seed = int(config["project"]["random_seed"])
    num_scenarios = int(config["simulation"]["num_scenarios"])
    num_min, num_max = [int(x) for x in config["simulation"]["num_cranes_range"]]
    split_assignments = split_scenario_ids(
        list(range(num_scenarios)),
        float(config["split"]["train_ratio"]),
        float(config["split"]["val_ratio"]),
        float(config["split"]["test_ratio"]),
        add_generalization_test=bool(config["split"].get("add_generalization_test", False)),
    )

    scenario_rows: list[dict[str, Any]] = []
    crane_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []

    for scenario_id in _progress(range(num_scenarios), desc="scenarios"):
        seed = scenario_seed(base_seed, scenario_id)
        rng = np.random.default_rng(seed)
        scene_type = sample_scene_type(rng, config["layout"]["overlap_scene_ratio"])
        num_cranes = _num_cranes_for_scene(scene_type, num_min, num_max, rng)
        cranes, tasks, states, geometries, edges = _simulate_one_scenario(
            scenario_id,
            scene_type,
            num_cranes,
            config,
            rng,
        )
        site_length, site_width = config["layout"]["site_size_m"]
        scenario_rows.append(
            {
                "scenario_id": scenario_id,
                "scene_type": scene_type,
                "num_cranes": num_cranes,
                "duration_s": float(config["simulation"]["scenario_duration_s"]),
                "dt": float(config["simulation"]["dt"]),
                "site_length_m": float(site_length),
                "site_width_m": float(site_width),
                "seed": seed,
                "split": split_assignments[scenario_id],
            }
        )
        crane_rows.extend(cranes)
        task_rows.extend(tasks)
        state_rows.extend(states)
        geometry_rows.extend(geometries)
        edge_rows.extend(edges)

    scenario_table = pd.DataFrame(scenario_rows, columns=SCENARIO_COLUMNS)
    crane_static = pd.DataFrame(crane_rows)
    task_table = pd.DataFrame(task_rows)
    state_true = pd.DataFrame(state_rows, columns=STATE_COLUMNS)
    geometry_table = pd.DataFrame(geometry_rows, columns=GEOMETRY_COLUMNS)
    edge_current = pd.DataFrame(edge_rows, columns=EDGE_COLUMNS)
    obs_rng = np.random.default_rng(scenario_seed(base_seed, 0, stream=9))
    state_obs = generate_observations(state_true, config, obs_rng)
    horizons_s = [float(x) for x in config["windowing"].get("prediction_horizons_s", [config["windowing"]["prediction_horizon_s"]])]
    edge_future_label = compute_future_labels(
        state_true,
        crane_static,
        dt=float(config["simulation"]["dt"]),
        horizons_s=horizons_s,
        thresholds=config["risk_thresholds"],
    )

    save_formats = _configured_save_formats(config)
    _write_table(scenario_table, paths.tables, "scenario_table", save_formats)
    _write_table(crane_static, paths.tables, "crane_static", save_formats)
    _write_table(task_table, paths.tables, "task_table", save_formats)
    _write_table(state_true, paths.tables, "state_true", save_formats)
    _write_table(state_obs, paths.tables, "state_obs", save_formats)
    _write_table(geometry_table, paths.tables, "geometry_table", save_formats)
    _write_table(edge_current, paths.tables, "edge_current", save_formats)
    _write_table(edge_future_label, paths.tables, "edge_future_label", save_formats)

    window_counts = make_windows(
        state_obs,
        state_true,
        crane_static,
        edge_current,
        edge_future_label,
        scenario_table,
        config,
        paths.windows,
    )
    _build_data_dictionary(paths.data_dictionary)
    _write_readme_copy(paths.readme)
    paths.metadata.write_text(
        json.dumps(
            {
                "project": config["project"]["name"],
                "version": config["project"]["version"],
                "num_scenarios": int(scenario_table["scenario_id"].nunique()),
                "splits": scenario_table["split"].value_counts().to_dict(),
                "tables_dir": str(paths.tables),
                "windows_dir": str(paths.windows),
                "quality_dir": str(paths.quality),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    stats = generate_quality_report(
        paths.quality,
        scenario_table,
        crane_static,
        task_table,
        state_true,
        state_obs,
        edge_current,
        edge_future_label,
        config,
        geometry_table=geometry_table,
        config_used_path=paths.config_used,
        window_counts=window_counts,
    )
    return Path(output_dir), stats
