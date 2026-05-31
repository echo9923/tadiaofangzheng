from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tower_sim.ids import crane_index_from_id, crane_key, scenario_id_from_index, scenario_index_from_id, scenario_key

NODE_FEATURE_NAMES = [
    "sin_theta",
    "cos_theta",
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
    "tower_height",
    "jib_length",
    "max_radius",
    "priority",
    "missing_mask_theta",
    "missing_mask_r",
    "missing_mask_h",
]

EDGE_FEATURE_NAMES = [
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

Y_TRAJ_FEATURE_NAMES = ["sin_theta_future", "cos_theta_future", "r_future", "h_future"]
Y_RISK_FEATURE_NAMES = [
    "risk_arm_arm",
    "risk_arm_hook_i_to_j",
    "risk_arm_hook_j_to_i",
    "risk_hook_hook",
]
Y_MIN_DISTANCE_FEATURE_NAMES = [
    "future_min_d_arm_arm",
    "future_min_d_arm_hook_i_to_j",
    "future_min_d_arm_hook_j_to_i",
    "future_min_d_hook_hook",
]


def assert_no_future_leakage(node_feature_names: list[str], edge_feature_names: list[str]) -> None:
    """Raise if input feature names include future label tokens."""

    bad_tokens = ("future_min_d", "ttc_label")
    bad_prefixes = (
        "risk_arm_arm",
        "risk_arm_hook",
        "risk_hook_hook",
        "future_risk",
        "label_risk",
    )
    offenders = [
        name
        for name in list(node_feature_names) + list(edge_feature_names)
        if any(token in name for token in bad_tokens) or any(name.startswith(prefix) for prefix in bad_prefixes)
    ]
    if offenders:
        raise ValueError(f"Future leakage in input features: {offenders}")


def validate_split_disjoint(train_ids: np.ndarray, val_ids: np.ndarray, test_ids: np.ndarray) -> None:
    """Validate that scenario ids are mutually exclusive across splits."""

    sets = [set(arr.tolist()) for arr in [train_ids, val_ids, test_ids]]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("scenario_id split overlap detected")


def validate_named_splits_disjoint(split_ids: dict[str, np.ndarray]) -> None:
    """Validate that all named split id arrays are mutually exclusive."""

    seen: dict[Any, str] = {}
    for split, ids in split_ids.items():
        for scenario_id in ids.tolist():
            if scenario_id in seen and seen[scenario_id] != split:
                raise ValueError(f"scenario_id split overlap detected: {scenario_id} in {seen[scenario_id]} and {split}")
            seen[scenario_id] = split


def _node_feature(row: pd.Series, static_row: pd.Series) -> np.ndarray:
    theta = float(row["theta"]) if not pd.isna(row["theta"]) else 0.0
    return np.array(
        [
            math.sin(theta),
            math.cos(theta),
            0.0 if pd.isna(row["r"]) else float(row["r"]),
            0.0 if pd.isna(row["h"]) else float(row["h"]),
            float(row["theta_dot"]),
            float(row["r_dot"]),
            float(row["h_dot"]),
            float(row["theta_ddot"]),
            float(row["r_ddot"]),
            float(row["h_ddot"]),
            float(row["load_weight"]),
            float(row["command_theta"]),
            float(row["command_r"]),
            float(row["command_h"]),
            float(row["brake_flag"]),
            float(row["emergency_flag"]),
            float(static_row["tower_height"]),
            float(static_row["jib_length"]),
            float(static_row["max_radius"]),
            float(static_row["priority"]),
            float(row.get("theta_missing", 0)),
            float(row.get("r_missing", 0)),
            float(row.get("h_missing", 0)),
        ],
        dtype=np.float32,
    )


def _empty_npz(path: Path) -> None:
    np.savez_compressed(
        path,
        node_features=np.zeros((0, 0, 0, len(NODE_FEATURE_NAMES)), dtype=np.float32),
        edge_features=np.zeros((0, 0, 0, 0, len(EDGE_FEATURE_NAMES)), dtype=np.float32),
        node_mask=np.zeros((0, 0), dtype=bool),
        edge_mask=np.zeros((0, 0, 0), dtype=bool),
        time_mask=np.zeros((0, 0), dtype=bool),
        y_traj=np.zeros((0, 0, 0, len(Y_TRAJ_FEATURE_NAMES)), dtype=np.float32),
        y_risk=np.zeros((0, 0, 0, len(Y_RISK_FEATURE_NAMES)), dtype=np.float32),
        y_min_distance=np.zeros((0, 0, 0, len(Y_MIN_DISTANCE_FEATURE_NAMES)), dtype=np.float32),
        node_feature_names=np.array(NODE_FEATURE_NAMES),
        edge_feature_names=np.array(EDGE_FEATURE_NAMES),
        y_traj_feature_names=np.array(Y_TRAJ_FEATURE_NAMES),
        y_risk_feature_names=np.array(Y_RISK_FEATURE_NAMES),
        y_min_distance_feature_names=np.array(Y_MIN_DISTANCE_FEATURE_NAMES),
        scenario_ids=np.array([], dtype=np.int32),
        scenario_indices=np.array([], dtype=np.int32),
        scenario_business_ids=np.array([], dtype="<U1"),
        scenario_uids=np.array([], dtype="<U1"),
        window_start_steps=np.array([], dtype=np.int32),
    )


def _with_index_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "scenario_index" not in result.columns and "scenario_id" in result.columns:
        result["scenario_index"] = result["scenario_id"].map(scenario_index_from_id)
    if "crane_index" not in result.columns and "crane_id" in result.columns:
        result["crane_index"] = result["crane_id"].map(crane_index_from_id)
    if "crane_i_index" not in result.columns and "crane_i" in result.columns:
        result["crane_i_index"] = result["crane_i"].map(crane_index_from_id)
    if "crane_j_index" not in result.columns and "crane_j" in result.columns:
        result["crane_j_index"] = result["crane_j"].map(crane_index_from_id)
    return result


def _business_scenario_id(row: pd.Series) -> str:
    value = row.get("scenario_id", None)
    if isinstance(value, str) and value.startswith("scenario_"):
        return value
    value = row.get("scenario_uid", None)
    if isinstance(value, str) and value.startswith("scenario_"):
        return value
    return scenario_id_from_index(int(row["scenario_index"]))


def make_windows(
    state_obs: pd.DataFrame,
    state_true: pd.DataFrame,
    crane_static: pd.DataFrame,
    edge_current: pd.DataFrame,
    edge_future_label: pd.DataFrame,
    scenario_table: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, int]:
    """Create train/val/test sliding-window npz files."""

    assert_no_future_leakage(NODE_FEATURE_NAMES, EDGE_FEATURE_NAMES)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dt = float(config["simulation"]["dt"])
    win_cfg = config["windowing"]
    t_in = max(1, int(round(float(win_cfg["input_window_s"]) / dt)))
    h_steps = max(1, int(round(float(win_cfg["prediction_horizon_s"]) / dt)))
    stride = max(1, int(round(float(win_cfg["stride_s"]) / dt)))
    n_max = int(win_cfg["max_cranes"])
    horizon_s = float(win_cfg["prediction_horizon_s"])
    scenario_table = _with_index_columns(scenario_table)
    state_obs = _with_index_columns(state_obs)
    state_true = _with_index_columns(state_true)
    crane_static = _with_index_columns(crane_static)
    edge_current = _with_index_columns(edge_current)
    edge_future_label = _with_index_columns(edge_future_label)

    configured_splits = ["train", "val", "test"]
    if bool(config.get("split", {}).get("add_generalization_test", False)):
        configured_splits.append("generalization")
    samples: dict[str, list[dict[str, Any]]] = {split: [] for split in configured_splits}
    split_by_scenario = {int(row["scenario_index"]): str(row["split"]) for _, row in scenario_table.iterrows()}
    uid_by_scenario = {
        int(row["scenario_index"]): _business_scenario_id(row)
        for _, row in scenario_table.iterrows()
    }
    static_by_scenario = {int(sid): g.set_index("crane_index") for sid, g in crane_static.groupby("scenario_index")}
    obs_by_key = {
        (scenario_key(row), int(row["step"]), crane_key(row)): row
        for _, row in state_obs.iterrows()
    }
    true_by_key = {
        (scenario_key(row), int(row["step"]), crane_key(row)): row
        for _, row in state_true.iterrows()
    }
    edge_by_key = {
        (scenario_key(row), int(row["step"]), int(row["crane_i_index"]), int(row["crane_j_index"])): row
        for _, row in edge_current.iterrows()
    }
    label_main = edge_future_label[edge_future_label["horizon_s"] == horizon_s]
    if label_main.empty:
        label_main = edge_future_label.copy()
    label_by_key = {
        (scenario_key(row), int(row["step"]), int(row["crane_i_index"]), int(row["crane_j_index"])): row
        for _, row in label_main.iterrows()
    }

    for scenario_id, group in state_obs.groupby("scenario_index"):
        sid = int(scenario_id)
        split = split_by_scenario.get(sid, "train")
        steps = sorted(int(x) for x in group["step"].unique())
        if not steps:
            continue
        crane_ids = sorted(int(x) for x in group["crane_index"].unique())[:n_max]
        id_to_idx = {cid: idx for idx, cid in enumerate(crane_ids)}
        static_index = static_by_scenario[sid]
        max_start = max(steps) - t_in - h_steps + 1
        for start in range(min(steps), max_start + 1, stride):
            input_steps = list(range(start, start + t_in))
            future_steps = list(range(start + t_in, start + t_in + h_steps))
            if any(step not in steps for step in input_steps + future_steps):
                continue
            node = np.zeros((t_in, n_max, len(NODE_FEATURE_NAMES)), dtype=np.float32)
            edge = np.zeros((t_in, n_max, n_max, len(EDGE_FEATURE_NAMES)), dtype=np.float32)
            node_mask = np.zeros((n_max,), dtype=bool)
            edge_mask = np.zeros((n_max, n_max), dtype=bool)
            time_mask = np.ones((t_in,), dtype=bool)
            y_traj = np.zeros((h_steps, n_max, len(Y_TRAJ_FEATURE_NAMES)), dtype=np.float32)
            y_risk = np.zeros((n_max, n_max, len(Y_RISK_FEATURE_NAMES)), dtype=np.float32)
            y_dist = np.zeros((n_max, n_max, len(Y_MIN_DISTANCE_FEATURE_NAMES)), dtype=np.float32)

            for crane_id, cidx in id_to_idx.items():
                node_mask[cidx] = True
                for tidx, step in enumerate(input_steps):
                    key = (sid, step, crane_id)
                    if key in obs_by_key:
                        node[tidx, cidx, :] = _node_feature(obs_by_key[key], static_index.loc[crane_id])
                for fidx, step in enumerate(future_steps):
                    key = (sid, step, crane_id)
                    if key in true_by_key:
                        true_row = true_by_key[key]
                        theta = float(true_row["theta"])
                        y_traj[fidx, cidx, :] = [
                            math.sin(theta),
                            math.cos(theta),
                            float(true_row["r"]),
                            float(true_row["h"]),
                        ]
            for i, i_idx in id_to_idx.items():
                for j, j_idx in id_to_idx.items():
                    if i == j:
                        continue
                    edge_mask[i_idx, j_idx] = True
                    for tidx, step in enumerate(input_steps):
                        row = edge_by_key.get((sid, step, i, j))
                        if row is not None:
                            edge[tidx, i_idx, j_idx, :] = [float(row[name]) for name in EDGE_FEATURE_NAMES]
                    label_row = label_by_key.get((sid, start + t_in - 1, i, j))
                    if label_row is not None:
                        y_risk[i_idx, j_idx, :] = [float(label_row[name]) for name in Y_RISK_FEATURE_NAMES]
                        y_dist[i_idx, j_idx, :] = [float(label_row[name]) for name in Y_MIN_DISTANCE_FEATURE_NAMES]
            samples.setdefault(split, []).append(
                {
                    "node_features": node,
                    "edge_features": edge,
                    "node_mask": node_mask,
                    "edge_mask": edge_mask,
                    "time_mask": time_mask,
                    "y_traj": y_traj,
                    "y_risk": y_risk,
                    "y_min_distance": y_dist,
                    "scenario_id": sid,
                    "scenario_business_id": uid_by_scenario.get(sid, scenario_id_from_index(sid)),
                    "scenario_uid": uid_by_scenario.get(sid, scenario_id_from_index(sid)),
                    "window_start_step": start,
                }
            )

    counts: dict[str, int] = {}
    for split, split_samples in samples.items():
        path = output_path / f"{split}_windows.npz"
        if not split_samples:
            _empty_npz(path)
            counts[split] = 0
            continue
        np.savez_compressed(
            path,
            node_features=np.stack([s["node_features"] for s in split_samples]),
            edge_features=np.stack([s["edge_features"] for s in split_samples]),
            node_mask=np.stack([s["node_mask"] for s in split_samples]),
            edge_mask=np.stack([s["edge_mask"] for s in split_samples]),
            time_mask=np.stack([s["time_mask"] for s in split_samples]),
            y_traj=np.stack([s["y_traj"] for s in split_samples]),
            y_risk=np.stack([s["y_risk"] for s in split_samples]),
            y_min_distance=np.stack([s["y_min_distance"] for s in split_samples]),
            node_feature_names=np.array(NODE_FEATURE_NAMES),
            edge_feature_names=np.array(EDGE_FEATURE_NAMES),
            y_traj_feature_names=np.array(Y_TRAJ_FEATURE_NAMES),
            y_risk_feature_names=np.array(Y_RISK_FEATURE_NAMES),
            y_min_distance_feature_names=np.array(Y_MIN_DISTANCE_FEATURE_NAMES),
            scenario_ids=np.array([s["scenario_id"] for s in split_samples], dtype=np.int32),
            scenario_indices=np.array([s["scenario_id"] for s in split_samples], dtype=np.int32),
            scenario_business_ids=np.array([s["scenario_business_id"] for s in split_samples]),
            scenario_uids=np.array([s["scenario_uid"] for s in split_samples]),
            window_start_steps=np.array([s["window_start_step"] for s in split_samples], dtype=np.int32),
        )
        counts[split] = len(split_samples)

    arrays = {
        split: np.load(output_path / f"{split}_windows.npz", allow_pickle=False)["scenario_ids"]
        for split in configured_splits
    }
    validate_named_splits_disjoint(arrays)
    return counts
