from __future__ import annotations

import itertools
import math
from typing import Any

import pandas as pd

from tower_sim.geometry import (
    point_point_distance,
    reconstruct_geometry_from_rows,
    segment_point_distance,
    segment_segment_distance,
)
from tower_sim.ids import crane_id_from_index, crane_index_from_id, crane_key, scenario_id_from_index, scenario_index_from_id

LABEL_COLUMNS = [
    "future_min_d_arm_arm",
    "future_min_d_arm_hook_i_to_j",
    "future_min_d_arm_hook_j_to_i",
    "future_min_d_hook_hook",
    "risk_arm_arm",
    "risk_arm_hook_i_to_j",
    "risk_arm_hook_j_to_i",
    "risk_hook_hook",
    "ttc_label_arm_arm",
    "ttc_label_arm_hook",
    "ttc_label_hook_hook",
]


def _distances(static_i: dict[str, Any], static_j: dict[str, Any], state_i: dict[str, Any], state_j: dict[str, Any]) -> tuple[float, float, float, float]:
    geom_i = reconstruct_geometry_from_rows(static_i, state_i)
    geom_j = reconstruct_geometry_from_rows(static_j, state_j)
    return (
        segment_segment_distance(geom_i.root, geom_i.tip, geom_j.root, geom_j.tip),
        segment_point_distance(geom_i.root, geom_i.tip, geom_j.hook),
        segment_point_distance(geom_j.root, geom_j.tip, geom_i.hook),
        point_point_distance(geom_i.hook, geom_j.hook),
    )


def _geometry_distances(geom_i: Any, geom_j: Any) -> tuple[float, float, float, float]:
    return (
        segment_segment_distance(geom_i.root, geom_i.tip, geom_j.root, geom_j.tip),
        segment_point_distance(geom_i.root, geom_i.tip, geom_j.hook),
        segment_point_distance(geom_j.root, geom_j.tip, geom_i.hook),
        point_point_distance(geom_i.hook, geom_j.hook),
    )


def _first_below(values: list[float], threshold: float, dt: float) -> float:
    for idx, value in enumerate(values, start=1):
        if value < threshold:
            return idx * dt
    return -1.0


def compute_future_labels(
    state_true: pd.DataFrame,
    crane_static: pd.DataFrame,
    dt: float,
    horizons_s: list[float],
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Compute future minimum distances and risk labels from true states."""

    rows: list[dict[str, float | int | str]] = []
    state_true_work = state_true.copy()
    crane_static_work = crane_static.copy()
    if "scenario_index" not in state_true_work.columns:
        state_true_work["scenario_index"] = state_true_work["scenario_id"].map(scenario_index_from_id)
    if "crane_index" not in state_true_work.columns:
        state_true_work["crane_index"] = state_true_work["crane_id"].map(crane_index_from_id)
    if "scenario_index" not in crane_static_work.columns:
        crane_static_work["scenario_index"] = crane_static_work["scenario_id"].map(scenario_index_from_id)
    if "crane_index" not in crane_static_work.columns:
        crane_static_work["crane_index"] = crane_static_work["crane_id"].map(crane_index_from_id)
    state_true_sorted = state_true_work.sort_values(["scenario_index", "step", "crane_index"])
    for scenario_index, s_group in state_true_sorted.groupby("scenario_index"):
        scenario_index_int = int(scenario_index)
        scenario_id = str(s_group["scenario_id"].iloc[0]) if "scenario_id" in s_group else scenario_id_from_index(scenario_index_int)
        c_group = crane_static_work[crane_static_work["scenario_index"] == scenario_index_int]
        static_rows = {
            crane_key(row): row.to_dict()
            for _, row in c_group.iterrows()
        }
        state_by_step: dict[int, dict[int, dict[str, Any]]] = {}
        timestamp_by_step: dict[int, float] = {}
        for _, row in s_group.iterrows():
            step = int(row["step"])
            state_by_step.setdefault(step, {})[crane_key(row)] = row.to_dict()
            timestamp_by_step[step] = float(row["timestamp"])
        sorted_steps = sorted(state_by_step)
        crane_ids = sorted(static_rows)
        geometry_by_step: dict[int, dict[int, Any]] = {}
        for step, states_by_crane in state_by_step.items():
            geometry_by_step[step] = {
                crane_id: reconstruct_geometry_from_rows(static_rows[crane_id], state_row)
                for crane_id, state_row in states_by_crane.items()
                if crane_id in static_rows
            }
        distances_by_pair: dict[tuple[int, int], dict[int, tuple[float, float, float, float]]] = {}
        for i, j in itertools.permutations(crane_ids, 2):
            pair_distances: dict[int, tuple[float, float, float, float]] = {}
            for step in sorted_steps:
                if i in geometry_by_step[step] and j in geometry_by_step[step]:
                    pair_distances[step] = _geometry_distances(geometry_by_step[step][i], geometry_by_step[step][j])
            distances_by_pair[(i, j)] = pair_distances

        label_payloads: dict[tuple[int, float, int, int], tuple[float, float, float, float, float, float, float]] = {}
        for (i, j), pair_distances in distances_by_pair.items():
            for step in sorted_steps:
                for horizon_s in horizons_s:
                    horizon_steps = max(1, int(round(float(horizon_s) / dt)))
                    future_steps = [future for future in sorted_steps if step < future <= step + horizon_steps]
                    if len(future_steps) < horizon_steps or any(future not in pair_distances for future in future_steps):
                        continue
                    distance_window = [pair_distances[future] for future in future_steps]
                    arm_arm = [values[0] for values in distance_window]
                    arm_hook_i_to_j = [values[1] for values in distance_window]
                    arm_hook_j_to_i = [values[2] for values in distance_window]
                    hook_hook = [values[3] for values in distance_window]
                    arm_hook_min = [min(left, right) for left, right in zip(arm_hook_i_to_j, arm_hook_j_to_i, strict=True)]
                    label_payloads[(step, float(horizon_s), i, j)] = (
                        min(arm_arm),
                        min(arm_hook_i_to_j),
                        min(arm_hook_j_to_i),
                        min(hook_hook),
                        _first_below(arm_arm, float(thresholds["d_safe_arm_arm_m"]), dt),
                        _first_below(arm_hook_min, float(thresholds["d_safe_arm_hook_m"]), dt),
                        _first_below(hook_hook, float(thresholds["d_safe_hook_hook_m"]), dt),
                    )

        for step in sorted_steps:
            for horizon_s in horizons_s:
                horizon_s_float = float(horizon_s)
                for i, j in itertools.permutations(crane_ids, 2):
                    payload = label_payloads.get((step, horizon_s_float, i, j))
                    if payload is None:
                        continue
                    (
                        min_arm_arm,
                        min_arm_hook_i_to_j,
                        min_arm_hook_j_to_i,
                        min_hook_hook,
                        ttc_arm_arm,
                        ttc_arm_hook,
                        ttc_hook_hook,
                    ) = payload
                    crane_i_id = str(static_rows[i].get("crane_id", crane_id_from_index(i)))
                    crane_j_id = str(static_rows[j].get("crane_id", crane_id_from_index(j)))
                    rows.append(
                        {
                            "scenario_id": scenario_id,
                            "scenario_uid": scenario_id,
                            "scenario_index": scenario_index_int,
                            "timestamp": timestamp_by_step[step],
                            "step": int(step),
                            "horizon_s": horizon_s_float,
                            "crane_i": crane_i_id,
                            "crane_j": crane_j_id,
                            "crane_i_uid": crane_i_id,
                            "crane_j_uid": crane_j_id,
                            "crane_i_index": int(i),
                            "crane_j_index": int(j),
                            "future_min_d_arm_arm": min_arm_arm,
                            "future_min_d_arm_hook_i_to_j": min_arm_hook_i_to_j,
                            "future_min_d_arm_hook_j_to_i": min_arm_hook_j_to_i,
                            "future_min_d_hook_hook": min_hook_hook,
                            "risk_arm_arm": int(min_arm_arm < float(thresholds["d_safe_arm_arm_m"])),
                            "risk_arm_hook_i_to_j": int(min_arm_hook_i_to_j < float(thresholds["d_safe_arm_hook_m"])),
                            "risk_arm_hook_j_to_i": int(min_arm_hook_j_to_i < float(thresholds["d_safe_arm_hook_m"])),
                            "risk_hook_hook": int(min_hook_hook < float(thresholds["d_safe_hook_hook_m"])),
                            "ttc_label_arm_arm": ttc_arm_arm,
                            "ttc_label_arm_hook": ttc_arm_hook,
                            "ttc_label_hook_hook": ttc_hook_hook,
                        }
                    )
    return pd.DataFrame(rows)
