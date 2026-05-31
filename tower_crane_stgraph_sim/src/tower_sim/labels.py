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
        for step in sorted_steps:
            for horizon_s in horizons_s:
                horizon_steps = max(1, int(round(float(horizon_s) / dt)))
                future_steps = [future for future in sorted_steps if step < future <= step + horizon_steps]
                if len(future_steps) < horizon_steps:
                    continue
                for i, j in itertools.permutations(crane_ids, 2):
                    min_arm_arm = math.inf
                    min_arm_hook_i_to_j = math.inf
                    min_arm_hook_j_to_i = math.inf
                    min_hook_hook = math.inf
                    ttc_arm_arm = -1.0
                    ttc_arm_hook = -1.0
                    ttc_hook_hook = -1.0
                    for future in future_steps:
                        if i not in state_by_step[future] or j not in state_by_step[future]:
                            min_arm_arm = math.inf
                            break
                        d_arm_arm, d_arm_hook_i_to_j, d_arm_hook_j_to_i, d_hook_hook = _distances(
                            static_rows[i],
                            static_rows[j],
                            state_by_step[future][i],
                            state_by_step[future][j],
                        )
                        elapsed = (future - step) * dt
                        if d_arm_arm < min_arm_arm:
                            min_arm_arm = d_arm_arm
                        if d_arm_hook_i_to_j < min_arm_hook_i_to_j:
                            min_arm_hook_i_to_j = d_arm_hook_i_to_j
                        if d_arm_hook_j_to_i < min_arm_hook_j_to_i:
                            min_arm_hook_j_to_i = d_arm_hook_j_to_i
                        if d_hook_hook < min_hook_hook:
                            min_hook_hook = d_hook_hook
                        if ttc_arm_arm < 0.0 and d_arm_arm < float(thresholds["d_safe_arm_arm_m"]):
                            ttc_arm_arm = elapsed
                        if ttc_arm_hook < 0.0 and min(d_arm_hook_i_to_j, d_arm_hook_j_to_i) < float(
                            thresholds["d_safe_arm_hook_m"]
                        ):
                            ttc_arm_hook = elapsed
                        if ttc_hook_hook < 0.0 and d_hook_hook < float(thresholds["d_safe_hook_hook_m"]):
                            ttc_hook_hook = elapsed
                    if math.isinf(min_arm_arm) or math.isinf(min_arm_hook_i_to_j) or math.isinf(min_arm_hook_j_to_i) or math.isinf(min_hook_hook):
                        continue
                    crane_i_id = str(static_rows[i].get("crane_id", crane_id_from_index(i)))
                    crane_j_id = str(static_rows[j].get("crane_id", crane_id_from_index(j)))
                    rows.append(
                        {
                            "scenario_id": scenario_id,
                            "scenario_uid": scenario_id,
                            "scenario_index": scenario_index_int,
                            "timestamp": timestamp_by_step[step],
                            "step": int(step),
                            "horizon_s": float(horizon_s),
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
