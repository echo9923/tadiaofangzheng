from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tower_sim.visualization.schemas import AnimationFrame, RISK_TYPE_SPECS, ScenarioData, ViewMode


@dataclass(frozen=True)
class ClipWindow:
    start_step: int
    end_step: int
    steps: list[int]


class FrameCache:
    """Build and cache animation frames for one scenario."""

    def __init__(self, scenario_data: ScenarioData, view_mode: str | ViewMode = ViewMode.DEBUG) -> None:
        self.scenario_data = scenario_data
        self.view_mode = ViewMode.parse(view_mode)
        state_source = scenario_data.state_obs if self.view_mode in {ViewMode.INPUT, ViewMode.DEBUG} else scenario_data.state_true
        self._state_source = state_source
        step_source = scenario_data.geometry if not scenario_data.geometry.empty else state_source
        self.steps = sorted(int(step) for step in step_source["step"].dropna().unique()) if "step" in step_source else []
        self._cache: dict[int, AnimationFrame] = {}

    def nearest_step(self, value: int | float) -> int:
        if not self.steps:
            raise ValueError("Scenario has no animation steps")
        numeric = int(round(float(value)))
        return min(self.steps, key=lambda step: abs(step - numeric))

    def step_at_or_before_timestamp(self, timestamp: float) -> int:
        if self._state_source.empty or "timestamp" not in self._state_source.columns:
            return self.nearest_step(timestamp)
        candidates = self._state_source[["step", "timestamp"]].drop_duplicates().sort_values("timestamp")
        before = candidates[pd.to_numeric(candidates["timestamp"], errors="coerce") <= float(timestamp)]
        if before.empty:
            return int(candidates.iloc[0]["step"])
        return int(before.iloc[-1]["step"])

    def get_frame(self, step: int | float) -> AnimationFrame:
        selected_step = self.nearest_step(step)
        if selected_step in self._cache:
            return self._cache[selected_step]
        data = self.scenario_data
        state = self._state_source[self._state_source["step"] == selected_step].copy()
        geometry = data.geometry[data.geometry["step"] == selected_step].copy()
        edges = data.edge_current[data.edge_current["step"] == selected_step].copy()
        labels = None
        if self.view_mode is not ViewMode.INPUT and data.edge_future_label is not None:
            labels = data.edge_future_label[data.edge_future_label["step"] == selected_step].copy()
        timestamp = 0.0
        if not state.empty and "timestamp" in state.columns:
            timestamp = float(state["timestamp"].iloc[0])
        elif not geometry.empty and "timestamp" in geometry.columns:
            timestamp = float(geometry["timestamp"].iloc[0])
        frame = AnimationFrame(
            scenario_id=data.scenario_id,
            step=selected_step,
            timestamp=timestamp,
            view_mode=self.view_mode,
            state=state,
            geometry=geometry,
            edges=edges,
            labels=labels,
            tasks=data.task_table.copy(),
            static=data.crane_static.copy(),
        )
        self._cache[selected_step] = frame
        return frame

    def available_pairs(self, step: int | float | None = None) -> list[tuple[Any, Any]]:
        """Return directed crane-pair options from edge_current."""

        edges = self.scenario_data.edge_current
        if edges.empty or not {"crane_i", "crane_j"}.issubset(edges.columns):
            return []
        if step is not None and "step" in edges.columns:
            selected_step = self.nearest_step(step)
            edges = edges[edges["step"] == selected_step]
        pairs = []
        seen: set[tuple[str, str]] = set()
        for _, row in edges[["crane_i", "crane_j"]].drop_duplicates().iterrows():
            pair = (row["crane_i"], row["crane_j"])
            key = (str(pair[0]), str(pair[1]))
            if key not in seen:
                pairs.append(pair)
                seen.add(key)
        return pairs

    def edge_series_for_pair(self, crane_i: Any, crane_j: Any) -> pd.DataFrame:
        """Return all edge_current rows for one directed pair."""

        edges = self.scenario_data.edge_current
        if edges.empty or not {"crane_i", "crane_j"}.issubset(edges.columns):
            return edges.iloc[0:0].copy()
        result = _filter_id_value(edges, "crane_i", crane_i)
        result = _filter_id_value(result, "crane_j", crane_j)
        if "timestamp" in result.columns:
            return result.sort_values("timestamp").copy()
        return result.sort_values("step").copy() if "step" in result.columns else result.copy()

    def trail_for_step(self, step: int | float, seconds: float = 20.0) -> pd.DataFrame:
        """Return hook/tip trajectory rows before and including the selected step."""

        geometry = self.scenario_data.geometry
        if geometry.empty or "step" not in geometry.columns:
            return geometry.iloc[0:0].copy()
        selected_step = self.nearest_step(step)
        if "timestamp" in geometry.columns:
            current = geometry[geometry["step"] == selected_step]
            if not current.empty:
                timestamp = float(pd.to_numeric(current["timestamp"], errors="coerce").dropna().iloc[0])
                start = timestamp - float(seconds)
                numeric_time = pd.to_numeric(geometry["timestamp"], errors="coerce")
                return geometry[(numeric_time >= start) & (numeric_time <= timestamp)].copy()
        timestamps = self._state_source[["step", "timestamp"]].drop_duplicates() if "timestamp" in self._state_source else pd.DataFrame()
        dt = 1.0
        if not timestamps.empty:
            diffs = pd.to_numeric(timestamps["timestamp"], errors="coerce").sort_values().diff().dropna()
            if not diffs.empty:
                dt = max(float(diffs.median()), 1e-9)
        pre_steps = int(round(float(seconds) / dt))
        start = max(min(self.steps), selected_step - pre_steps)
        return geometry[(geometry["step"] >= start) & (geometry["step"] <= selected_step)].copy()

    def strongest_risk_pair(self, step: int | float | None = None) -> tuple[Any, Any] | None:
        """Return the highest-priority risk pair, otherwise the closest current pair."""

        selected_step = self.nearest_step(step if step is not None else self.steps[0]) if self.steps else None
        if selected_step is None:
            return None
        if self.view_mode is not ViewMode.INPUT and self.scenario_data.edge_future_label is not None:
            labels = self.scenario_data.edge_future_label
            if not labels.empty and "step" in labels.columns:
                labels = labels[labels["step"] == selected_step]
                risk_columns = [spec.risk_column for spec in RISK_TYPE_SPECS.values() if spec.risk_column in labels.columns]
                if risk_columns and {"crane_i", "crane_j"}.issubset(labels.columns):
                    risk_values = labels[risk_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
                    risky = labels[risk_values.max(axis=1) > 0].copy()
                    if not risky.empty:
                        distance_columns = [spec.future_distance_column for spec in RISK_TYPE_SPECS.values() if spec.future_distance_column in risky.columns]
                        if distance_columns:
                            distances = risky[distance_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
                            risky["_risk_sort_distance"] = distances.min(axis=1)
                            risky = risky.sort_values("_risk_sort_distance", na_position="last")
                        row = risky.iloc[0]
                        return row["crane_i"], row["crane_j"]
        edges = self.scenario_data.edge_current
        if edges.empty or not {"crane_i", "crane_j"}.issubset(edges.columns):
            return None
        if "step" in edges.columns:
            edges = edges[edges["step"] == selected_step].copy()
        distance_columns = [spec.current_distance_column for spec in RISK_TYPE_SPECS.values() if spec.current_distance_column in edges.columns]
        if distance_columns and not edges.empty:
            distances = edges[distance_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            edges["_risk_sort_distance"] = distances.min(axis=1)
            edges = edges.sort_values("_risk_sort_distance", na_position="last")
        if edges.empty:
            pairs = self.available_pairs(selected_step)
            return pairs[0] if pairs else None
        row = edges.iloc[0]
        return row["crane_i"], row["crane_j"]

    def clip_around_step(self, center_step: int, pre_steps: int = 10, post_steps: int = 10) -> ClipWindow:
        if not self.steps:
            return ClipWindow(center_step, center_step, [])
        start = max(min(self.steps), int(center_step) - int(pre_steps))
        end = min(max(self.steps), int(center_step) + int(post_steps))
        steps = [step for step in self.steps if start <= step <= end]
        return ClipWindow(start, end, steps)

    def clip_around_event(self, event: Any, pre_seconds: float = 10.0, post_seconds: float = 10.0) -> ClipWindow:
        if self._state_source.empty:
            return self.clip_around_step(int(event.step), int(pre_seconds), int(post_seconds))
        timestamps = self._state_source[["step", "timestamp"]].drop_duplicates()
        if len(timestamps) < 2:
            return self.clip_around_step(int(event.step), int(pre_seconds), int(post_seconds))
        dt = float(pd.to_numeric(timestamps["timestamp"], errors="coerce").sort_values().diff().dropna().median())
        dt = dt if dt > 0 else 1.0
        return self.clip_around_step(int(event.step), int(round(pre_seconds / dt)), int(round(post_seconds / dt)))


def _filter_id_value(df: pd.DataFrame, column: str, value: Any) -> pd.DataFrame:
    if column not in df.columns:
        return df.iloc[0:0].copy()
    numeric = pd.to_numeric(df[column], errors="coerce")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        numeric_value = None
    if numeric_value is not None and not numeric.isna().all():
        matched = df[numeric == numeric_value]
        if not matched.empty:
            return matched
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return df[df[column].astype(str).isin({str(value), text})]
