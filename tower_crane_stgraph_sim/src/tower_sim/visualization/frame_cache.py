from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from tower_sim.visualization.schemas import AnimationFrame, ScenarioData, ViewMode


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
