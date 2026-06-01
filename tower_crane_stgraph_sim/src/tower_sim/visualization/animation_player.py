from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tower_sim.visualization.frame_cache import ClipWindow, FrameCache
from tower_sim.visualization.i18n_zh import view_mode_label
from tower_sim.visualization.loaders import RunDataRepository
from tower_sim.visualization.schemas import AnimationFrame, ViewMode, assert_input_columns_safe


@dataclass(frozen=True)
class PlaybackState:
    scenario_id: Any
    step: int
    speed: float
    playing: bool
    loop: bool
    view_mode: ViewMode


def create_frame_cache(repository: RunDataRepository, scenario_id: Any, view_mode: str | ViewMode) -> FrameCache:
    mode = ViewMode.parse(view_mode)
    data = repository.load_scenario_data(scenario_id, view_mode=mode)
    return FrameCache(data, view_mode=mode)


def get_animation_frame(repository: RunDataRepository, scenario_id: Any, step: int, view_mode: str | ViewMode = ViewMode.DEBUG) -> AnimationFrame:
    return create_frame_cache(repository, scenario_id, view_mode).get_frame(step)


def view_mode_warning(view_mode: str | ViewMode) -> str:
    mode = ViewMode.parse(view_mode)
    if mode is ViewMode.INPUT:
        return "输入视图只使用 state_obs 和 edge_current，不会加载或显示任何未来标签。"
    if mode is ViewMode.LABEL:
        return "标签视图展示来自 edge_future_label 的未来标签证据；这些字段只用于验收，不是模型输入。"
    return f"{view_mode_label(mode)}会叠加当前输入和未来标签，仅用于论文解释与调试验收。"


def assert_frame_respects_view_mode(frame: AnimationFrame) -> None:
    if frame.view_mode is ViewMode.INPUT and frame.labels is not None:
        raise ValueError("输入视图帧不能包含 edge_future_label 行")
    if frame.view_mode is ViewMode.INPUT:
        assert_input_columns_safe(list(frame.state.columns), context="Input View state columns")
        assert_input_columns_safe(list(frame.edges.columns), context="Input View edge columns")


def risk_clip_for_event(cache: FrameCache, event: Any, pre_seconds: float = 10.0, post_seconds: float = 10.0) -> ClipWindow:
    return cache.clip_around_event(event, pre_seconds=pre_seconds, post_seconds=post_seconds)


def advance_playback(cache: FrameCache, state: PlaybackState, direction: int = 1) -> PlaybackState:
    if not cache.steps:
        return state
    current = cache.nearest_step(state.step)
    index = cache.steps.index(current)
    next_index = index + int(direction)
    if state.loop:
        next_step = cache.steps[next_index % len(cache.steps)]
    else:
        next_step = cache.steps[max(0, min(len(cache.steps) - 1, next_index))]
    return PlaybackState(
        scenario_id=state.scenario_id,
        step=next_step,
        speed=state.speed,
        playing=state.playing,
        loop=state.loop,
        view_mode=state.view_mode,
    )
