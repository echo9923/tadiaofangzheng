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
    clip_start_step: int | None = None
    clip_end_step: int | None = None
    selected_pair: tuple[Any, Any] | None = None


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


def initialize_playback_state(
    cache: FrameCache,
    scenario_id: Any,
    view_mode: str | ViewMode,
    *,
    speed: float = 1.0,
    loop: bool = False,
    step: int | None = None,
    selected_pair: tuple[Any, Any] | None = None,
) -> PlaybackState:
    mode = ViewMode.parse(view_mode)
    selected_step = cache.nearest_step(step) if step is not None and cache.steps else cache.steps[0] if cache.steps else 0
    return PlaybackState(
        scenario_id=scenario_id,
        step=selected_step,
        speed=float(speed),
        playing=False,
        loop=bool(loop),
        view_mode=mode,
        selected_pair=selected_pair or cache.strongest_risk_pair(selected_step),
    )


def _active_steps(cache: FrameCache, state: PlaybackState) -> list[int]:
    steps = cache.steps
    if state.clip_start_step is None and state.clip_end_step is None:
        return steps
    start = state.clip_start_step if state.clip_start_step is not None else min(steps)
    end = state.clip_end_step if state.clip_end_step is not None else max(steps)
    return [step for step in steps if int(start) <= step <= int(end)]


def next_step_by_speed(cache: FrameCache, state: PlaybackState, direction: int = 1) -> tuple[int, bool]:
    """Return the next step and whether playback should keep running."""

    if not cache.steps:
        return state.step, False
    steps = _active_steps(cache, state) or cache.steps
    current = min(steps, key=lambda step: abs(step - cache.nearest_step(state.step)))
    index = steps.index(current)
    jump = max(1, int(round(abs(float(state.speed)))))
    next_index = index + (jump * (1 if direction >= 0 else -1))
    if state.loop:
        return steps[next_index % len(steps)], True
    if next_index < 0:
        return steps[0], False
    if next_index >= len(steps):
        return steps[-1], False
    return steps[next_index], state.playing


def advance_playback(cache: FrameCache, state: PlaybackState, direction: int = 1) -> PlaybackState:
    next_step, keep_playing = next_step_by_speed(cache, state, direction=direction)
    return PlaybackState(
        scenario_id=state.scenario_id,
        step=next_step,
        speed=state.speed,
        playing=keep_playing if state.playing else state.playing,
        loop=state.loop,
        view_mode=state.view_mode,
        clip_start_step=state.clip_start_step,
        clip_end_step=state.clip_end_step,
        selected_pair=state.selected_pair,
    )


def playback_tick(cache: FrameCache, state: PlaybackState) -> PlaybackState:
    if not state.playing:
        return state
    return advance_playback(cache, state, direction=1)


def clip_steps_for_risk_event(
    cache: FrameCache,
    event: Any,
    *,
    pre_seconds: float = 10.0,
    post_seconds: float = 10.0,
) -> PlaybackState:
    clip = cache.clip_around_event(event, pre_seconds=pre_seconds, post_seconds=post_seconds)
    start_step = clip.steps[0] if clip.steps else clip.start_step
    end_step = clip.steps[-1] if clip.steps else clip.end_step
    selected_pair = (getattr(event, "crane_i", None), getattr(event, "crane_j", None))
    if selected_pair == (None, None):
        selected_pair = cache.strongest_risk_pair(getattr(event, "step", start_step))
    return PlaybackState(
        scenario_id=getattr(event, "scenario_id", cache.scenario_data.scenario_id),
        step=start_step,
        speed=1.0,
        playing=True,
        loop=False,
        view_mode=cache.view_mode,
        clip_start_step=start_step,
        clip_end_step=end_step,
        selected_pair=selected_pair,
    )
