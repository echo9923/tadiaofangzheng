from pathlib import Path
import uuid

from tower_sim.visualization.animation_player import (
    PlaybackState,
    advance_playback,
    initialize_playback_state,
    next_step_by_speed,
    playback_tick,
)
from tower_sim.visualization.export import export_risk_clip_gif, export_risk_clip_mp4
from tower_sim.visualization.frame_cache import FrameCache
from tower_sim.visualization.loaders import RunDataRepository
from tower_sim.visualization.plotting import plotly_25d_animation_for_scenario
from tower_sim.visualization.risk_events import extract_risk_events
from tower_sim.visualization.schemas import ViewMode


def _repo() -> RunDataRepository:
    return RunDataRepository(Path(__file__).resolve().parents[1] / "outputs" / "debug_small", prefer_format="csv")


def _cache(view_mode: str = "debug") -> FrameCache:
    return FrameCache(_repo().load_scenario_data(0, view_mode=view_mode), view_mode=view_mode)


def test_playback_state_advances_with_loop_and_stops_at_end() -> None:
    cache = _cache()
    state = initialize_playback_state(cache, 0, "debug", speed=1.0)

    next_state = advance_playback(cache, state, direction=1)
    assert next_state.step == cache.steps[1]

    end_state = PlaybackState(0, cache.steps[-1], 1.0, True, False, ViewMode.DEBUG)
    next_step, keep_playing = next_step_by_speed(cache, end_state, direction=1)
    assert next_step == cache.steps[-1]
    assert keep_playing is False

    loop_state = PlaybackState(0, cache.steps[-1], 1.0, True, True, ViewMode.DEBUG)
    looped = playback_tick(cache, loop_state)
    assert looped.step == cache.steps[0]
    assert looped.playing is True


def test_frame_cache_pairs_trails_and_strongest_pair() -> None:
    cache = _cache()

    pairs = cache.available_pairs(0)
    assert pairs
    assert cache.strongest_risk_pair(0) in pairs

    trail = cache.trail_for_step(cache.steps[min(2, len(cache.steps) - 1)], seconds=20.0)
    assert {"hook_x", "hook_y", "tip_x", "tip_y"}.issubset(trail.columns)
    assert trail["step"].max() <= cache.steps[min(2, len(cache.steps) - 1)]

    series = cache.edge_series_for_pair(*pairs[0])
    assert not series.empty
    assert {"d_arm_arm", "d_hook_hook"}.issubset(series.columns)


def test_plotly_25d_animation_has_controls_and_no_input_label_leakage() -> None:
    cache = _cache("input")
    fig, sampled = plotly_25d_animation_for_scenario(cache, steps=cache.steps[:5], max_frames=5)

    assert sampled is False
    assert fig.frames
    assert fig.layout.updatemenus
    labels = [button["label"] for button in fig.layout.updatemenus[0].buttons]
    assert "播放" in labels
    assert "暂停" in labels
    subplot_titles = [annotation.text for annotation in fig.layout.annotations]
    assert any("高度剖面" in title for title in subplot_titles)
    trace_names = [trace.name for trace in fig.data if getattr(trace, "name", None)]
    assert any("轨迹" in name for name in trace_names)
    text = fig.to_json()
    assert "edge_future_label" not in text
    assert "future_min_d" not in text
    assert "ttc_label" not in text
    assert "risk_arm_arm" not in text


def test_risk_clip_gif_and_mp4_export() -> None:
    repo = _repo()
    cache = _cache()
    labels = repo.load_table("edge_future_label").copy()
    labels.loc[0, "risk_arm_arm"] = 1
    labels.loc[0, "future_min_d_arm_arm"] = 0.25
    events = extract_risk_events(labels, risk_types=["arm_arm"])
    event = events[0]
    output_dir = Path(__file__).resolve().parents[1] / "test_artifacts" / f"animation_export_{uuid.uuid4().hex}"

    gif_path = export_risk_clip_gif(cache, event, output_dir / f"{event.scenario_id}_{event.step}_{event.risk_type}.gif", pre_seconds=0.0, post_seconds=0.0, fps=2)
    mp4_path = export_risk_clip_mp4(cache, event, output_dir / f"{event.scenario_id}_{event.step}_{event.risk_type}.mp4", pre_seconds=0.0, post_seconds=0.0, fps=2)

    assert gif_path.exists()
    assert gif_path.stat().st_size > 0
    assert str(event.scenario_id) in gif_path.name
    assert str(event.step) in gif_path.name
    assert event.risk_type in gif_path.name
    assert mp4_path.exists()
    assert mp4_path.stat().st_size > 0
