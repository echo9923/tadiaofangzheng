from __future__ import annotations

import sys
from pathlib import Path
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tower_sim.visualization.animation_player import (
    PlaybackState,
    advance_playback,
    create_frame_cache,
    initialize_playback_state,
    playback_tick,
    view_mode_warning,
)
from tower_sim.visualization.export import export_frame_png, export_risk_clip_frames, export_risk_clip_gif, export_risk_clip_mp4
from tower_sim.visualization.i18n_zh import (
    APP_TITLE,
    PAGE_LABELS,
    UNTRANSLATED_FIELD_NOTE,
    localize_dict_keys,
    localize_records_frame,
    page_label,
    rename_columns_for_display,
    risk_type_label,
    split_label,
    tensor_shape_summary,
    view_mode_label,
)
from tower_sim.visualization.indexing import build_scenario_index, summarize_run
from tower_sim.visualization.loaders import RunDataRepository
from tower_sim.visualization.plotting import DEFAULT_PLOTLY_LAYERS, plot_distance_curve, plotly_25d_animation_for_scenario
from tower_sim.visualization.quality_view import load_quality_view
from tower_sim.visualization.risk_events import current_edge_row_for_event, edge_series_for_event, explain_risk_event, extract_risk_events
from tower_sim.visualization.schemas import ViewMode
from tower_sim.visualization.ui_state import load_visualization_settings
from tower_sim.visualization.window_view import get_window_sample, load_window_bundle, summarize_window_bundle


def _pair_label(pair: tuple[object, object] | None) -> str:
    return "未选择塔吊对" if pair is None else f"{pair[0]} -> {pair[1]}"


def _state_to_session(st, state: PlaybackState) -> None:
    st.session_state.animation_state = state


def _session_state_for(st, cache, scenario_id, mode, settings) -> PlaybackState:
    state = st.session_state.get("animation_state")
    parsed_mode = ViewMode.parse(mode)
    if (
        state is None
        or str(getattr(state, "scenario_id", "")) != str(scenario_id)
        or getattr(state, "view_mode", None) != parsed_mode
    ):
        state = initialize_playback_state(cache, scenario_id, mode, speed=settings.default_speed)
        _state_to_session(st, state)
    return state


def _with_state(state: PlaybackState, **changes) -> PlaybackState:
    data = {
        "scenario_id": state.scenario_id,
        "step": state.step,
        "speed": state.speed,
        "playing": state.playing,
        "loop": state.loop,
        "view_mode": state.view_mode,
        "clip_start_step": state.clip_start_step,
        "clip_end_step": state.clip_end_step,
        "selected_pair": state.selected_pair,
    }
    data.update(changes)
    return PlaybackState(**data)


def _streamlit_main() -> None:
    import streamlit as st

    settings = load_visualization_settings(PROJECT_ROOT / "configs" / "visualization.yaml")
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    run_root_text = st.sidebar.text_input("运行目录", value=str(PROJECT_ROOT / settings.default_run_root))
    repo = RunDataRepository(run_root_text, prefer_format=settings.prefer_format)
    missing = repo.validate(required_tables=["scenario_table", "crane_static", "state_obs", "geometry_table", "edge_current"])
    if missing:
        st.error("缺少必要文件：\n" + "\n".join(f"- {item}" for item in missing))
        return

    page_keys = list(PAGE_LABELS)
    page_override = st.session_state.pop("page_key_override", None)
    if page_override in page_keys:
        st.session_state.page_key = page_override
    page = st.sidebar.radio(
        "页面",
        page_keys,
        format_func=page_label,
        key="page_key",
    )

    if page == "Run Dashboard":
        summary = summarize_run(repo)
        cols = st.columns(4)
        cols[0].metric("场景数", summary.get("num_scenarios", 0))
        cols[1].metric("总体风险比例", "无" if summary.get("risk_any_ratio") is None else f"{summary['risk_any_ratio']:.3f}")
        cols[2].metric("数据划分数", len(summary.get("split_counts", {})))
        cols[3].metric("场景类型数", len(summary.get("scene_type_counts", {})))
        st.subheader("场景索引")
        st.caption(UNTRANSLATED_FIELD_NOTE)
        st.dataframe(localize_records_frame(build_scenario_index(repo)), use_container_width=True)
        return

    scenarios = repo.list_scenarios()
    scenario_options = scenarios["scenario_id"].tolist()
    scenario_override = st.session_state.pop("scenario_id_override", None)
    if scenario_override in scenario_options:
        st.session_state.scenario_id_select = scenario_override
    scenario_id = st.sidebar.selectbox("场景", scenario_options, key="scenario_id_select")

    if page == "Animation Player":
        mode = st.sidebar.selectbox("视图模式", ["input", "label", "debug"], index=2, format_func=view_mode_label)
        cache = create_frame_cache(repo, scenario_id, mode)
        if not cache.steps:
            st.warning("当前场景没有可播放步数。")
            return
        state = _session_state_for(st, cache, scenario_id, mode, settings)
        if "animation_jump" in st.session_state:
            state = st.session_state.pop("animation_jump")
            _state_to_session(st, state)
        pair_options = cache.available_pairs(state.step)
        if state.selected_pair not in pair_options:
            state = _with_state(state, selected_pair=cache.strongest_risk_pair(state.step) or (pair_options[0] if pair_options else None))
            _state_to_session(st, state)
        step_min = state.clip_start_step if state.clip_start_step is not None else min(cache.steps)
        step_max = state.clip_end_step if state.clip_end_step is not None else max(cache.steps)
        active_steps = [step for step in cache.steps if step_min <= step <= step_max]
        st.info(view_mode_warning(mode))
        st.caption("输入视图不会加载未来标签；标签/调试视图中的未来标签只用于验收解释。")

        with st.sidebar.expander("播放控制", expanded=True):
            speed = st.selectbox("播放速度", list(settings.available_speeds), index=list(settings.available_speeds).index(state.speed) if state.speed in settings.available_speeds else 0, format_func=lambda value: f"{value:g}x")
            loop = st.checkbox("循环播放", value=state.loop)
            selected_step = st.slider("当前步数", min_value=min(cache.steps), max_value=max(cache.steps), value=cache.nearest_step(state.step))
            timestamp_value = st.number_input("跳转到时间/s", value=float(cache.get_frame(selected_step).timestamp), step=1.0)
            if st.button("跳转时间"):
                selected_step = cache.step_at_or_before_timestamp(timestamp_value)
            state = _with_state(state, speed=float(speed), loop=loop, step=selected_step)
            cols = st.columns(4)
            if cols[0].button("播放"):
                state = _with_state(state, playing=True)
            if cols[1].button("暂停"):
                state = _with_state(state, playing=False)
            if cols[2].button("上一帧"):
                state = advance_playback(cache, _with_state(state, playing=False), direction=-1)
            if cols[3].button("下一帧"):
                state = advance_playback(cache, _with_state(state, playing=False), direction=1)

        with st.sidebar.expander("塔吊对与图层", expanded=True):
            selected_pair = st.selectbox("选中塔吊对", pair_options, index=pair_options.index(state.selected_pair) if state.selected_pair in pair_options else 0, format_func=_pair_label) if pair_options else None
            layers = {
                "radius": st.checkbox("作业半径", value=DEFAULT_PLOTLY_LAYERS["radius"]),
                "tasks": st.checkbox("任务点", value=DEFAULT_PLOTLY_LAYERS["tasks"]),
                "edges": st.checkbox("当前边", value=DEFAULT_PLOTLY_LAYERS["edges"]),
                "future_risk": st.checkbox("未来风险边", value=DEFAULT_PLOTLY_LAYERS["future_risk"]),
                "trail": st.checkbox("轨迹尾迹", value=DEFAULT_PLOTLY_LAYERS["trail"]),
                "height_text": st.checkbox("高度文本", value=DEFAULT_PLOTLY_LAYERS["height_text"]),
                "distance_curve": st.checkbox("距离/TTC 曲线", value=DEFAULT_PLOTLY_LAYERS["distance_curve"]),
            }
            state = _with_state(state, selected_pair=selected_pair)

        config = repo.load_config()
        events = extract_risk_events(repo.load_table("edge_future_label"), config=config) if mode != "input" else []
        scenario_events = [event for event in events if str(event.scenario_id) == str(scenario_id)]
        with st.sidebar.expander("风险跳转与导出", expanded=True):
            if scenario_events:
                event_labels = [
                    f"步数 {event.step} | {event.crane_i}->{event.crane_j} | {risk_type_label(event.risk_type)}"
                    for event in scenario_events
                ]
                event_index = st.selectbox("风险事件", range(len(scenario_events)), format_func=lambda idx: event_labels[idx])
                selected_event = scenario_events[event_index]
                if st.button("跳到风险事件"):
                    state = _with_state(state, step=selected_event.step, selected_pair=(selected_event.crane_i, selected_event.crane_j), playing=False)
                if st.button("播放风险片段"):
                    clip = cache.clip_around_event(selected_event, pre_seconds=settings.risk_clip_pre_seconds, post_seconds=settings.risk_clip_post_seconds)
                    state = _with_state(
                        state,
                        step=clip.steps[0] if clip.steps else selected_event.step,
                        clip_start_step=clip.steps[0] if clip.steps else clip.start_step,
                        clip_end_step=clip.steps[-1] if clip.steps else clip.end_step,
                        selected_pair=(selected_event.crane_i, selected_event.crane_j),
                        playing=True,
                    )
            else:
                selected_event = None
                st.caption("当前视图没有可跳转的风险事件。")

            export_root = Path(run_root_text) / settings.export_dir / "animation_player"
            if st.button("导出当前帧 PNG"):
                frame_for_export = cache.get_frame(state.step)
                path = export_root / f"{scenario_id}_{state.step:06d}_25d.png"
                export_frame_png(
                    frame_for_export,
                    path,
                    selected_pair=state.selected_pair,
                    trail=cache.trail_for_step(state.step, seconds=20.0),
                    edge_series=cache.edge_series_for_pair(*state.selected_pair) if state.selected_pair else None,
                    layers=layers,
                )
                st.success(f"已导出：{path}")
            if selected_event is not None:
                if st.button("导出风险片段 PNG 序列"):
                    paths = export_risk_clip_frames(cache, selected_event, export_root / f"{scenario_id}_{selected_event.step}_{selected_event.risk_type}_frames", pre_seconds=settings.risk_clip_pre_seconds, post_seconds=settings.risk_clip_post_seconds, layers=layers)
                    st.success(f"已导出 {len(paths)} 帧：{paths[0].parent if paths else export_root}")
                if st.button("导出风险片段 GIF"):
                    path = export_risk_clip_gif(cache, selected_event, export_root / f"{scenario_id}_{selected_event.step}_{selected_event.risk_type}.gif", pre_seconds=settings.risk_clip_pre_seconds, post_seconds=settings.risk_clip_post_seconds, layers=layers)
                    st.success(f"已导出：{path}")
                if st.button("导出风险片段 MP4"):
                    path = export_risk_clip_mp4(cache, selected_event, export_root / f"{scenario_id}_{selected_event.step}_{selected_event.risk_type}.mp4", pre_seconds=settings.risk_clip_pre_seconds, post_seconds=settings.risk_clip_post_seconds, layers=layers)
                    st.success(f"已导出：{path}")

        state = playback_tick(cache, state)
        _state_to_session(st, state)
        st.info(view_mode_warning(mode))
        fig, sampled = plotly_25d_animation_for_scenario(
            cache,
            steps=active_steps,
            selected_pair=state.selected_pair,
            layers=layers,
            trail_seconds=20.0,
            max_frames=300,
        )
        if sampled:
            st.warning("当前场景帧数较多，已为性能抽样到最多 300 帧。")
        st.plotly_chart(fig, use_container_width=True)
        frame = cache.get_frame(state.step)
        metric_cols = st.columns(4)
        metric_cols[0].metric("当前步数", state.step)
        metric_cols[1].metric("当前时间/s", f"{frame.timestamp:.2f}")
        metric_cols[2].metric("播放状态", "播放中" if state.playing else "已暂停")
        metric_cols[3].metric("选中塔吊对", _pair_label(state.selected_pair))
        detail_cols = st.columns(2)
        detail_cols[0].subheader("选中塔吊状态")
        detail_cols[0].caption(UNTRANSLATED_FIELD_NOTE)
        if state.selected_pair:
            selected_cranes = {str(state.selected_pair[0]), str(state.selected_pair[1])}
            state_rows = frame.state[frame.state["crane_id"].astype(str).isin(selected_cranes)] if "crane_id" in frame.state else frame.state
        else:
            state_rows = frame.state
        detail_cols[0].dataframe(rename_columns_for_display(state_rows), use_container_width=True)
        detail_cols[1].subheader("选中边详情")
        detail_cols[1].caption(UNTRANSLATED_FIELD_NOTE)
        if state.selected_pair and {"crane_i", "crane_j"}.issubset(frame.edges.columns):
            edge_rows = frame.edges[
                (frame.edges["crane_i"].astype(str) == str(state.selected_pair[0]))
                & (frame.edges["crane_j"].astype(str) == str(state.selected_pair[1]))
            ]
            if frame.labels is not None and not frame.labels.empty and {"crane_i", "crane_j"}.issubset(frame.labels.columns):
                label_rows = frame.labels[
                    (frame.labels["crane_i"].astype(str) == str(state.selected_pair[0]))
                    & (frame.labels["crane_j"].astype(str) == str(state.selected_pair[1]))
                ]
                edge_rows = edge_rows.merge(label_rows, how="left", on=[col for col in ["scenario_id", "timestamp", "step", "crane_i", "crane_j"] if col in edge_rows.columns and col in label_rows.columns], suffixes=("", "_label"))
        else:
            edge_rows = frame.edges
        detail_cols[1].dataframe(rename_columns_for_display(edge_rows), use_container_width=True)
        if state.playing:
            time.sleep(max(0.05, 0.8 / max(float(state.speed), 0.1)))
            st.rerun()
        return

    if page == "Risk Inspector":
        config = repo.load_config()
        events = extract_risk_events(repo.load_table("edge_future_label"), config=config)
        if not events:
            st.warning("没有找到风险为正的事件。")
            return
        labels = [
            f"场景 {event.scenario_id}｜步数 {event.step}｜{event.crane_i}->{event.crane_j}｜{risk_type_label(event.risk_type)}"
            for event in events
        ]
        event = events[st.sidebar.selectbox("风险事件", range(len(events)), format_func=lambda idx: labels[idx])]
        scenario_data = repo.load_scenario_data(event.scenario_id, view_mode="debug")
        current_row = current_edge_row_for_event(scenario_data.edge_current, event)
        st.markdown(explain_risk_event(event, current_row))
        series = edge_series_for_event(scenario_data.edge_current, event)
        if st.button("在动画播放器中播放该事件"):
            event_cache = create_frame_cache(repo, event.scenario_id, "debug")
            clip = event_cache.clip_around_event(event, pre_seconds=settings.risk_clip_pre_seconds, post_seconds=settings.risk_clip_post_seconds)
            st.session_state.animation_jump = PlaybackState(
                scenario_id=event.scenario_id,
                step=clip.steps[0] if clip.steps else event.step,
                speed=settings.default_speed,
                playing=True,
                loop=False,
                view_mode=ViewMode.DEBUG,
                clip_start_step=clip.steps[0] if clip.steps else clip.start_step,
                clip_end_step=clip.steps[-1] if clip.steps else clip.end_step,
                selected_pair=(event.crane_i, event.crane_j),
            )
            st.session_state.scenario_id_override = event.scenario_id
            st.session_state.page_key_override = "Animation Player"
            st.rerun()
        fig = plot_distance_curve(
            series,
            event.risk_type,
            current_timestamp=event.timestamp,
            horizon_s=event.horizon_s,
            threshold=event.threshold,
        )
        st.pyplot(fig, clear_figure=True)
        return

    if page == "Window Explorer":
        split = st.sidebar.selectbox("数据划分", ["train", "val", "test", "generalization"], format_func=split_label)
        path = repo.windows_dir / f"{split}_windows.npz"
        if not path.exists():
            st.warning(f"缺少窗口文件：{path}")
            return
        bundle = load_window_bundle(path, split=split)
        summary = summarize_window_bundle(bundle)
        st.subheader("窗口文件摘要")
        st.json(localize_dict_keys(summary))
        if "shapes" in summary:
            st.subheader("张量形状")
            st.dataframe(tensor_shape_summary(summary["shapes"]), use_container_width=True)
        if bundle.num_samples:
            sample_index = st.slider("样本索引", 0, bundle.num_samples - 1, 0)
            sample = get_window_sample(bundle, sample_index)
            st.subheader("样本摘要")
            st.json(
                localize_dict_keys(
                    {
                        "scenario_id": str(sample.scenario_id),
                        "scenario_index": sample.scenario_index,
                        "window_start_step": sample.window_start_step,
                        "risk_positive_count": sample.risk_positive_count,
                    }
                )
            )
            st.dataframe(tensor_shape_summary(sample.shapes), use_container_width=True)
        return

    if page == "Quality View":
        quality = load_quality_view(repo)
        st.subheader("质量摘要")
        st.json(localize_dict_keys(quality.dashboard_metrics))
        st.subheader("按场景统计的风险比例")
        st.caption(UNTRANSLATED_FIELD_NOTE)
        st.dataframe(localize_records_frame(quality.risk_ratio_by_scenario), use_container_width=True)
        st.subheader("异常场景")
        st.dataframe(localize_records_frame(quality.anomalous_scenarios), use_container_width=True)
        report_path = repo.quality_dir / "quality_report.md"
        st.subheader("原始质量报告")
        if quality.report_markdown:
            st.info(f"原始英文质量报告文件：{report_path}")
        else:
            st.warning(f"未找到原始质量报告：{report_path}")
        for path in quality.plot_paths:
            st.image(str(path), caption=path.name)
        return

    if page == "Data Browser":
        table_name = st.sidebar.selectbox(
            "数据表",
            ["scenario_table", "crane_static", "task_table", "state_true", "state_obs", "geometry_table", "edge_current", "edge_future_label"],
        )
        data = repo.load_table(table_name)
        if table_name != "scenario_table":
            data = data[data["scenario_id"].astype(str) == str(scenario_id)] if "scenario_id" in data else data
        if table_name == "edge_future_label":
            st.warning("edge_future_label 包含未来标签，不能作为模型输入。")
        st.caption(UNTRANSLATED_FIELD_NOTE)
        st.dataframe(rename_columns_for_display(data), use_container_width=True)


def main() -> None:
    try:
        import streamlit  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "未安装 Streamlit。请先运行 `pip install streamlit`，然后执行 "
            "`streamlit run apps/visual_dashboard.py`."
        ) from exc
    _streamlit_main()


if __name__ == "__main__":
    main()
