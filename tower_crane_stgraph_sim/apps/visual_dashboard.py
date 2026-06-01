from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tower_sim.visualization.animation_player import create_frame_cache, view_mode_warning
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
from tower_sim.visualization.plotting import plot_distance_curve, plot_topview_frame
from tower_sim.visualization.quality_view import load_quality_view
from tower_sim.visualization.risk_events import current_edge_row_for_event, edge_series_for_event, explain_risk_event, extract_risk_events
from tower_sim.visualization.ui_state import load_visualization_settings
from tower_sim.visualization.window_view import get_window_sample, load_window_bundle, summarize_window_bundle


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
    page = st.sidebar.radio(
        "页面",
        page_keys,
        format_func=page_label,
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
    scenario_id = st.sidebar.selectbox("场景", scenarios["scenario_id"].tolist())

    if page == "Animation Player":
        mode = st.sidebar.selectbox("视图模式", ["input", "label", "debug"], index=2, format_func=view_mode_label)
        cache = create_frame_cache(repo, scenario_id, mode)
        if not cache.steps:
            st.warning("当前场景没有可播放步数。")
            return
        step = st.slider("步数", min_value=min(cache.steps), max_value=max(cache.steps), value=min(cache.steps))
        st.info(view_mode_warning(mode))
        frame = cache.get_frame(step)
        fig = plot_topview_frame(frame)
        st.pyplot(fig, clear_figure=True)
        cols = st.columns(2)
        cols[0].subheader("当前状态")
        cols[0].caption(UNTRANSLATED_FIELD_NOTE)
        cols[0].dataframe(rename_columns_for_display(frame.state), use_container_width=True)
        cols[1].subheader("当前边特征")
        cols[1].caption(UNTRANSLATED_FIELD_NOTE)
        cols[1].dataframe(rename_columns_for_display(frame.edges), use_container_width=True)
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
