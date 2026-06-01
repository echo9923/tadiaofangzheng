from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tower_sim.visualization.frame_cache import FrameCache
from tower_sim.visualization.i18n_zh import field_label, localize_dict_keys
from tower_sim.visualization.plotting import plot_distance_curve, plot_topview_frame
from tower_sim.visualization.risk_events import RiskEvent, current_edge_row_for_event, edge_series_for_event, explain_risk_event
from tower_sim.visualization.schemas import RISK_TYPE_SPECS
from tower_sim.visualization.window_view import WindowSampleView


def export_frame_png(frame: Any, output_path: str | Path, *, dpi: int = 300) -> Path:
    return Path(plot_topview_frame(frame, output_path, dpi=dpi))


def export_risk_explanation_package(
    event: RiskEvent,
    edge_current,
    output_dir: str | Path,
    *,
    input_window_s: float | None = None,
) -> dict[str, Path]:
    """Export a Chinese risk distance chart, Markdown explanation, and CSV evidence."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = RISK_TYPE_SPECS[event.risk_type]
    edge_series = edge_series_for_event(edge_current, event)
    chart_path = out / f"{event.scenario_id}_{event.step}_{event.risk_type}_distance.png"
    csv_path = out / f"{event.scenario_id}_{event.step}_{event.risk_type}_edge_series.csv"
    markdown_path = out / f"{event.scenario_id}_{event.step}_{event.risk_type}_explanation.md"
    plot_distance_curve(
        edge_series,
        spec,
        chart_path,
        current_timestamp=event.timestamp,
        horizon_s=event.horizon_s,
        input_window_s=input_window_s,
        threshold=event.threshold,
        dpi=300,
    )
    edge_series.to_csv(csv_path, index=False, encoding="utf-8")
    current_row = current_edge_row_for_event(edge_current, event)
    markdown_path.write_text(explain_risk_event(event, current_row), encoding="utf-8")
    return {"chart": chart_path, "markdown": markdown_path, "csv": csv_path}


def export_window_sample_summary(sample: WindowSampleView, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "数据划分": sample.split,
        "样本索引": sample.sample_index,
        "场景ID": str(sample.scenario_id),
        "场景索引": sample.scenario_index,
        "窗口起始步数": sample.window_start_step,
        "张量形状": {field_label(key): list(value) for key, value in sample.shapes.items()},
        "风险正标签数量": sample.risk_positive_count,
        "有效节点数": int(sample.masks.get("node_mask", []).sum()) if "node_mask" in sample.masks else None,
        "有效边数": int(sample.masks.get("edge_mask", []).sum()) if "edge_mask" in sample.masks else None,
        "有效时间步数": int(sample.masks.get("time_mask", []).sum()) if "time_mask" in sample.masks else None,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def export_quality_summary_markdown(metrics: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# 群塔吊可视化质量摘要", ""]
    localized = localize_dict_keys(metrics)
    for key in sorted(localized):
        value = localized[key]
        if isinstance(value, dict):
            lines.append(f"- {key}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}")
        elif isinstance(value, list):
            lines.append(f"- {key}: {len(value)} 项")
        else:
            lines.append(f"- {key}: {value}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def export_risk_clip_frames(cache: FrameCache, event: RiskEvent, output_dir: str | Path, *, pre_seconds: float = 10.0, post_seconds: float = 10.0) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    clip = cache.clip_around_event(event, pre_seconds=pre_seconds, post_seconds=post_seconds)
    paths: list[Path] = []
    for step in clip.steps:
        frame = cache.get_frame(step)
        path = out / f"frame_{step:06d}.png"
        export_frame_png(frame, path)
        paths.append(path)
    return paths
