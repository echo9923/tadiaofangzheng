from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from tower_sim.visualization.i18n_zh import field_label, risk_type_label, view_mode_label
from tower_sim.visualization.schemas import AnimationFrame, RISK_TYPE_SPECS, RiskTypeSpec


DEFAULT_PLOTLY_LAYERS = {
    "radius": True,
    "tasks": True,
    "edges": True,
    "future_risk": True,
    "trail": True,
    "height_text": True,
    "distance_curve": True,
}


def _try_import_plotly():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        return go, make_subplots
    except Exception as exc:
        raise RuntimeError("未安装 Plotly。请运行 `python -m pip install -e \".[visual]\"` 后再使用 2.5D 动画播放器。") from exc


def _try_import_pyplot():
    try:
        cache_dir = Path(os.environ.get("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib_cache")))
        cache_dir.mkdir(parents=True, exist_ok=True)
        for lock_file in cache_dir.glob("*.matplotlib-lock"):
            try:
                lock_file.unlink()
            except OSError:
                pass
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = [
            "Noto Sans SC",
            "Microsoft YaHei",
            "SimHei",
            "DengXian",
            "SimSun",
            "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False
        return plt
    except Exception:
        return None


def _write_placeholder_png(path: Path) -> None:
    path.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
            "1f15c4890000000a49444154789c6360000002000100ffff030000060005"
            "57bfab0000000049454e44ae426082"
        )
    )


def _save_or_return(fig: Any, path: str | Path | None, dpi: int = 140):
    if path is None:
        return fig
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    fig.canvas.draw()
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass
    return output


def hist_plot(df: pd.DataFrame, column: str, path: str | Path, title: str) -> None:
    """Write a histogram plot, falling back to placeholder PNG without matplotlib."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt = _try_import_pyplot()
    if plt is None or column not in df:
        _write_placeholder_png(output)
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    ax.hist(series, bins=40, color="#2f6f9f", alpha=0.82)
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("数量")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def sample_scene_topview(crane_static: pd.DataFrame, geometry_table: pd.DataFrame, path: str | Path) -> None:
    """Plot a sample scenario top view."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt = _try_import_pyplot()
    if plt is None or crane_static.empty or geometry_table.empty:
        _write_placeholder_png(output)
        return
    scenario_id = crane_static["scenario_id"].iloc[0]
    static = crane_static[crane_static["scenario_id"] == scenario_id]
    geom = geometry_table[(geometry_table["scenario_id"] == scenario_id) & (geometry_table["step"] == 0)]
    fig, ax = plt.subplots(figsize=(6, 6))
    for _, row in static.iterrows():
        ax.scatter(row["base_x"], row["base_y"], s=60, label=f"塔吊 {row['crane_id']}")
        circle = plt.Circle((row["base_x"], row["base_y"]), row["max_radius"], fill=False, alpha=0.25)
        ax.add_patch(circle)
    for _, row in geom.iterrows():
        ax.plot([row["root_x"], row["tip_x"]], [row["root_y"], row["tip_y"]], linewidth=2)
        ax.scatter(row["hook_x"], row["hook_y"], marker="x")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.legend(loc="best")
    ax.set_title("示例场景俯视图")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def sample_time_series(state_true: pd.DataFrame, path: str | Path) -> None:
    """Plot theta/r/h time series for a sample scenario and crane."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt = _try_import_pyplot()
    if plt is None or state_true.empty:
        _write_placeholder_png(output)
        return
    scenario_id = state_true["scenario_id"].iloc[0]
    crane_id = state_true[state_true["scenario_id"] == scenario_id]["crane_id"].iloc[0]
    data = state_true[(state_true["scenario_id"] == scenario_id) & (state_true["crane_id"] == crane_id)]
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    for ax, col in zip(axes, ["theta", "r", "h"], strict=True):
        ax.plot(data["timestamp"], data[col], color="#2f6f9f")
        ax.set_ylabel(col)
    axes[-1].set_xlabel("时间 / s")
    fig.suptitle(f"场景 {scenario_id}，塔吊 {crane_id}")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def risk_ratio_plot(edge_future_label: pd.DataFrame, path: str | Path) -> None:
    """Plot risk ratios by scenario."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt = _try_import_pyplot()
    if plt is None or edge_future_label.empty:
        _write_placeholder_png(output)
        return
    data = edge_future_label.copy()
    data["risk_any"] = data[
        ["risk_arm_arm", "risk_arm_hook_i_to_j", "risk_arm_hook_j_to_i", "risk_hook_hook"]
    ].max(axis=1)
    ratios = data.groupby("scenario_id")["risk_any"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(ratios.index.astype(str), ratios.values, color="#547a4a")
    ax.set_xlabel("场景ID")
    ax.set_ylabel("风险比例")
    ax.set_ylim(0, 1)
    ax.set_title("按场景统计的风险比例")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def _crane_lookup(df: pd.DataFrame) -> dict[Any, pd.Series]:
    result: dict[Any, pd.Series] = {}
    for _, row in df.iterrows():
        key = row.get("crane_index", row.get("crane_id"))
        result[key] = row
        result[str(row.get("crane_id", key))] = row
    return result


def _task_point(base: pd.Series, row: pd.Series, prefix: str) -> tuple[float, float] | None:
    import math

    theta_col = f"{prefix}_theta"
    r_col = f"{prefix}_r"
    if theta_col not in row or r_col not in row:
        return None
    return (
        float(base["base_x"]) + float(row[r_col]) * math.cos(float(row[theta_col])),
        float(base["base_y"]) + float(row[r_col]) * math.sin(float(row[theta_col])),
    )


def project_25d(x: float, y: float, z: float = 0.0, z_scale: float = 0.45) -> tuple[float, float]:
    """Project x/y/z into a stable isometric 2.5D drawing plane."""

    xp = float(x) - 0.55 * float(y)
    yp = 0.35 * float(x) + 0.35 * float(y) + float(z_scale) * float(z)
    return xp, yp


def _project_series_point(row: pd.Series, x_col: str, y_col: str, z_col: str | None = None) -> tuple[float, float]:
    z = float(row.get(z_col, 0.0)) if z_col is not None else 0.0
    return project_25d(float(row[x_col]), float(row[y_col]), z)


def _project_task_point(base: pd.Series, row: pd.Series, prefix: str) -> tuple[float, float] | None:
    point = _task_point(base, row, prefix)
    if point is None:
        return None
    height = float(row.get(f"{prefix}_h", row.get("transport_h", 0.0)))
    return project_25d(point[0], point[1], height)


def _project_radius(base_x: float, base_y: float, radius: float, z: float = 0.0) -> tuple[list[float], list[float]]:
    import math

    points = [project_25d(base_x + radius * math.cos(t), base_y + radius * math.sin(t), z) for t in [i * 6.28318530718 / 96 for i in range(97)]]
    return [point[0] for point in points], [point[1] for point in points]


def _id_matches(left: Any, right: Any) -> bool:
    if str(left) == str(right):
        return True
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return False


def _geometry_row(geometry: pd.DataFrame, crane: Any) -> pd.Series | None:
    if geometry.empty:
        return None
    for column in ["crane_id", "crane_index"]:
        if column not in geometry.columns:
            continue
        for _, row in geometry.iterrows():
            if _id_matches(row[column], crane):
                return row
    return None


def _edge_row(edges: pd.DataFrame, pair: tuple[Any, Any] | None) -> pd.Series | None:
    if pair is None or edges.empty or not {"crane_i", "crane_j"}.issubset(edges.columns):
        return None
    for _, row in edges.iterrows():
        if _id_matches(row["crane_i"], pair[0]) and _id_matches(row["crane_j"], pair[1]):
            return row
    return None


def _label_row(labels: pd.DataFrame | None, pair: tuple[Any, Any] | None) -> pd.Series | None:
    if pair is None or labels is None or labels.empty or not {"crane_i", "crane_j"}.issubset(labels.columns):
        return None
    for _, row in labels.iterrows():
        if _id_matches(row["crane_i"], pair[0]) and _id_matches(row["crane_j"], pair[1]):
            return row
    return None


def _risk_positive(row: pd.Series | None) -> bool:
    if row is None:
        return False
    for spec in RISK_TYPE_SPECS.values():
        if spec.risk_column in row and float(row.get(spec.risk_column, 0) or 0) > 0:
            return True
    return False


def _selected_pair_label(pair: tuple[Any, Any] | None) -> str:
    if pair is None:
        return "未选择塔吊对"
    return f"{pair[0]} -> {pair[1]}"


def _sample_steps(steps: list[int], max_frames: int) -> tuple[list[int], bool]:
    if len(steps) <= max_frames:
        return steps, False
    stride = max(1, (len(steps) + max_frames - 1) // max_frames)
    sampled = steps[::stride]
    if steps[-1] not in sampled:
        sampled.append(steps[-1])
    return sampled[:max_frames], True


def _plotly_frame_traces(
    go: Any,
    frame: AnimationFrame,
    *,
    selected_pair: tuple[Any, Any] | None,
    trail: pd.DataFrame,
    edge_series: pd.DataFrame,
    layers: dict[str, bool],
) -> list[Any]:
    traces: list[Any] = []
    static_lookup = _crane_lookup(frame.static)
    selected_ids = {str(selected_pair[0]), str(selected_pair[1])} if selected_pair is not None else set()

    if layers.get("radius", True):
        for _, row in frame.static.iterrows():
            base_x = float(row["base_x"])
            base_y = float(row["base_y"])
            for radius_col, dash in [("max_radius", "solid"), ("min_radius", "dash")]:
                if radius_col not in row:
                    continue
                radius = float(row[radius_col])
                radius_x, radius_y = _project_radius(base_x, base_y, radius, z=0.0)
                traces.append(
                    go.Scatter(
                        x=radius_x,
                        y=radius_y,
                        mode="lines",
                        line={"color": "#9ca3af", "width": 1, "dash": dash},
                        opacity=0.28,
                        name=field_label(radius_col),
                        showlegend=False,
                        xaxis="x",
                        yaxis="y",
                    )
                )

    if layers.get("tasks", True):
        for _, row in frame.tasks.iterrows():
            crane_key = row.get("crane_index", row.get("crane_id"))
            base = static_lookup.get(crane_key)
            if base is None:
                base = static_lookup.get(str(row.get("crane_id", crane_key)))
            if base is None:
                continue
            pickup = _project_task_point(base, row, "pickup")
            dropoff = _project_task_point(base, row, "dropoff")
            if pickup is not None:
                traces.append(go.Scatter(x=[pickup[0]], y=[pickup[1]], mode="markers", marker={"symbol": "triangle-up", "size": 9, "color": "#b7791f"}, name="取货点", showlegend=False, xaxis="x", yaxis="y"))
            if dropoff is not None:
                traces.append(go.Scatter(x=[dropoff[0]], y=[dropoff[1]], mode="markers", marker={"symbol": "square", "size": 8, "color": "#2f855a"}, name="卸货点", showlegend=False, xaxis="x", yaxis="y"))

    if layers.get("trail", True) and not trail.empty:
        for crane_id, data in trail.groupby("crane_id" if "crane_id" in trail.columns else "crane_index"):
            points = [_project_series_point(row, "hook_x", "hook_y", "hook_z") for _, row in data.iterrows()]
            traces.append(
                go.Scatter(
                    x=[point[0] for point in points],
                    y=[point[1] for point in points],
                    mode="lines",
                    line={"color": "#64748b", "width": 1},
                    opacity=0.38,
                    name=f"轨迹 {crane_id}",
                    showlegend=False,
                    xaxis="x",
                    yaxis="y",
                )
            )

    if layers.get("edges", True):
        geometry_lookup = _crane_lookup(frame.geometry)
        for _, row in frame.edges.iterrows():
            left = geometry_lookup.get(row.get("crane_i_index", row.get("crane_i")))
            if left is None:
                left = geometry_lookup.get(str(row.get("crane_i")))
            right = geometry_lookup.get(row.get("crane_j_index", row.get("crane_j")))
            if right is None:
                right = geometry_lookup.get(str(row.get("crane_j")))
            if left is None or right is None:
                continue
            is_selected = selected_pair is not None and _id_matches(row.get("crane_i"), selected_pair[0]) and _id_matches(row.get("crane_j"), selected_pair[1])
            min_distance = min(float(row.get(col, 9999.0)) for col in ["d_arm_arm", "d_arm_hook_i_to_j", "d_arm_hook_j_to_i", "d_hook_hook"])
            color = "#dc2626" if is_selected else "#d97706" if min_distance < 5.0 else "#94a3b8"
            width = 3 if is_selected else 1
            left_hook = _project_series_point(left, "hook_x", "hook_y", "hook_z")
            right_hook = _project_series_point(right, "hook_x", "hook_y", "hook_z")
            traces.append(go.Scatter(x=[left_hook[0], right_hook[0]], y=[left_hook[1], right_hook[1]], mode="lines", line={"color": color, "width": width}, opacity=0.65, name="当前边", showlegend=False, xaxis="x", yaxis="y"))

    if layers.get("future_risk", True) and frame.labels is not None and not frame.labels.empty:
        geometry_lookup = _crane_lookup(frame.geometry)
        for _, row in frame.labels.iterrows():
            if not _risk_positive(row):
                continue
            left = geometry_lookup.get(row.get("crane_i_index", row.get("crane_i")))
            if left is None:
                left = geometry_lookup.get(str(row.get("crane_i")))
            right = geometry_lookup.get(row.get("crane_j_index", row.get("crane_j")))
            if right is None:
                right = geometry_lookup.get(str(row.get("crane_j")))
            if left is None or right is None:
                continue
            left_hook = _project_series_point(left, "hook_x", "hook_y", "hook_z")
            right_hook = _project_series_point(right, "hook_x", "hook_y", "hook_z")
            traces.append(go.Scatter(x=[left_hook[0], right_hook[0]], y=[left_hook[1], right_hook[1]], mode="lines", line={"color": "#be123c", "width": 3, "dash": "dash"}, opacity=0.88, name="未来风险边", showlegend=False, xaxis="x", yaxis="y"))

    for _, row in frame.geometry.iterrows():
        crane_id = row.get("crane_id", row.get("crane_index", ""))
        is_selected = str(crane_id) in selected_ids
        color = "#0f766e" if is_selected else "#2563eb"
        width = 4 if is_selected else 2
        root = _project_series_point(row, "root_x", "root_y", "root_z")
        tip = _project_series_point(row, "tip_x", "tip_y", "tip_z")
        hook = _project_series_point(row, "hook_x", "hook_y", "hook_z")
        hook_top = project_25d(float(row["hook_x"]), float(row["hook_y"]), float(row.get("tip_z", row.get("root_z", 0.0))))
        traces.append(go.Scatter(x=[root[0], tip[0]], y=[root[1], tip[1]], mode="lines", line={"color": color, "width": width}, name=f"吊臂 {crane_id}", showlegend=False, xaxis="x", yaxis="y"))
        traces.append(go.Scatter(x=[hook_top[0], hook[0]], y=[hook_top[1], hook[1]], mode="lines", line={"color": "#475569", "width": 1, "dash": "dot"}, opacity=0.72, name=f"吊钩钢丝 {crane_id}", showlegend=False, xaxis="x", yaxis="y"))
        text = f"{crane_id}<br>h={float(row.get('hook_z', 0.0)):.1f}m" if layers.get("height_text", True) else str(crane_id)
        traces.append(go.Scatter(x=[hook[0]], y=[hook[1]], mode="markers+text", marker={"symbol": "x", "size": 11, "color": "#c2410c"}, text=[text], textposition="top center", name=f"吊钩 {crane_id}", showlegend=False, xaxis="x", yaxis="y"))

    for _, row in frame.static.iterrows():
        crane_id = row.get("crane_id", row.get("crane_index", ""))
        base_z = float(row.get("base_z", 0.0))
        base = project_25d(float(row["base_x"]), float(row["base_y"]), base_z)
        root = project_25d(float(row["base_x"]), float(row["base_y"]), base_z + float(row.get("tower_height", 0.0)))
        traces.append(go.Scatter(x=[base[0], root[0]], y=[base[1], root[1]], mode="lines", line={"color": "#334155", "width": 2}, opacity=0.76, name=f"塔身 {crane_id}", showlegend=False, xaxis="x", yaxis="y"))
        traces.append(go.Scatter(x=[base[0]], y=[base[1]], mode="markers+text", marker={"size": 9, "color": "#1f2937"}, text=[str(crane_id)], textposition="middle right", name=f"基座 {crane_id}", showlegend=False, xaxis="x", yaxis="y"))

    row_i = _geometry_row(frame.geometry, selected_pair[0]) if selected_pair is not None else None
    row_j = _geometry_row(frame.geometry, selected_pair[1]) if selected_pair is not None else None
    if row_i is not None and row_j is not None:
        labels = [str(selected_pair[0]), str(selected_pair[1])]
        root_z = [float(row_i.get("root_z", 0.0)), float(row_j.get("root_z", 0.0))]
        hook_z = [float(row_i.get("hook_z", 0.0)), float(row_j.get("hook_z", 0.0))]
        traces.append(go.Bar(x=labels, y=root_z, marker_color="#4c78a8", name="臂根高度", xaxis="x2", yaxis="y2"))
        traces.append(go.Bar(x=labels, y=hook_z, marker_color="#f58518", name="吊钩高度", xaxis="x2", yaxis="y2"))
    else:
        traces.append(go.Scatter(x=[], y=[], mode="markers", name="高度剖面", xaxis="x2", yaxis="y2"))

    if layers.get("distance_curve", True) and not edge_series.empty:
        x_col = "timestamp" if "timestamp" in edge_series.columns else "step"
        for column, color in zip(["d_arm_arm", "d_arm_hook_i_to_j", "d_arm_hook_j_to_i", "d_hook_hook"], ["#4c78a8", "#f58518", "#54a24b", "#b279a2"], strict=True):
            if column in edge_series.columns:
                traces.append(go.Scatter(x=edge_series[x_col], y=edge_series[column], mode="lines", line={"color": color, "width": 1.5}, name=field_label(column), xaxis="x3", yaxis="y3"))
        traces.append(go.Scatter(x=[frame.timestamp, frame.timestamp], y=[0, max(1.0, float(edge_series[[c for c in ["d_arm_arm", "d_arm_hook_i_to_j", "d_arm_hook_j_to_i", "d_hook_hook"] if c in edge_series]].max().max()))], mode="lines", line={"color": "#111827", "width": 1, "dash": "dot"}, name="当前时刻", showlegend=False, xaxis="x3", yaxis="y3"))
    else:
        traces.append(go.Scatter(x=[], y=[], mode="lines", name="距离曲线", xaxis="x3", yaxis="y3"))
    return traces


def plotly_25d_animation_for_scenario(
    cache: Any,
    *,
    steps: list[int] | None = None,
    current_step: int | None = None,
    selected_pair: tuple[Any, Any] | None = None,
    layers: dict[str, bool] | None = None,
    trail_seconds: float = 20.0,
    max_frames: int = 300,
) -> tuple[Any, bool]:
    """Build a Plotly 2.5D animation figure for one cached scenario."""

    go, make_subplots = _try_import_plotly()
    active_layers = {**DEFAULT_PLOTLY_LAYERS, **(layers or {})}
    source_steps = steps or cache.steps
    sampled_steps, sampled = _sample_steps([int(step) for step in source_steps], max_frames=max_frames)
    if not sampled_steps:
        fig = go.Figure()
        return fig, False
    active_step = cache.nearest_step(current_step) if current_step is not None else sampled_steps[0]
    if active_step not in sampled_steps:
        sampled_steps = [active_step] + sampled_steps
        sampled_steps = sorted(dict.fromkeys(sampled_steps), key=lambda value: cache.steps.index(cache.nearest_step(value)))
    active_index = sampled_steps.index(active_step)
    first_frame = cache.get_frame(active_step)
    pair = selected_pair or cache.strongest_risk_pair(active_step)
    edge_series = cache.edge_series_for_pair(*pair) if pair is not None else pd.DataFrame()

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"rowspan": 2}, {}], [None, {}]],
        column_widths=[0.68, 0.32],
        row_heights=[0.46, 0.54],
        subplot_titles=("2.5D 等轴动画", f"高度剖面：{_selected_pair_label(pair)}", "距离/TTC 曲线"),
    )
    first_traces = _plotly_frame_traces(
        go,
        first_frame,
        selected_pair=pair,
        trail=cache.trail_for_step(active_step, seconds=trail_seconds),
        edge_series=edge_series,
        layers=active_layers,
    )
    for trace in first_traces:
        fig.add_trace(trace)

    frames = []
    for step in sampled_steps:
        frame = cache.get_frame(step)
        frames.append(
            go.Frame(
                name=str(step),
                data=_plotly_frame_traces(
                    go,
                    frame,
                    selected_pair=pair,
                    trail=cache.trail_for_step(step, seconds=trail_seconds),
                    edge_series=edge_series,
                    layers=active_layers,
                ),
                layout={"title": {"text": f"场景 {frame.scenario_id} | 步数 {frame.step} | t={frame.timestamp:.2f}s | {view_mode_label(frame.view_mode)}"}},
            )
        )
    fig.frames = frames
    fig.update_layout(
        title=f"场景 {first_frame.scenario_id} | 步数 {first_frame.step} | t={first_frame.timestamp:.2f}s | {view_mode_label(first_frame.view_mode)}",
        height=760,
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 80, "b": 35},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.08, "xanchor": "left", "x": 0},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": 1.08,
                "buttons": [
                    {"label": "播放", "method": "animate", "args": [None, {"frame": {"duration": 300, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]},
                    {"label": "暂停", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]},
                ],
            }
        ],
        sliders=[
            {
                "active": active_index,
                "currentvalue": {"prefix": "步数 "},
                "steps": [
                    {"label": str(step), "method": "animate", "args": [[str(step)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]}
                    for step in sampled_steps
                ],
            }
        ],
    )
    fig.update_xaxes(title_text="等轴 x / m", row=1, col=1)
    fig.update_yaxes(title_text="等轴 y / m", scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_yaxes(title_text="高度 / m", row=1, col=2)
    fig.update_xaxes(title_text="时间 / s", row=2, col=2)
    fig.update_yaxes(title_text="距离 / m", row=2, col=2)
    return fig, sampled


def plot_topview_frame(
    frame: AnimationFrame,
    path: str | Path | None = None,
    *,
    title: str | None = None,
    dpi: int = 160,
):
    """Draw the 2.5D top view for one animation frame."""

    plt = _try_import_pyplot()
    if plt is None:
        if path is None:
            raise RuntimeError("matplotlib is not available")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_placeholder_png(output)
        return output

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    static_lookup = _crane_lookup(frame.static)
    geometry_lookup = _crane_lookup(frame.geometry)

    for _, row in frame.static.iterrows():
        base_x = float(row["base_x"])
        base_y = float(row["base_y"])
        crane_label = str(row.get("crane_id", row.get("crane_index", "")))
        if "max_radius" in row:
            ax.add_patch(plt.Circle((base_x, base_y), float(row["max_radius"]), fill=False, alpha=0.18, color="#577590"))
        if "min_radius" in row:
            ax.add_patch(
                plt.Circle((base_x, base_y), float(row["min_radius"]), fill=False, alpha=0.28, color="#577590", linestyle="--")
            )
        ax.scatter(base_x, base_y, s=48, color="#243b53", zorder=4)
        ax.text(base_x, base_y, f" {crane_label}", fontsize=8, va="center", color="#1f2933")

    for _, row in frame.tasks.iterrows():
        crane_key = row.get("crane_index", row.get("crane_id"))
        base = static_lookup.get(crane_key)
        if base is None:
            base = static_lookup.get(str(row.get("crane_id", crane_key)))
        if base is None:
            continue
        pickup = _task_point(base, row, "pickup")
        dropoff = _task_point(base, row, "dropoff")
        if pickup is not None:
            ax.scatter(*pickup, marker="^", color="#b7791f", s=34, alpha=0.75, zorder=3)
        if dropoff is not None:
            ax.scatter(*dropoff, marker="s", color="#2f855a", s=30, alpha=0.75, zorder=3)

    for _, row in frame.geometry.iterrows():
        root = (float(row["root_x"]), float(row["root_y"]))
        tip = (float(row["tip_x"]), float(row["tip_y"]))
        hook = (float(row["hook_x"]), float(row["hook_y"]))
        ax.plot([root[0], tip[0]], [root[1], tip[1]], color="#1f77b4", linewidth=2.0, zorder=5)
        ax.scatter(*hook, marker="x", color="#c2410c", s=40, zorder=6)
        ax.plot([tip[0], hook[0]], [tip[1], hook[1]], color="#c2410c", linewidth=0.8, alpha=0.45, zorder=4)

    for _, row in frame.edges.iterrows():
        left = geometry_lookup.get(row.get("crane_i_index", row.get("crane_i")))
        if left is None:
            left = geometry_lookup.get(str(row.get("crane_i")))
        right = geometry_lookup.get(row.get("crane_j_index", row.get("crane_j")))
        if right is None:
            right = geometry_lookup.get(str(row.get("crane_j")))
        if left is None or right is None:
            continue
        min_distance = min(
            float(row.get("d_arm_arm", 9999.0)),
            float(row.get("d_arm_hook_i_to_j", 9999.0)),
            float(row.get("d_arm_hook_j_to_i", 9999.0)),
            float(row.get("d_hook_hook", 9999.0)),
        )
        color = "#9ca3af" if min_distance >= 5.0 else "#d97706"
        ax.plot(
            [float(left["hook_x"]), float(right["hook_x"])],
            [float(left["hook_y"]), float(right["hook_y"])],
            color=color,
            linewidth=0.9,
            alpha=0.45,
            zorder=2,
        )

    if frame.labels is not None and not frame.labels.empty:
        for _, row in frame.labels.iterrows():
            risk_columns = [spec.risk_column for spec in RISK_TYPE_SPECS.values() if spec.risk_column in row]
            if not risk_columns or max(float(row[col]) for col in risk_columns) <= 0:
                continue
            left = geometry_lookup.get(row.get("crane_i_index", row.get("crane_i")))
            if left is None:
                left = geometry_lookup.get(str(row.get("crane_i")))
            right = geometry_lookup.get(row.get("crane_j_index", row.get("crane_j")))
            if right is None:
                right = geometry_lookup.get(str(row.get("crane_j")))
            if left is None or right is None:
                continue
            ax.plot(
                [float(left["hook_x"]), float(right["hook_x"])],
                [float(left["hook_y"]), float(right["hook_y"])],
                color="#be123c",
                linewidth=1.8,
                linestyle="--",
                alpha=0.85,
                zorder=7,
            )

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="#e5e7eb", linewidth=0.6)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title(
        title
        or f"场景 {frame.scenario_id}｜步数 {frame.step}｜t={frame.timestamp:.2f}s｜{view_mode_label(frame.view_mode)}"
    )
    fig.tight_layout()
    return _save_or_return(fig, path, dpi=dpi)


def plot_height_profile(
    frame: AnimationFrame,
    crane_i: Any,
    crane_j: Any,
    path: str | Path | None = None,
    *,
    dpi: int = 160,
):
    """Draw hook and jib-root heights for a selected crane pair."""

    plt = _try_import_pyplot()
    if plt is None:
        if path is None:
            raise RuntimeError("matplotlib is not available")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_placeholder_png(output)
        return output
    lookup = _crane_lookup(frame.geometry)
    row_i = lookup.get(crane_i)
    if row_i is None:
        row_i = lookup.get(str(crane_i))
    row_j = lookup.get(crane_j)
    if row_j is None:
        row_j = lookup.get(str(crane_j))
    rows = [row_i, row_j]
    rows = [row for row in rows if row is not None]
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    labels = []
    root_z = []
    hook_z = []
    for row in rows:
        labels.append(str(row.get("crane_id", row.get("crane_index", ""))))
        root_z.append(float(row.get("root_z", 0.0)))
        hook_z.append(float(row.get("hook_z", 0.0)))
    x = range(len(labels))
    ax.bar([value - 0.16 for value in x], root_z, width=0.32, label="臂根高度", color="#4c78a8")
    ax.bar([value + 0.16 for value in x], hook_z, width=0.32, label="吊钩高度", color="#f58518")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("高度 / m")
    ax.set_title("高度剖面")
    ax.legend(loc="best")
    fig.tight_layout()
    return _save_or_return(fig, path, dpi=dpi)


def plot_25d_frame(
    frame: AnimationFrame,
    path: str | Path | None = None,
    *,
    selected_pair: tuple[Any, Any] | None = None,
    trail: pd.DataFrame | None = None,
    edge_series: pd.DataFrame | None = None,
    layers: dict[str, bool] | None = None,
    dpi: int = 160,
):
    """Draw a static 2.5D frame used by PNG/GIF/MP4 export."""

    active_layers = {**DEFAULT_PLOTLY_LAYERS, **(layers or {})}
    plt = _try_import_pyplot()
    if plt is None:
        if path is None:
            raise RuntimeError("matplotlib is not available")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_placeholder_png(output)
        return output
    fig = plt.figure(figsize=(11, 6.5))
    grid = fig.add_gridspec(2, 2, width_ratios=[2.05, 1.0], height_ratios=[1.0, 1.0])
    ax_top = fig.add_subplot(grid[:, 0])
    ax_height = fig.add_subplot(grid[0, 1])
    ax_curve = fig.add_subplot(grid[1, 1])
    static_lookup = _crane_lookup(frame.static)
    geometry_lookup = _crane_lookup(frame.geometry)
    selected_ids = {str(selected_pair[0]), str(selected_pair[1])} if selected_pair is not None else set()

    if active_layers.get("radius", True):
        for _, row in frame.static.iterrows():
            base_x = float(row["base_x"])
            base_y = float(row["base_y"])
            if "max_radius" in row:
                radius_x, radius_y = _project_radius(base_x, base_y, float(row["max_radius"]), z=0.0)
                ax_top.plot(radius_x, radius_y, color="#64748b", linewidth=0.8, alpha=0.16)
            if "min_radius" in row:
                radius_x, radius_y = _project_radius(base_x, base_y, float(row["min_radius"]), z=0.0)
                ax_top.plot(radius_x, radius_y, color="#64748b", linewidth=0.8, linestyle="--", alpha=0.25)

    if active_layers.get("tasks", True):
        for _, row in frame.tasks.iterrows():
            crane_key = row.get("crane_index", row.get("crane_id"))
            base = static_lookup.get(crane_key)
            if base is None:
                base = static_lookup.get(str(row.get("crane_id", crane_key)))
            if base is None:
                continue
            pickup = _project_task_point(base, row, "pickup")
            dropoff = _project_task_point(base, row, "dropoff")
            if pickup is not None:
                ax_top.scatter(*pickup, marker="^", color="#b7791f", s=28, alpha=0.7)
            if dropoff is not None:
                ax_top.scatter(*dropoff, marker="s", color="#2f855a", s=24, alpha=0.7)

    if active_layers.get("trail", True) and trail is not None and not trail.empty:
        group_col = "crane_id" if "crane_id" in trail.columns else "crane_index"
        for _, data in trail.groupby(group_col):
            points = [_project_series_point(row, "hook_x", "hook_y", "hook_z") for _, row in data.iterrows()]
            ax_top.plot([point[0] for point in points], [point[1] for point in points], color="#64748b", linewidth=1.0, alpha=0.38)

    if active_layers.get("edges", True):
        for _, row in frame.edges.iterrows():
            left = geometry_lookup.get(row.get("crane_i_index", row.get("crane_i")))
            if left is None:
                left = geometry_lookup.get(str(row.get("crane_i")))
            right = geometry_lookup.get(row.get("crane_j_index", row.get("crane_j")))
            if right is None:
                right = geometry_lookup.get(str(row.get("crane_j")))
            if left is None or right is None:
                continue
            is_selected = selected_pair is not None and _id_matches(row.get("crane_i"), selected_pair[0]) and _id_matches(row.get("crane_j"), selected_pair[1])
            min_distance = min(float(row.get(col, 9999.0)) for col in ["d_arm_arm", "d_arm_hook_i_to_j", "d_arm_hook_j_to_i", "d_hook_hook"])
            color = "#dc2626" if is_selected else "#d97706" if min_distance < 5.0 else "#94a3b8"
            left_hook = _project_series_point(left, "hook_x", "hook_y", "hook_z")
            right_hook = _project_series_point(right, "hook_x", "hook_y", "hook_z")
            ax_top.plot([left_hook[0], right_hook[0]], [left_hook[1], right_hook[1]], color=color, linewidth=2.2 if is_selected else 0.9, alpha=0.62)

    if active_layers.get("future_risk", True) and frame.labels is not None and not frame.labels.empty:
        for _, row in frame.labels.iterrows():
            if not _risk_positive(row):
                continue
            left = geometry_lookup.get(row.get("crane_i_index", row.get("crane_i")))
            if left is None:
                left = geometry_lookup.get(str(row.get("crane_i")))
            right = geometry_lookup.get(row.get("crane_j_index", row.get("crane_j")))
            if right is None:
                right = geometry_lookup.get(str(row.get("crane_j")))
            if left is None or right is None:
                continue
            left_hook = _project_series_point(left, "hook_x", "hook_y", "hook_z")
            right_hook = _project_series_point(right, "hook_x", "hook_y", "hook_z")
            ax_top.plot([left_hook[0], right_hook[0]], [left_hook[1], right_hook[1]], color="#be123c", linewidth=2.4, linestyle="--", alpha=0.85)

    for _, row in frame.geometry.iterrows():
        crane_id = row.get("crane_id", row.get("crane_index", ""))
        is_selected = str(crane_id) in selected_ids
        color = "#0f766e" if is_selected else "#2563eb"
        root = _project_series_point(row, "root_x", "root_y", "root_z")
        tip = _project_series_point(row, "tip_x", "tip_y", "tip_z")
        hook = _project_series_point(row, "hook_x", "hook_y", "hook_z")
        hook_top = project_25d(float(row["hook_x"]), float(row["hook_y"]), float(row.get("tip_z", row.get("root_z", 0.0))))
        ax_top.plot([root[0], tip[0]], [root[1], tip[1]], color=color, linewidth=3.2 if is_selected else 1.8)
        ax_top.plot([hook_top[0], hook[0]], [hook_top[1], hook[1]], color="#475569", linewidth=0.8, linestyle=":", alpha=0.72)
        ax_top.scatter(hook[0], hook[1], marker="x", color="#c2410c", s=42)
        label = f"{crane_id}\nh={float(row.get('hook_z', 0.0)):.1f}m" if active_layers.get("height_text", True) else str(crane_id)
        ax_top.text(hook[0], hook[1], label, fontsize=7, color="#1f2937")

    for _, row in frame.static.iterrows():
        base_z = float(row.get("base_z", 0.0))
        base = project_25d(float(row["base_x"]), float(row["base_y"]), base_z)
        root = project_25d(float(row["base_x"]), float(row["base_y"]), base_z + float(row.get("tower_height", 0.0)))
        ax_top.plot([base[0], root[0]], [base[1], root[1]], color="#334155", linewidth=1.6, alpha=0.76)
        ax_top.scatter(base[0], base[1], color="#111827", s=36)
        ax_top.text(base[0], base[1], f" {row.get('crane_id', row.get('crane_index', ''))}", fontsize=8)

    ax_top.set_title("2.5D 等轴动画")
    ax_top.set_xlabel("等轴 x / m")
    ax_top.set_ylabel("等轴 y / m")
    ax_top.set_aspect("equal", adjustable="datalim")
    ax_top.grid(True, color="#e5e7eb", linewidth=0.5)

    row_i = _geometry_row(frame.geometry, selected_pair[0]) if selected_pair is not None else None
    row_j = _geometry_row(frame.geometry, selected_pair[1]) if selected_pair is not None else None
    if row_i is not None and row_j is not None:
        labels = [str(selected_pair[0]), str(selected_pair[1])]
        x = range(len(labels))
        ax_height.bar([v - 0.16 for v in x], [float(row_i.get("root_z", 0.0)), float(row_j.get("root_z", 0.0))], width=0.32, label="臂根高度", color="#4c78a8")
        ax_height.bar([v + 0.16 for v in x], [float(row_i.get("hook_z", 0.0)), float(row_j.get("hook_z", 0.0))], width=0.32, label="吊钩高度", color="#f58518")
        ax_height.set_xticks(list(x), labels)
    ax_height.set_title(f"高度剖面：{_selected_pair_label(selected_pair)}")
    ax_height.set_ylabel("高度 / m")
    if ax_height.get_legend_handles_labels()[0]:
        ax_height.legend(loc="best", fontsize=7)
    ax_height.grid(True, color="#e5e7eb", linewidth=0.5)

    if active_layers.get("distance_curve", True) and edge_series is not None and not edge_series.empty:
        data = edge_series.sort_values("timestamp" if "timestamp" in edge_series.columns else "step")
        x_col = "timestamp" if "timestamp" in data.columns else "step"
        for column, color in zip(["d_arm_arm", "d_arm_hook_i_to_j", "d_arm_hook_j_to_i", "d_hook_hook"], ["#4c78a8", "#f58518", "#54a24b", "#b279a2"], strict=True):
            if column in data.columns:
                ax_curve.plot(data[x_col], data[column], label=field_label(column), color=color, linewidth=1.2)
        ax_curve.axvline(frame.timestamp if x_col == "timestamp" else frame.step, color="#111827", linestyle=":", linewidth=1.0)
    ax_curve.set_title("距离/TTC 曲线")
    ax_curve.set_xlabel("时间 / s")
    ax_curve.set_ylabel("距离 / m")
    if ax_curve.get_legend_handles_labels()[0]:
        ax_curve.legend(loc="best", fontsize=6)
    ax_curve.grid(True, color="#e5e7eb", linewidth=0.5)

    fig.suptitle(f"场景 {frame.scenario_id} | 步数 {frame.step} | t={frame.timestamp:.2f}s | {view_mode_label(frame.view_mode)}")
    fig.tight_layout()
    return _save_or_return(fig, path, dpi=dpi)


def plot_distance_curve(
    edge_series: pd.DataFrame,
    risk_spec: RiskTypeSpec | str,
    path: str | Path | None = None,
    *,
    current_timestamp: float | None = None,
    horizon_s: float | None = None,
    input_window_s: float | None = None,
    threshold: float | None = None,
    dpi: int = 160,
):
    """Draw risk-inspector distance curves for one directed crane pair."""

    spec = RISK_TYPE_SPECS[risk_spec] if isinstance(risk_spec, str) else risk_spec
    plt = _try_import_pyplot()
    if plt is None:
        if path is None:
            raise RuntimeError("matplotlib is not available")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_placeholder_png(output)
        return output
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    if not edge_series.empty:
        data = edge_series.sort_values("timestamp" if "timestamp" in edge_series.columns else "step")
        x_col = "timestamp" if "timestamp" in data.columns else "step"
        distance_columns = [
            "d_arm_arm",
            "d_arm_hook_i_to_j",
            "d_arm_hook_j_to_i",
            "d_hook_hook",
        ]
        colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
        for column, color in zip(distance_columns, colors, strict=True):
            if column in data.columns:
                linewidth = 2.2 if column == spec.current_distance_column else 1.0
                alpha = 1.0 if column == spec.current_distance_column else 0.55
                ax.plot(data[x_col], data[column], label=field_label(column), linewidth=linewidth, alpha=alpha, color=color)
    if current_timestamp is not None:
        ax.axvline(current_timestamp, color="#111827", linewidth=1.2, linestyle="-", label="当前时刻")
        if input_window_s is not None:
            ax.axvspan(current_timestamp - input_window_s, current_timestamp, color="#60a5fa", alpha=0.12, label="输入窗口")
        if horizon_s is not None:
            ax.axvspan(current_timestamp, current_timestamp + horizon_s, color="#fb7185", alpha=0.10, label="标签窗口")
    if threshold is not None:
        ax.axhline(float(threshold), color="#dc2626", linestyle="--", linewidth=1.0, label="安全阈值")
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("距离 / m")
    ax.set_title(f"距离曲线：{risk_type_label(spec.risk_type)}")
    ax.grid(True, color="#e5e7eb", linewidth=0.6)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return _save_or_return(fig, path, dpi=dpi)
