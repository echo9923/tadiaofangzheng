from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from tower_sim.visualization.i18n_zh import field_label, risk_type_label, view_mode_label
from tower_sim.visualization.schemas import AnimationFrame, RISK_TYPE_SPECS, RiskTypeSpec


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
