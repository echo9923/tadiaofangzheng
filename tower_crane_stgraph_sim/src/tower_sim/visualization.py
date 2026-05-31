from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


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

        return plt
    except Exception:
        return None


def _write_placeholder_png(path: Path) -> None:
    # 1x1 transparent PNG
    path.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
            "1f15c4890000000a49444154789c6360000002000100ffff030000060005"
            "57bfab0000000049454e44ae426082"
        )
    )


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
    ax.hist(series, bins=40, color="#1f77b4", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("count")
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
        ax.scatter(row["base_x"], row["base_y"], s=60, label=f"crane {row['crane_id']}")
        circle = plt.Circle((row["base_x"], row["base_y"]), row["max_radius"], fill=False, alpha=0.25)
        ax.add_patch(circle)
    for _, row in geom.iterrows():
        ax.plot([row["root_x"], row["tip_x"]], [row["root_y"], row["tip_y"]], linewidth=2)
        ax.scatter(row["hook_x"], row["hook_y"], marker="x")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.legend(loc="best")
    ax.set_title("Sample scene top view")
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
        ax.plot(data["timestamp"], data[col])
        ax.set_ylabel(col)
    axes[-1].set_xlabel("time / s")
    fig.suptitle(f"Scenario {scenario_id}, crane {crane_id}")
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
    ax.bar(ratios.index.astype(str), ratios.values)
    ax.set_xlabel("scenario_id")
    ax.set_ylabel("risk ratio")
    ax.set_ylim(0, 1)
    ax.set_title("Risk ratio by scenario")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)
