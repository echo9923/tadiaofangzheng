from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class VisualizationSettings:
    default_run_root: str = "outputs/debug_small"
    prefer_format: str = "parquet"
    max_points_per_curve: int = 2000
    default_speed: float = 1.0
    available_speeds: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0)
    risk_clip_pre_seconds: float = 10.0
    risk_clip_post_seconds: float = 10.0
    export_dir: str = "exports"


def load_visualization_settings(path: str | Path) -> VisualizationSettings:
    config_path = Path(path)
    if not config_path.exists():
        return VisualizationSettings()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    visualization = data.get("visualization", data)
    playback = visualization.get("playback", {})
    export = visualization.get("export", {})
    return VisualizationSettings(
        default_run_root=str(visualization.get("default_run_root", VisualizationSettings.default_run_root)),
        prefer_format=str(visualization.get("prefer_format", VisualizationSettings.prefer_format)),
        max_points_per_curve=int(visualization.get("max_points_per_curve", VisualizationSettings.max_points_per_curve)),
        default_speed=float(playback.get("default_speed", VisualizationSettings.default_speed)),
        available_speeds=tuple(float(value) for value in playback.get("available_speeds", VisualizationSettings.available_speeds)),
        risk_clip_pre_seconds=float(playback.get("risk_clip_pre_seconds", VisualizationSettings.risk_clip_pre_seconds)),
        risk_clip_post_seconds=float(playback.get("risk_clip_post_seconds", VisualizationSettings.risk_clip_post_seconds)),
        export_dir=str(export.get("output_dir", VisualizationSettings.export_dir)),
    )
