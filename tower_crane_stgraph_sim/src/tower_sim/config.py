from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


def _simple_yaml_load(text: str) -> dict[str, Any]:
    """Small YAML fallback supporting the project's simple mapping/list config."""

    result: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, result)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value_text = stripped.split(":", 1)
        key = key.strip()
        value_text = value_text.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value_text == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value_text)
    return result


def _parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("'\"")


def _simple_yaml_dump(data: dict[str, Any], indent: int = 0) -> str:
    lines: list[str] = []
    pad = " " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            lines.append(_simple_yaml_dump(value, indent + 2).rstrip())
        else:
            lines.append(f"{pad}{key}: {_format_yaml_scalar(value)}")
    return "\n".join(lines) + "\n"


def _format_yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ", ".join(_format_yaml_scalar(item) for item in value) + "]"
    return str(value)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml

        loaded = yaml.safe_load(text)
    except ModuleNotFoundError:
        loaded = _simple_yaml_load(text)
    return loaded or {}


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save the effective configuration to YAML."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        output_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    except ModuleNotFoundError:
        output_path.write_text(_simple_yaml_dump(config), encoding="utf-8")


def apply_overrides(
    config: dict[str, Any],
    num_scenarios: int | None = None,
    output_dir: str | None = None,
    duration: float | None = None,
    seed: int | None = None,
    dt: float | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Return a deep-copied config with CLI overrides applied."""

    cfg = copy.deepcopy(config)
    if num_scenarios is not None:
        cfg.setdefault("simulation", {})["num_scenarios"] = int(num_scenarios)
    if output_dir is not None:
        cfg.setdefault("project", {})["output_dir"] = output_dir
    if duration is not None:
        cfg.setdefault("simulation", {})["scenario_duration_s"] = float(duration)
    if seed is not None:
        cfg.setdefault("project", {})["random_seed"] = int(seed)
    if dt is not None:
        cfg.setdefault("simulation", {})["dt"] = float(dt)
    if run_id is not None:
        cfg.setdefault("project", {})["run_id"] = run_id
    return cfg


def get_stage_tolerance(task_cfg: dict[str, Any]) -> tuple[float, float, float]:
    """Return task-stage tolerances as ``(theta_rad, r_m, h_m)``."""

    tol = task_cfg["stage_tolerance"]
    if isinstance(tol, dict):
        return float(tol["theta_rad"]), float(tol["r_m"]), float(tol["h_m"])
    if isinstance(tol, (list, tuple)):
        if len(tol) != 3:
            raise ValueError("stage_tolerance list must contain theta_rad, r_m, and h_m")
        return float(tol[0]), float(tol[1]), float(tol[2])
    scalar = float(tol)
    return scalar, scalar, scalar


def sample_ratio(rng: Any, value: Any) -> float:
    """Read a scalar ratio or sample uniformly from a two-value ratio range."""

    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("Ratio range must contain exactly two values")
        return float(rng.uniform(float(value[0]), float(value[1])))
    return float(value)


def get_command_smoothing(controller_cfg: dict[str, Any]) -> float:
    """Parse command smoothing as a numeric alpha or boolean feature toggle."""

    value = controller_cfg.get("command_smoothing", 0.0)
    if isinstance(value, bool):
        if not value:
            return 0.0
        return _clip_unit_interval(float(controller_cfg.get("command_smoothing_alpha", 0.2)))
    if isinstance(value, dict):
        if not bool(value.get("enabled", True)):
            return 0.0
        return _clip_unit_interval(float(value.get("alpha", 0.2)))
    return _clip_unit_interval(float(value))


def _clip_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, value))


def get_range(config: dict[str, Any], path: str) -> tuple[float, float]:
    """Read a two-value range from a dotted config path."""

    value = get_by_path(config, path)
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"Config value {path!r} must be a two-value range")
    return float(value[0]), float(value[1])


def get_by_path(config: dict[str, Any], path: str, default: Any = None) -> Any:
    """Read a dotted-path value from nested dictionaries."""

    cur: Any = config
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def scenario_seed(base_seed: int, scenario_id: int, stream: int = 0) -> int:
    """Derive a deterministic seed for a scenario and stream."""

    return int((base_seed + 1000003 * scenario_id + 9176 * stream) % (2**32 - 1))
