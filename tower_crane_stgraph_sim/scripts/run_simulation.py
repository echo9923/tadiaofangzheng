from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tower_sim.config import apply_overrides, load_config
from tower_sim.simulation import simulate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tower crane ST-graph simulation data.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file.")
    parser.add_argument("--num-scenarios", type=int, default=None, help="Override simulation.num_scenarios.")
    parser.add_argument("--output-dir", default=None, help="Override project.output_dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.num_scenarios, args.output_dir)
    mpl_cache = Path(config["project"]["output_dir"]) / ".matplotlib_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache.resolve()))
    output_dir, stats = simulate_dataset(config)
    print(f"Output directory: {output_dir}")
    print(f"Tables directory: {output_dir / 'tables'}")
    print(f"Windows directory: {output_dir / 'windows'}")
    print(f"Quality report: {output_dir / 'quality' / 'quality_report.md'}")
    print(f"Scenarios: {stats['num_scenarios']}")
    print(f"Tasks: {stats['num_tasks']}")
    print(f"Overall risk ratio: {stats['risk_any_ratio']:.6f}")
    print(f"NaN in required fields: {stats['has_nan_required']}")
    print(f"Future leakage detected: {stats['future_leakage']}")


if __name__ == "__main__":
    main()
