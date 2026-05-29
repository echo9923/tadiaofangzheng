from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tower_sim.config import load_config
from tower_sim.quality import generate_quality_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate quality_report.md and diagnostic plots.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    config = load_config(args.config)
    mpl_cache = data_dir / ".matplotlib_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache.resolve()))
    stats = generate_quality_report(
        data_dir,
        pd.read_csv(data_dir / "scenario_table.csv"),
        pd.read_csv(data_dir / "crane_static.csv"),
        pd.read_csv(data_dir / "task_table.csv"),
        pd.read_csv(data_dir / "state_true.csv"),
        pd.read_csv(data_dir / "state_obs.csv"),
        pd.read_csv(data_dir / "edge_current.csv"),
        pd.read_csv(data_dir / "edge_future_label.csv"),
        config,
    )
    print(f"Quality stats: {stats}")


if __name__ == "__main__":
    main()
