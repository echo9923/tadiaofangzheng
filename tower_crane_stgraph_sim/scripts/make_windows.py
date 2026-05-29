from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tower_sim.config import load_config
from tower_sim.io_utils import dataset_paths
from tower_sim.windowing import make_windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild train/val/test window npz files from CSV tables.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    paths = dataset_paths(data_dir)
    tables_dir = paths.tables if paths.tables.exists() else data_dir
    windows_dir = paths.windows if paths.tables.exists() else data_dir
    config = load_config(args.config)
    counts = make_windows(
        pd.read_csv(tables_dir / "state_obs.csv"),
        pd.read_csv(tables_dir / "state_true.csv"),
        pd.read_csv(tables_dir / "crane_static.csv"),
        pd.read_csv(tables_dir / "edge_current.csv"),
        pd.read_csv(tables_dir / "edge_future_label.csv"),
        pd.read_csv(tables_dir / "scenario_table.csv"),
        config,
        windows_dir,
    )
    print(f"Window counts: {counts}")


if __name__ == "__main__":
    main()
