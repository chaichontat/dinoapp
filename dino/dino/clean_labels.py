from __future__ import annotations

import argparse
import fcntl
from datetime import datetime
from pathlib import Path

import pandas as pd

from dino.artifacts import write_csv_atomic


def main() -> None:
    p = argparse.ArgumentParser(description="Normalize BrdU/EdU labels CSV (dedupe, type-cast, sort).")
    p.add_argument("--barrage-dir", type=Path, default=Path("output/brdu_edu_barrage"))
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()

    labels_path = Path(args.barrage_dir).resolve() / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    with labels_path.with_suffix(labels_path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        original = labels_path.read_bytes()
        labels = pd.read_csv(labels_path)
        missing_labels = sorted({"cell", "brdu", "edu"} - set(labels.columns))
        if missing_labels:
            raise ValueError(f"labels.csv missing required columns: {missing_labels}")

        labels = labels[["cell", "brdu", "edu"]].copy()
        labels["cell"] = labels["cell"].astype(str)
        labels["brdu"] = labels["brdu"].astype(int)
        labels["edu"] = labels["edu"].astype(int)
        labels = labels.drop_duplicates(subset=["cell"], keep="last").reset_index(drop=True)

        print(f"labels: {len(labels)}")

        if not bool(args.apply):
            print("Dry run; pass --apply to modify labels.csv.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup = labels_path.with_name(f"{labels_path.name}.bak_{ts}")
        with backup.open("xb") as stream:
            stream.write(original)
        print(f"Backup written: {backup}")

        labels = labels.sort_values("cell", kind="stable").reset_index(drop=True)
        write_csv_atomic(labels, labels_path)
        print(f"Updated labels written: {labels_path} (rows={len(labels)})")


if __name__ == "__main__":
    main()
