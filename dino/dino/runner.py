from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from fishtools.brdu.barrage import load_obs_from_h5ad

DEFAULT_H5AD = Path("~/nvme/vzsvz.h5ad").expanduser()
DEFAULT_MODEL = Path("output/brdu_edu_barrage/brdu_edu_cell_dino_hpa_vitl16_logreg.joblib")
DEFAULT_OUTPUT_DIR = Path("~/nvme/dinoinfer").expanduser()


@dataclass(frozen=True)
class RunnerConfig:
    h5ad_path: Path
    model_path: Path
    datasets: tuple[str, ...]
    batch_size: int
    num_workers: int
    skip_existing: bool


def _parse_args() -> RunnerConfig:
    parser = argparse.ArgumentParser(
        description="Run Cell-DINO inference over datasets in ~/nvme/vzsvz.h5ad."
    )
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        default=[],
        help="Optional dataset filter. Repeat to run multiple datasets.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    return RunnerConfig(
        h5ad_path=Path(args.h5ad).expanduser(),
        model_path=Path(args.model_path).expanduser(),
        datasets=tuple(str(x) for x in args.datasets),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        skip_existing=bool(args.skip_existing),
    )


def _datasets_to_run(cfg: RunnerConfig) -> list[str]:
    obs = load_obs_from_h5ad(cfg.h5ad_path)
    if "dataset" not in obs.columns:
        raise ValueError(f"Input h5ad is missing 'dataset': {cfg.h5ad_path}")

    datasets = obs["dataset"].astype(str).drop_duplicates().tolist()
    if not cfg.datasets:
        return datasets

    available = set(datasets)
    missing = [dataset for dataset in cfg.datasets if dataset not in available]
    if missing:
        raise ValueError(f"Requested dataset(s) not found in {cfg.h5ad_path}: {missing}")
    requested = set(cfg.datasets)
    return [dataset for dataset in datasets if dataset in requested]


def _output_path_for_dataset(dataset: str) -> Path:
    return DEFAULT_OUTPUT_DIR / f"{dataset}-dino-vit16.parquet"


def main() -> None:
    cfg = _parse_args()
    datasets = _datasets_to_run(cfg)

    for dataset in datasets:
        output_path = _output_path_for_dataset(dataset)
        cmd = [
            sys.executable,
            "-m",
            "dino.infer",
            "--h5ad",
            str(cfg.h5ad_path),
            "--model-path",
            str(cfg.model_path),
            "--dataset",
            dataset,
            "--output-path",
            str(output_path),
            "--batch-size",
            str(cfg.batch_size),
            "--num-workers",
            str(cfg.num_workers),
        ]
        if cfg.skip_existing:
            cmd.append("--skip-existing")
        print(f"[run] dataset={dataset} output={output_path}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
