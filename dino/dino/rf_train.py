from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from fishtools.brdu.model import fit_models_from_barrage_dir, save_model_bundle


@dataclass(frozen=True)
class TrainConfig:
    barrage_dir: Path
    model_out: Path
    n_estimators: int
    seed: int


def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(
        description="Train BrdU/EdU RF model from a barrage dir and save a model artifact."
    )
    p.add_argument("--barrage-dir", type=Path, default=Path("output/brdu_edu_barrage"))
    p.add_argument(
        "--model-out",
        type=Path,
        default=None,
        help="Output model path (.joblib). Default: <barrage-dir>/brdu_edu_model.joblib",
    )
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    barrage_dir = Path(args.barrage_dir)
    model_out = (
        Path(args.model_out) if args.model_out is not None else (barrage_dir / "brdu_edu_model.joblib")
    )
    return TrainConfig(
        barrage_dir=barrage_dir,
        model_out=model_out,
        n_estimators=int(args.n_estimators),
        seed=int(args.seed),
    )


def main() -> None:
    cfg = _parse_args()
    bundle = fit_models_from_barrage_dir(
        barrage_dir=cfg.barrage_dir,
        n_estimators=cfg.n_estimators,
        seed=cfg.seed,
    )
    save_model_bundle(bundle, path=cfg.model_out)


if __name__ == "__main__":
    main()
