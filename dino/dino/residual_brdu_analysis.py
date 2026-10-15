"""Summarize paired raw nuclear medians from the residual BrdU measurement."""

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


def summarize_field(group: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Keep measured ADU and subtract the matching called-negative reference."""
    dataset, roi, tile = group.iloc[0][["dataset", "roi", "tile"]]
    x = group.brdu_median_raw_adu
    y = group.final_median_raw_adu
    edu = group.edu_median_raw_adu
    brdu_negative = ~group.brdu_positive
    edu_negative = ~group.edu_positive
    if not brdu_negative.any() or not edu_negative.any() or not group.edu_positive.any():
        raise ValueError(f"Missing required call reference in {dataset}/{roi}/{tile}")
    cutoff = float(x.quantile(0.9))
    bright = x >= cutoff
    final_baseline = float(y[brdu_negative].median())
    edu_baseline = float(edu[edu_negative].median())
    excess = float(y[bright].median() - final_baseline)
    edu_q10 = float(edu[group.edu_positive].quantile(0.1) - edu_baseline)
    edu_median = float(edu[group.edu_positive].median() - edu_baseline)
    if edu_q10 <= 0 or edu_median <= 0:
        raise ValueError(f"EdU-positive reference does not exceed negative median in {dataset}/{roi}/{tile}")
    centered_x = x.to_numpy() - x.mean()
    centered_y = y.to_numpy() - y.mean()
    slope = float(centered_x @ centered_y / (centered_x @ centered_x))
    row = {
        "dataset": dataset, "roi": str(roi), "tile": int(tile), "n": len(group),
        "n_bright": int(bright.sum()), "n_brdu_negative": int(brdu_negative.sum()),
        "n_edu_positive": int(group.edu_positive.sum()), "n_edu_negative": int(edu_negative.sum()),
        "brdu_bright_cutoff_adu": cutoff,
        "brdu_bright_median_adu": float(x[bright].median()),
        "brdu_negative_median_adu": float(x[brdu_negative].median()),
        "final_bright_median_adu": float(y[bright].median()),
        "final_negative_median_adu": final_baseline,
        "residual_excess_adu": excess,
        "edu_negative_median_adu": edu_baseline,
        "edu_positive_excess_median_adu": edu_median,
        "edu_positive_excess_q10_adu": edu_q10,
        "ratio_median": excess / edu_median,
        "ratio_q10": excess / edu_q10,
        "slope_final_adu_per_brdu_adu": slope,
        "slope_intercept_adu": float(y.mean() - slope * x.mean()),
        "spearman_rho": float(spearmanr(x, y).statistic),
    }
    row["fitted_bright_minus_negative_adu"] = slope * (
        row["brdu_bright_median_adu"] - row["brdu_negative_median_adu"]
    )
    row["fitted_contrast_over_edu_median"] = row["fitted_bright_minus_negative_adu"] / edu_median
    bins = group.assign(bin=pd.qcut(x, 20, duplicates="drop")).groupby("bin", observed=True).agg(
        brdu_x=("brdu_median_raw_adu", "median"),
        final_y=("final_median_raw_adu", "median"),
        final_q25=("final_median_raw_adu", lambda values: values.quantile(0.25)),
        final_q75=("final_median_raw_adu", lambda values: values.quantile(0.75)),
        n=("cell", "size"),
    ).reset_index(drop=True)
    bins.insert(0, "tile", int(tile))
    bins.insert(0, "roi", str(roi))
    bins.insert(0, "dataset", dataset)
    return row, bins


def run(measurements: Path, calls_path: Path, output: Path) -> None:
    measured = pd.read_parquet(measurements)
    calls = pd.read_parquet(calls_path)
    measured["roi"] = measured.roi.astype(str)
    calls["roi"] = calls.roi.astype(str)
    cells = measured.loc[measured.valid].merge(
        calls[["cell", "dataset", "roi", "brdu_positive", "edu_positive"]],
        on=["cell", "dataset", "roi"], how="left", validate="many_to_one", indicator=True,
    )
    if not cells._merge.eq("both").all():
        raise ValueError("Valid measured nuclei are missing current pulse calls")
    cells = cells.drop(columns="_merge")
    if cells.cell.duplicated().any():
        raise ValueError("Overlapping fields contain duplicate nuclei; assign one field before summarizing")
    rows, bins = [], []
    for _, group in cells.groupby(["dataset", "roi", "tile"], sort=True):
        row, field_bins = summarize_field(group)
        rows.append(row)
        bins.append(field_bins)
    references = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(output / "cells.parquet", index=False)
    pd.concat(bins, ignore_index=True).to_parquet(output / "bins.parquet", index=False)
    references.to_parquet(output / "references.parquet", index=False)
    references.to_csv(output / "field_summary.tsv", sep="\t", index=False)
    summary = {
        "cells": len(cells), "fields": len(references), "slides": cells.dataset.nunique(),
        "measurements": str(measurements), "calls": str(calls_path),
        "bright_definition": "At or above field-specific 90th percentile of initial BrdU nuclear median",
        "residual_definition": "Median final-readout nuclear median in bright nuclei minus median in current BrdU-negative nuclei",
        "edu_references": "Current EdU-positive median and 10th percentile, each minus current EdU-negative nuclear median in the same field",
        "units": "Acquired raw ADU; no illumination power correction",
        "slope": "Descriptive within-field least squares of final nuclear median on initial BrdU nuclear median",
        "bins": "Twenty quantile bins per field; observed medians and interquartile ranges",
        "calling_reference": "Measured positive intensity references; current classifier has no scalar ADU threshold",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(references.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--calls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.measurements, args.calls, args.output)


if __name__ == "__main__":
    main()
