from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from dino.artifacts import file_sha256, read_parquet_artifact
from dino.artifacts import row_feature_cache_key as _row_feature_cache_key


LABEL_COLUMNS = ("cell", "brdu", "edu")
SCORE_COLUMNS = {"brdu": "brdu_prob", "edu": "edu_prob"}
PANEL_COLORS = {"brdu": "#1f5a7a", "edu": "#b15928"}
ALL_PANEL_COLORS = {"brdu": "#70a4bf", "edu": "#d69a69"}
EXAMPLE_UM_PER_PIXEL = 0.216


@dataclass(frozen=True)
class RocConfig:
    labels_csv: Path
    dinoinfer_dir: Path
    bundle_path: Path | None
    split: str
    example_crop_size: int
    example_outline_width: float
    example_scale_bar_um: float
    out_png: Path
    out_csv: Path


def _parse_args() -> RocConfig:
    parser = argparse.ArgumentParser(
        description="Plot ROC curves for dinoinfer scores against manually labeled barrage cells."
    )
    parser.add_argument("--labels-csv", type=Path, default=Path("output/brdu_edu_barrage/labels.csv"))
    parser.add_argument("--dinoinfer-dir", type=Path, default=Path("~/nvme/dinoinfer").expanduser())
    parser.add_argument(
        "--bundle-path",
        type=Path,
        default=Path("output/brdu_edu_barrage/brdu_edu_cell_dino_hpa_vitl16_logreg.joblib"),
        help="DINO bundle used to recover the cached labeled cohort and held-out split.",
    )
    parser.add_argument(
        "--split",
        choices=("all", "cache", "train", "test"),
        default="test",
        help="Which label subset to evaluate. Train/test require saved cell identities in the bundle.",
    )
    parser.add_argument("--example-crop-size", type=int, default=80)
    parser.add_argument("--example-outline-width", type=float, default=1.5)
    parser.add_argument("--example-scale-bar-um", type=float, default=5.0)
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("output/brdu_edu_barrage/dinoinfer_barrage_roc.png"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("output/brdu_edu_barrage/dinoinfer_barrage_auc.csv"),
    )
    args = parser.parse_args()
    return RocConfig(
        labels_csv=Path(args.labels_csv).expanduser(),
        dinoinfer_dir=Path(args.dinoinfer_dir).expanduser(),
        bundle_path=Path(args.bundle_path).expanduser() if args.bundle_path is not None else None,
        split=str(args.split),
        example_crop_size=int(args.example_crop_size),
        example_outline_width=float(args.example_outline_width),
        example_scale_bar_um=float(args.example_scale_bar_um),
        out_png=Path(args.out_png).expanduser(),
        out_csv=Path(args.out_csv).expanduser(),
    )


def _load_labels(labels_csv: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    missing = sorted(set(LABEL_COLUMNS) - set(labels.columns))
    if missing:
        raise ValueError(f"labels.csv missing required columns: {missing}")

    labels = labels.loc[:, LABEL_COLUMNS].copy()
    labels["cell"] = labels["cell"].astype(str)
    if labels.duplicated("cell").any():
        duplicate_count = int(labels.duplicated("cell").sum())
        logger.warning(
            f"labels.csv contains {duplicate_count} duplicate cells; keeping the last label for each cell."
        )
        labels = labels.drop_duplicates("cell", keep="last")

    for target in ("brdu", "edu"):
        labels[target] = labels[target].astype(int)

    return labels


def _choose_stratify_labels(frame: pd.DataFrame) -> np.ndarray | None:
    candidates = [
        frame["brdu"].astype(str) + "_" + frame["edu"].astype(str),
        frame["brdu"].astype(str),
        frame["edu"].astype(str),
    ]
    for candidate in candidates:
        counts = candidate.value_counts()
        if not counts.empty and int(counts.min()) >= 2:
            return candidate.to_numpy()
    return None


def _load_feature_cache_keys(feature_cache_path: Path) -> set[str]:
    bundle = joblib.load(feature_cache_path)
    if not isinstance(bundle, dict) or "entries" not in bundle or not isinstance(bundle["entries"], dict):
        raise ValueError(f"Unexpected feature cache bundle structure at {feature_cache_path}")
    return set(str(k) for k in bundle["entries"])


def _load_bundle(bundle_path: Path) -> dict[str, object]:
    bundle = joblib.load(bundle_path)
    if not isinstance(bundle, dict):
        raise TypeError(f"Unexpected bundle type at {bundle_path}: {type(bundle)}")
    return bundle


def _resolve_feature_cache_path(bundle: dict[str, object], bundle_path: Path) -> Path:
    feature_cache = bundle.get("feature_cache")
    if not isinstance(feature_cache, str):
        raise ValueError(f"Bundle at {bundle_path} is missing string feature_cache metadata.")
    return Path(feature_cache).expanduser()


def _recover_cached_labeled_rows(
    *, labels: pd.DataFrame, labels_csv: Path, feature_cache_path: Path
) -> pd.DataFrame:
    meta_cols = ["cell", "dataset", "roi", "x", "y", "z"]
    index = pd.read_csv(labels_csv.with_name("index.csv"), usecols=meta_cols)
    joined = labels.merge(index, on="cell", how="inner", validate="one_to_one")
    cache_keys = _load_feature_cache_keys(feature_cache_path)
    mask = joined.apply(_row_feature_cache_key, axis=1).isin(cache_keys)
    recovered = joined.loc[mask].reset_index(drop=True)
    if recovered.empty:
        raise ValueError(f"No labeled rows from {labels_csv} matched feature cache {feature_cache_path}")
    return recovered


def _split_cached_rows(
    cached_rows: pd.DataFrame,
    *,
    seed: int,
    test_fraction: float,
) -> dict[str, pd.DataFrame]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be between 0 and 1, got {test_fraction}")

    train_idx, test_idx = train_test_split(
        np.arange(len(cached_rows)),
        test_size=test_fraction,
        random_state=seed,
        stratify=_choose_stratify_labels(cached_rows),
    )
    train_rows = cached_rows.iloc[np.sort(train_idx)].reset_index(drop=True)
    test_rows = cached_rows.iloc[np.sort(test_idx)].reset_index(drop=True)

    return {
        "cache": cached_rows.reset_index(drop=True),
        "train": train_rows,
        "test": test_rows,
    }


def _load_dinoinfer_scores(dinoinfer_dir: Path, *, expected_model_sha256: str | None = None) -> pd.DataFrame:
    parquet_paths = sorted(dinoinfer_dir.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No dinoinfer parquet files found in {dinoinfer_dir}")

    parts: list[pd.DataFrame] = []
    for path in parquet_paths:
        if expected_model_sha256 is not None:
            provenance, _ = read_parquet_artifact(path)
            if provenance.get("model_sha256") != expected_model_sha256:
                raise ValueError(f"Prediction model_sha256 does not match the supplied bundle: {path}")
        frame = pd.read_parquet(path)
        if "cell" in frame.columns:
            frame = frame.set_index("cell", drop=True)
        frame.index = frame.index.astype(str)

        missing = sorted(set(SCORE_COLUMNS.values()) - set(frame.columns))
        if missing:
            raise ValueError(f"{path} missing required score columns: {missing}")

        keep_cols = [col for col in ("dataset", "roi", *SCORE_COLUMNS.values()) if col in frame.columns]
        part = frame.loc[:, keep_cols].copy()
        part.index.name = "cell"
        parts.append(part)

    scores = pd.concat(parts, axis=0, copy=False)
    if scores.index.duplicated().any():
        dupes = scores.index[scores.index.duplicated()].unique().tolist()[:10]
        raise ValueError(f"Duplicate cell ids across dinoinfer parquets (showing up to 10): {dupes}")

    return scores


def _select_labels_for_split(
    *,
    labels_csv: Path,
    bundle_path: Path | None,
    split: str,
) -> tuple[pd.DataFrame, str]:
    labels = _load_labels(labels_csv)
    if split == "all":
        return labels, f"All labeled barrage cells (n={len(labels)})"

    if bundle_path is None:
        raise ValueError(f"split={split!r} requires --bundle-path")

    bundle = _load_bundle(bundle_path)
    if split == "cache":
        cached_rows = _recover_cached_labeled_rows(
            labels=labels,
            labels_csv=labels_csv,
            feature_cache_path=_resolve_feature_cache_path(bundle, bundle_path),
        )
        return cached_rows.loc[:, LABEL_COLUMNS].copy(), f"Cached labeled cohort (n={len(cached_rows)})"
    if split not in ("train", "test"):
        raise ValueError(f"Unsupported split {split!r}")

    saved_train_cells = bundle.get("train_cells")
    saved_test_cells = bundle.get("test_cells")
    if not isinstance(saved_train_cells, list) or not isinstance(saved_test_cells, list):
        raise ValueError(
            "Train/test evaluation requires saved train_cells and test_cells; retrain the bundle."
        )
    train_cells = [str(cell) for cell in saved_train_cells]
    test_cells = [str(cell) for cell in saved_test_cells]
    if len(set(train_cells)) != len(train_cells) or len(set(test_cells)) != len(test_cells):
        raise ValueError("Bundle saved split cell identities contain duplicates.")
    if set(train_cells) & set(test_cells):
        raise ValueError("Bundle saved train/test cell identities overlap.")
    selected_cells = train_cells if split == "train" else test_cells
    missing = set(selected_cells) - set(labels["cell"])
    if missing:
        raise ValueError(f"Labels are missing {len(missing)} saved {split} cells.")
    return (
        labels.set_index("cell").loc[selected_cells].reset_index(),
        f"Saved DINO {split} split (train={len(train_cells)}, test={len(test_cells)})",
    )


def _load_matched_labels_and_scores(
    *,
    labels_csv: Path,
    dinoinfer_dir: Path,
    bundle_path: Path | None,
    split: str,
) -> tuple[pd.DataFrame, int, str]:
    labels, split_description = _select_labels_for_split(
        labels_csv=labels_csv,
        bundle_path=bundle_path,
        split=split,
    )
    scores = _load_dinoinfer_scores(
        dinoinfer_dir,
        expected_model_sha256=file_sha256(bundle_path) if bundle_path is not None else None,
    )

    joined = labels.join(scores, on="cell", how="left")
    matched = joined.dropna(subset=list(SCORE_COLUMNS.values())).copy()
    if matched.empty:
        raise ValueError("No labeled barrage cells matched dinoinfer scores.")

    missing_count = int(len(joined) - len(matched))
    if missing_count:
        logger.warning(f"Skipping {missing_count} labeled cells without matching dinoinfer scores.")

    return matched, int(len(labels)), split_description


def _evaluate_targets(
    matched: pd.DataFrame, *, total_labels: int
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray | float]]]:
    rows: list[dict[str, int | float | str]] = []
    curves: dict[str, dict[str, np.ndarray | float]] = {}
    matched_count = int(len(matched))

    for target, score_col in SCORE_COLUMNS.items():
        y_true = matched[target].to_numpy(dtype=np.int64, copy=False)
        score = matched[score_col].to_numpy(dtype=np.float64, copy=False)
        class_counts = dict(zip(*np.unique(y_true, return_counts=True), strict=True))
        if len(class_counts) != 2:
            raise ValueError(f"Target {target!r} needs both classes present, got {class_counts}")

        fpr, tpr, _ = roc_curve(y_true, score)
        auc = float(roc_auc_score(y_true, score))
        positives = int(class_counts.get(1, 0))
        negatives = int(class_counts.get(0, 0))

        rows.append(
            {
                "target": target,
                "auc": auc,
                "matched_labels": matched_count,
                "total_labels": total_labels,
                "positives": positives,
                "negatives": negatives,
            }
        )
        curves[target] = {"fpr": fpr, "tpr": tpr, "auc": auc}

    summary = pd.DataFrame(rows).sort_values("target").reset_index(drop=True)
    return summary, curves


def _select_channel_example_rows(labels_csv: Path, *, n_per_class: int = 16) -> pd.DataFrame:
    if n_per_class <= 0:
        raise ValueError(f"n_per_class must be positive, got {n_per_class}")

    index_columns = ["rank", "cell", "dataset", "roi", "x", "y", "z"]
    index = pd.read_csv(labels_csv.with_name("index.csv"), usecols=index_columns)
    labeled = _load_labels(labels_csv).merge(index, on="cell", how="inner", validate="one_to_one")

    groups: list[pd.DataFrame] = []
    for channel, channel_index in (("brdu", 0), ("edu", 1)):
        for label, status in ((1, "Positive"), (0, "Negative")):
            group = labeled.loc[labeled[channel] == label].sort_values("rank", kind="stable")
            if len(group) < n_per_class:
                raise ValueError(
                    f"Need {n_per_class} {status.lower()} {channel} examples, found {len(group)}"
                )
            selected = group.head(n_per_class).copy()
            selected["channel"] = channel
            selected["channel_index"] = channel_index
            selected["status"] = status
            groups.append(selected)
    return pd.concat(groups, ignore_index=True)


def _render_channel_examples(*, labels_csv: Path, crop_size: int, outline_linewidth: float) -> pd.DataFrame:
    import anndata as ad
    import matplotlib.pyplot as plt

    from dino import barrage

    if crop_size <= 0:
        raise ValueError(f"example crop size must be positive, got {crop_size}")
    if outline_linewidth <= 0:
        raise ValueError(f"example outline width must be positive, got {outline_linewidth}")

    plt.style.use(Path(__file__).resolve().parent / "styles" / "thesis.mplstyle")
    examples = _select_channel_example_rows(labels_csv)
    obs = examples.drop_duplicates("cell", keep="first").set_index("cell", drop=False)
    adata = ad.AnnData(X=np.empty((len(obs), 0), dtype=np.float32), obs=obs)

    workspace_cache: dict[str, object] = {}
    fused_cache: dict[tuple[str, str, str], object] = {}
    seg_cache: dict[tuple[str, str, str, str], object] = {}
    images: list[np.ndarray] = []
    for row in examples.itertuples(index=False):
        dataset = str(row.dataset)
        cell = str(row.cell)
        ws = barrage._workspace_for_dataset(dataset=dataset, cache=workspace_cache)
        fig, _, _ = barrage.plot_cell_thumbnail_with_outline(
            adata=adata,
            dataset=dataset,
            ws=ws,
            cell=cell,
            codebook="edu",
            seg_codebook="pi",
            segmentation_name=barrage.DEFAULT_SEGMENTATION_NAME,
            fused_cache=fused_cache,
            seg_cache=seg_cache,
            size=crop_size,
            n_channels=1,
            channels=[int(row.channel_index)],
            cmap="magma",
            outline_color="cyan",
            outline_linewidth=outline_linewidth,
            render_all_outlines=True,
            detailed_titles=False,
        )
        ax = fig.axes[0]
        ax.set_title("")
        ax.set_position((0.0, 0.0, 1.0, 1.0))
        fig.set_size_inches(1.0, 1.0)
        fig.set_dpi(300)
        fig.canvas.draw()
        images.append(np.asarray(fig.canvas.buffer_rgba()).copy())
        plt.close(fig)
    rendered = examples.copy()
    rendered["image"] = images
    return rendered


def _make_channel_montage(
    examples: pd.DataFrame, *, channel: str, gap: int = 2, group_gap: int = 12
) -> np.ndarray:
    if gap < 0 or group_gap < 0:
        raise ValueError(f"gaps must be nonnegative, got gap={gap}, group_gap={group_gap}")

    groups = []
    for status in ("Negative", "Positive"):
        group = examples.loc[(examples["channel"] == channel) & (examples["status"] == status)]
        if len(group) != 16:
            raise ValueError(f"Expected 16 {status.lower()} {channel} examples, got {len(group)}")
        groups.append(group)

    first = np.asarray(groups[0].iloc[0]["image"])
    if first.ndim != 3:
        raise ValueError(f"Expected HWC example image, got shape={first.shape}")
    height, width, channels = first.shape
    block_width = 4 * width + 3 * gap
    montage = np.full(
        (4 * height + 3 * gap, 2 * block_width + group_gap, channels),
        255,
        dtype=first.dtype,
    )
    for column_group, group in enumerate(groups):
        for index, image in enumerate(group["image"]):
            tile = np.asarray(image)
            if tile.shape != first.shape:
                raise ValueError(f"Example image shape {tile.shape} does not match {first.shape}")
            row = index // 4
            y0 = row * (height + gap)
            x0 = column_group * (block_width + group_gap) + (index % 4) * (width + gap)
            montage[y0 : y0 + height, x0 : x0 + width] = tile
    return montage


def _plot_roc_curves(
    *,
    selected_split: str,
    selected_summary: pd.DataFrame,
    selected_curves: dict[str, dict[str, np.ndarray | float]],
    all_summary: pd.DataFrame,
    all_curves: dict[str, dict[str, np.ndarray | float]],
    examples: pd.DataFrame,
    example_crop_size: int,
    example_scale_bar_um: float,
    out_png: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use(Path(__file__).resolve().parent / "styles" / "thesis.mplstyle")
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11})

    expected_example_rows = 64
    if len(examples) != expected_example_rows:
        raise ValueError(f"Expected {expected_example_rows} channel examples, got {len(examples)}")
    if example_crop_size <= 0 or example_scale_bar_um <= 0:
        raise ValueError("Example crop size and scale bar length must be positive")

    fig = plt.figure(figsize=(8.8, 13.2))
    grid = fig.add_gridspec(
        5,
        2,
        height_ratios=(1.0, 0.09, 1.08, 0.01, 1.08),
        hspace=0.06,
        wspace=0.2,
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    axes[1].sharex(axes[0])
    axes[1].sharey(axes[0])
    target_labels = {"brdu": "BrdU", "edu": "EdU"}
    selected_label = "Held-out test" if selected_split == "test" else selected_split.capitalize()
    panel_titles = []
    for ax, panel_label, target in zip(axes, ("A", "B"), ("brdu", "edu"), strict=True):
        ax.plot([0, 1], [0, 1], linestyle="--", color="#bdbdbd", linewidth=1.0, zorder=1)
        for cohort_label, summary, curves, color, linewidth, zorder in (
            (selected_label, selected_summary, selected_curves, PANEL_COLORS[target], 2.6, 3),
            ("All scored", all_summary, all_curves, ALL_PANEL_COLORS[target], 2.2, 2),
        ):
            row = summary.loc[summary["target"] == target].iloc[0]
            curve = curves[target]
            ax.plot(
                curve["fpr"],
                curve["tpr"],
                color=color,
                linewidth=linewidth,
                label=(f"{cohort_label}: AUC = {float(row['auc']):.3f} (n = {int(row['matched_labels']):,})"),
                zorder=zorder,
            )
        panel_titles.append(
            ax.text(
                -0.12,
                1.03,
                f"{panel_label}  {target_labels[target]}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("False positive rate")
        ax.legend(
            loc="lower right",
            frameon=False,
            fontsize=9.5,
            handlelength=1.6,
            labelspacing=0.3,
        )

    axes[0].set_ylabel("True positive rate")
    montage_axes = []
    for grid_cell, panel_label, channel, channel_label in (
        (grid[2, :], "C", "brdu", "BrdU"),
        (grid[4, :], "D", "edu", "EdU"),
    ):
        ax = fig.add_subplot(grid_cell)
        montage_axes.append(ax)
        montage = _make_channel_montage(examples, channel=channel, gap=2, group_gap=12)
        ax.imshow(montage, interpolation="nearest")
        ax.axis("off")
        tile_width = np.asarray(examples.iloc[0]["image"]).shape[1]
        bar_length = example_scale_bar_um / EXAMPLE_UM_PER_PIXEL * tile_width / example_crop_size
        bar_right = montage.shape[1] - 12
        bar_y = montage.shape[0] - 20
        ax.plot(
            [bar_right - bar_length, bar_right],
            [bar_y, bar_y],
            color="white",
            linewidth=2.2,
            solid_capstyle="butt",
            zorder=10,
        )
        ax.text(
            0.0,
            1.005,
            f"{panel_label}  {channel_label}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )
        ax.text(
            0.25,
            1.005,
            "Negative",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="normal",
        )
        ax.text(
            0.75,
            1.005,
            "Positive",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="normal",
        )

    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.03, top=0.97)
    target = montage_axes[0].get_position()
    target_center = target.x0 + target.width / 2
    for _ in range(3):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tight_boxes = [ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()) for ax in axes]
        tight_left = min(box.x0 for box in tight_boxes)
        tight_right = max(box.x1 for box in tight_boxes)
        tight_center = (tight_left + tight_right) / 2
        scale = min(1.0, target.width / (tight_right - tight_left))
        for ax in axes:
            position = ax.get_position()
            center_x = target_center + (position.x0 + position.width / 2 - tight_center) * scale
            width = position.width * scale
            height = position.height * scale
            ax.set_position(
                (
                    center_x - width / 2,
                    position.y0 + (position.height - height) / 2,
                    width,
                    height,
                )
            )
    first_position = axes[0].get_position()
    panel_titles[0].set_x((target.x0 - first_position.x0) / first_position.width)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = _parse_args()
    selected_matched, selected_total, _ = _load_matched_labels_and_scores(
        labels_csv=cfg.labels_csv,
        dinoinfer_dir=cfg.dinoinfer_dir,
        bundle_path=cfg.bundle_path,
        split=cfg.split,
    )
    selected_summary, selected_curves = _evaluate_targets(selected_matched, total_labels=selected_total)
    all_matched, all_total, _ = _load_matched_labels_and_scores(
        labels_csv=cfg.labels_csv,
        dinoinfer_dir=cfg.dinoinfer_dir,
        bundle_path=cfg.bundle_path,
        split="all",
    )
    all_summary, all_curves = _evaluate_targets(all_matched, total_labels=all_total)
    examples = _render_channel_examples(
        labels_csv=cfg.labels_csv,
        crop_size=cfg.example_crop_size,
        outline_linewidth=cfg.example_outline_width,
    )
    logger.info("ROC montage cells: " + ", ".join(examples["cell"].astype(str).tolist()))

    cfg.out_csv.parent.mkdir(parents=True, exist_ok=True)
    selected_summary.insert(1, "split", cfg.split)
    all_summary.insert(1, "split", "all")
    summary = pd.concat([selected_summary, all_summary], ignore_index=True)
    summary.to_csv(cfg.out_csv, index=False)
    logger.info(f"Wrote ROC summary: {cfg.out_csv}")

    _plot_roc_curves(
        selected_split=cfg.split,
        selected_summary=selected_summary,
        selected_curves=selected_curves,
        all_summary=all_summary,
        all_curves=all_curves,
        examples=examples,
        example_crop_size=cfg.example_crop_size,
        example_scale_bar_um=cfg.example_scale_bar_um,
        out_png=cfg.out_png,
    )
    logger.info(f"Wrote ROC figure: {cfg.out_png}")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
