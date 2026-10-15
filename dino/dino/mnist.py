from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import v2

from dinov3.data.transforms import make_classification_eval_transform
from dinov3.hub.backbones import dinov3_vits16plus


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

C_POWER_RANGE = np.linspace(-6.0, 5.0, 45)


class MnistOddEvenDataset(Dataset[tuple[torch.Tensor, int]]):
    """Resize MNIST digits for DINOv3 and map labels to odd/even."""

    def __init__(self, root: Path, *, train: bool, resize_size: int, crop_size: int) -> None:
        base_transform = make_classification_eval_transform(
            resize_size=resize_size,
            crop_size=crop_size,
            interpolation=v2.InterpolationMode.BICUBIC,
        )
        self.transform = v2.Compose(
            [
                v2.Lambda(lambda image: image.convert("RGB")),
                base_transform,
            ]
        )
        self.dataset = MNIST(root=root, train=train, download=True, transform=self.transform)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, digit = self.dataset[index]
        target = int(digit % 2 == 1)
        return image, target


@dataclass(frozen=True)
class SplitData:
    train: Dataset[tuple[torch.Tensor, int]]
    val: Dataset[tuple[torch.Tensor, int]]
    test: Dataset[tuple[torch.Tensor, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DINOv3 ViT-S+ frozen-feature logreg on MNIST odd/even.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("~/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth").expanduser(),
        help="Local DINOv3 ViT-S+ checkpoint path.",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/mnist"), help="MNIST download/cache directory."
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Feature extraction batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument("--resize-size", type=int, default=256, help="DINOv3 eval resize size.")
    parser.add_argument("--crop-size", type=int, default=224, help="DINOv3 eval crop size.")
    parser.add_argument(
        "--val-fraction", type=float, default=0.1, help="Fraction of train split used for C sweep."
    )
    parser.add_argument(
        "--train-limit", type=int, default=0, help="Optional cap on train examples before split."
    )
    parser.add_argument("--test-limit", type=int, default=0, help="Optional cap on test examples.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def maybe_limit_dataset(
    dataset: Dataset[tuple[torch.Tensor, int]], limit: int
) -> Dataset[tuple[torch.Tensor, int]]:
    if limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def make_splits(args: argparse.Namespace) -> SplitData:
    train_full = maybe_limit_dataset(
        MnistOddEvenDataset(
            args.data_root,
            train=True,
            resize_size=args.resize_size,
            crop_size=args.crop_size,
        ),
        args.train_limit,
    )
    test = maybe_limit_dataset(
        MnistOddEvenDataset(
            args.data_root,
            train=False,
            resize_size=args.resize_size,
            crop_size=args.crop_size,
        ),
        args.test_limit,
    )

    labels = np.fromiter((train_full[index][1] for index in range(len(train_full))), dtype=np.int64)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=args.seed)
    train_indices, val_indices = next(splitter.split(np.zeros(len(labels)), labels))

    return SplitData(
        train=Subset(train_full, train_indices.tolist()),
        val=Subset(train_full, val_indices.tolist()),
        test=test,
    )


def extract_features(
    *,
    model: torch.nn.Module,
    dataset: Dataset[tuple[torch.Tensor, int]],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for images, target in loader:
        images = images.to(device, non_blocking=True)
        with torch.inference_mode():
            batch_features = F.normalize(model(images), dim=1, p=2)
        features.append(batch_features.cpu().numpy())
        labels.append(target.numpy())

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def fit_and_score(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
) -> tuple[float, float]:
    best_c = 1.0
    best_accuracy = -1.0

    for exponent in C_POWER_RANGE:
        c_value = float(10.0**exponent)
        clf = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            C=c_value,
            max_iter=1_000,
            tol=1e-12,
        )
        clf.fit(train_features, train_labels)
        val_accuracy = accuracy_score(val_labels, clf.predict(val_features))
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_c = c_value

    return best_c, best_accuracy


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device={device} with weights={args.weights}")

    splits = make_splits(args)
    logger.info(
        f"Prepared MNIST odd/even splits: train={len(splits.train)}, val={len(splits.val)}, test={len(splits.test)}"
    )

    model = dinov3_vits16plus(weights=str(args.weights)).eval().to(device)

    train_features, train_labels = extract_features(
        model=model,
        dataset=splits.train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    val_features, val_labels = extract_features(
        model=model,
        dataset=splits.val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    test_features, test_labels = extract_features(
        model=model,
        dataset=splits.test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    logger.info(
        f"Extracted features: train={train_features.shape}, val={val_features.shape}, test={test_features.shape}"
    )

    best_c, best_val_accuracy = fit_and_score(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
    )
    logger.info(f"Best validation C={best_c:.4g}, val_accuracy={best_val_accuracy:.4f}")

    final_train_features = np.concatenate([train_features, val_features], axis=0)
    final_train_labels = np.concatenate([train_labels, val_labels], axis=0)
    final_model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        C=best_c,
        max_iter=1_000,
        tol=1e-12,
    )
    final_model.fit(final_train_features, final_train_labels)
    test_predictions = final_model.predict(test_features)
    test_accuracy = accuracy_score(test_labels, test_predictions)

    print(f"device={device}")
    print(f"weights={args.weights}")
    print(f"train_size={len(splits.train)}")
    print(f"val_size={len(splits.val)}")
    print(f"test_size={len(splits.test)}")
    print(f"feature_dim={train_features.shape[1]}")
    print(f"best_c={best_c:.10g}")
    print(f"val_accuracy={best_val_accuracy:.6f}")
    print(f"test_accuracy={test_accuracy:.6f}")


if __name__ == "__main__":
    main()
