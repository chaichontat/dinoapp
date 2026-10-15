from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Sequence

import numpy as np
import zarr
from scipy.ndimage import binary_erosion, gaussian_filter
from skimage.morphology import disk
from squisher.jpegxr_zarr import register_jpegxr_codec

from dino.artifacts import file_sha256


SOURCE_SCHEMA = "dino-zarr-source-v1"


@dataclass(frozen=True)
class CellCrop:
    labels: np.ndarray
    mask: np.ndarray
    neighbor_mask: np.ndarray
    channels: dict[str, np.ndarray]


ChunkKey = tuple[int, int, int]
ChunkRequest = tuple[str, zarr.Array, ChunkKey]


@dataclass(frozen=True)
class CellReadPlan:
    cells: tuple[tuple[int, int, int], ...]
    requests: tuple[ChunkRequest, ...]
    decoded_bytes: int


@dataclass(frozen=True)
class OpenSource:
    manifest: dict[str, Any]
    labels: zarr.Array
    channels: dict[str, zarr.Array]

    @property
    def offset_zyx(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.manifest["offset_zyx"])


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "source_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(value: Mapping[str, Any], path: Path, *, overwrite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=".dino-", suffix=".tmp", delete=False
    ) as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _multiscales(group: zarr.Group) -> Any:
    multiscales = group.attrs.get("multiscales")
    if multiscales is None:
        ome = group.attrs.get("ome")
        multiscales = ome.get("multiscales") if isinstance(ome, dict) else None
    return multiscales


def _array_path_from_group(group: zarr.Group) -> str:
    multiscales = _multiscales(group)
    if isinstance(multiscales, list) and multiscales:
        datasets = multiscales[0].get("datasets")
        if isinstance(datasets, list) and datasets and isinstance(datasets[0].get("path"), str):
            return str(datasets[0]["path"])
    if "0" in group:
        return "0"
    arrays = list(group.array_keys())
    if len(arrays) == 1:
        return arrays[0]
    raise ValueError(f"Unable to resolve one level-0 array from Zarr group with arrays={arrays}")


def _ome_level_metadata(root: Path, array_path: str) -> dict[str, Any] | None:
    node = zarr.open(root, mode="r")
    if isinstance(node, zarr.Array):
        return None
    multiscales = _multiscales(node)
    if not isinstance(multiscales, list) or not multiscales:
        return None
    multiscale = multiscales[0]
    datasets = multiscale.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError(f"OME-Zarr multiscales datasets are invalid: {root}")
    dataset = next((item for item in datasets if item.get("path") == array_path), None)
    if dataset is None:
        raise ValueError(f"OME-Zarr metadata does not describe array {array_path!r}: {root}")
    transformations = dataset.get("coordinateTransformations", dataset.get("coordinate_transformations"))
    return {
        "axes": multiscale.get("axes"),
        "coordinate_transformations": transformations,
    }


def resolve_array(root: Path, *, array_path: str | None = None) -> tuple[zarr.Array, str]:
    register_jpegxr_codec()
    if not root.exists():
        raise FileNotFoundError(f"Zarr source does not exist: {root}")
    node = zarr.open(root, mode="r")
    if isinstance(node, zarr.Array):
        if array_path not in (None, ""):
            raise ValueError(f"Array path {array_path!r} supplied for root Zarr array: {root}")
        return node, ""
    resolved = _array_path_from_group(node) if array_path is None else array_path
    child = node[resolved]
    if not isinstance(child, zarr.Array):
        raise TypeError(f"Resolved Zarr node is not an array: root={root} array={resolved!r}")
    return child, resolved


def metadata_digest(root: Path, array_path: str) -> str:
    candidates = [root / "zarr.json", root / ".zarray", root / ".zattrs", root / ".zgroup"]
    if array_path:
        array_root = root / array_path
        candidates.extend([array_root / "zarr.json", array_root / ".zarray", array_root / ".zattrs"])
    metadata_paths = [candidate for candidate in candidates if candidate.is_file()]
    if not metadata_paths:
        raise ValueError(f"No Zarr metadata files found under {root}")
    digest = hashlib.sha256()
    for candidate in metadata_paths:
        relative = candidate.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(candidate.read_bytes())
    return digest.hexdigest()


def _array_content_digest(root: Path, array_path: str) -> str:
    """Fingerprint stored level-0 bytes without decoding chunks or loading a volume."""
    array_root = root / array_path
    digest = hashlib.sha256()

    def visit(directory: Path) -> None:
        for candidate in sorted(directory.iterdir()):
            if candidate.is_dir():
                visit(candidate)
            else:
                relative = candidate.relative_to(array_root).as_posix().encode("utf-8")
                digest.update(len(relative).to_bytes(8, "big"))
                digest.update(relative)
                digest.update(bytes.fromhex(file_sha256(candidate)))

    visit(array_root)
    return digest.hexdigest()


def build_source_manifest(
    *,
    labels_path: Path,
    channel_paths: Mapping[str, Path],
    offset_zyx: tuple[int, int, int],
    required_channels: Sequence[str],
    dataset: str | None = None,
    roi: str | None = None,
) -> dict[str, Any]:
    if (dataset is None) != (roi is None):
        raise ValueError("Explicit source identity requires both dataset and roi")
    if dataset is not None and any(
        not isinstance(value, str) or not value.strip() or "|" in value
        for value in (dataset, roi)
    ):
        raise ValueError("Source identity must contain nonempty names without the cell delimiter")
    labels, labels_array_path = resolve_array(labels_path)
    missing_channels = sorted(set(required_channels) - set(channel_paths))
    if missing_channels:
        raise ValueError(f"Missing required channel sources: {missing_channels}")
    channel_names = (
        *required_channels,
        *sorted(set(channel_paths) - set(required_channels)),
    )
    channels: dict[str, Any] = {}
    fused_shape: tuple[int, ...] | None = None
    fused_ome_metadata: dict[str, Any] | None = None
    for name in channel_names:
        root = channel_paths[name].resolve()
        array, array_path = resolve_array(root)
        if array.ndim != 3:
            raise ValueError(f"Expected {name} source with ZYX axes, got shape={array.shape}")
        if fused_shape is None:
            fused_shape = tuple(int(value) for value in array.shape)
        elif tuple(array.shape) != fused_shape:
            raise ValueError(f"Fused channel shape mismatch for {name}: {array.shape} != {fused_shape}")
        ome_metadata = _ome_level_metadata(root, array_path)
        if fused_ome_metadata is None:
            fused_ome_metadata = ome_metadata
        elif ome_metadata != fused_ome_metadata:
            raise ValueError(f"OME axes or level-0 transform mismatch for fused channel {name}")
        channels[name] = {
            "root": str(root),
            "array": array_path,
            "shape": list(array.shape),
            "chunks": list(array.chunks),
            "dtype": str(array.dtype),
            "metadata_sha256": metadata_digest(root, array_path),
            "content_sha256": _array_content_digest(root, array_path),
            "ome": ome_metadata,
        }

    if labels.ndim != 3:
        raise ValueError(f"Expected labels with ZYX axes, got shape={labels.shape}")
    if fused_shape is None:
        raise RuntimeError("Expected fused shape to be initialized")
    for axis, (offset, label_size, fused_size) in enumerate(
        zip(offset_zyx, labels.shape, fused_shape, strict=True)
    ):
        if offset < 0 or offset + label_size > fused_size:
            raise ValueError(
                f"Label ROI does not fit fused source on axis {axis}: "
                f"offset={offset} label={label_size} fused={fused_size}"
            )

    labels_root = labels_path.resolve()
    if dataset is None:
        roi = labels_root.parent.parent.name
        dataset = labels_root.parent.parent.parent.name
    manifest: dict[str, Any] = {
        "schema": SOURCE_SCHEMA,
        "dataset": dataset,
        "roi": roi,
        "offset_zyx": list(offset_zyx),
        "labels": {
            "root": str(labels_root),
            "array": labels_array_path,
            "shape": list(labels.shape),
            "chunks": list(labels.chunks),
            "dtype": str(labels.dtype),
            "metadata_sha256": metadata_digest(labels_root, labels_array_path),
            "content_sha256": _array_content_digest(labels_root, labels_array_path),
        },
        "channels": channels,
    }
    manifest["source_sha256"] = manifest_digest(manifest)
    return manifest


def load_source(source_dir: Path) -> OpenSource:
    path = source_dir / "source.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing direct-Zarr source manifest: {path}")
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema") != SOURCE_SCHEMA:
        raise ValueError(f"Unsupported direct-Zarr source manifest: {path}")
    expected = manifest.get("source_sha256")
    actual = manifest_digest(manifest)
    if expected != actual:
        raise ValueError(f"Source manifest digest mismatch: {path}")

    label_spec = manifest["labels"]
    labels, resolved = resolve_array(Path(label_spec["root"]), array_path=label_spec["array"] or None)
    _validate_array_spec(
        role="masked labels",
        array=labels,
        resolved=resolved,
        spec=label_spec,
        check_ome=False,
    )
    channels: dict[str, zarr.Array] = {}
    for name, spec in manifest["channels"].items():
        array, resolved = resolve_array(Path(spec["root"]), array_path=spec["array"] or None)
        _validate_array_spec(
            role=f"channel {name}",
            array=array,
            resolved=resolved,
            spec=spec,
            check_ome=True,
        )
        channels[name] = array
    return OpenSource(manifest=manifest, labels=labels, channels=channels)


def _validate_array_spec(
    *,
    role: str,
    array: zarr.Array,
    resolved: str,
    spec: Mapping[str, Any],
    check_ome: bool,
) -> None:
    root = Path(spec["root"])
    actual: dict[str, Any] = {
        "array": resolved,
        "shape": list(array.shape),
        "chunks": list(array.chunks),
        "dtype": str(array.dtype),
        "metadata_sha256": metadata_digest(root, resolved),
        "content_sha256": _array_content_digest(root, resolved),
    }
    if check_ome:
        actual["ome"] = _ome_level_metadata(root, resolved)
    mismatches = {
        key: {"expected": spec.get(key), "actual": value}
        for key, value in actual.items()
        if spec.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Direct-Zarr {role} no longer matches source.json: {mismatches}")


class _ChunkCache:
    def __init__(self, *, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("cache_bytes must be positive")
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self.values: OrderedDict[tuple[str, int, int, int], np.ndarray] = OrderedDict()
        self._executor: ThreadPoolExecutor | None = None
        self._workers: int | None = None
        self._lock = Lock()

    @staticmethod
    def _load(array: zarr.Array, key: ChunkKey) -> np.ndarray:
        chunk_z, chunk_y, chunk_x = key
        depth, height, width = (int(value) for value in array.chunks)
        z0, y0, x0 = chunk_z * depth, chunk_y * height, chunk_x * width
        return np.asarray(
            array[
                z0 : min(z0 + depth, array.shape[0]),
                y0 : min(y0 + height, array.shape[1]),
                x0 : min(x0 + width, array.shape[2]),
            ]
        )

    def _store(self, cache_key: tuple[str, int, int, int], block: np.ndarray) -> None:
        with self._lock:
            while self.values and self.current_bytes + block.nbytes > self.max_bytes:
                _, evicted = self.values.popitem(last=False)
                self.current_bytes -= evicted.nbytes
            if block.nbytes <= self.max_bytes:
                self.values[cache_key] = block
                self.current_bytes += block.nbytes

    @staticmethod
    def _decoded_bytes(array: zarr.Array, key: ChunkKey) -> int:
        starts = tuple(index * int(chunk) for index, chunk in zip(key, array.chunks, strict=True))
        shape = tuple(
            max(0, min(int(chunk), int(size) - start))
            for start, chunk, size in zip(starts, array.chunks, array.shape, strict=True)
        )
        return int(np.prod(shape, dtype=np.int64)) * int(np.dtype(array.dtype).itemsize)

    def _pool(self, workers: int) -> ThreadPoolExecutor:
        if self._executor is None:
            self._workers = workers
            self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dino-zarr")
        elif self._workers != workers:
            raise ValueError(
                f"A crop reader must use one prefetch worker count: {self._workers} != {workers}"
            )
        return self._executor

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
            self._workers = None

    def get(self, role: str, array: zarr.Array, key: tuple[int, int, int]) -> np.ndarray:
        cache_key = (role, *key)
        with self._lock:
            cached = self.values.pop(cache_key, None)
            if cached is not None:
                self.values[cache_key] = cached
                return cached
        block = self._load(array, key)
        self._store(cache_key, block)
        return block

    def prefetch(
        self,
        requests: tuple[ChunkRequest, ...] | list[ChunkRequest],
        *,
        workers: int,
    ) -> None:
        if workers <= 0:
            raise ValueError("prefetch workers must be positive")
        required: OrderedDict[tuple[str, int, int, int], tuple[zarr.Array, ChunkKey]] = OrderedDict()
        for role, array, key in requests:
            required.setdefault((role, *key), (array, key))
        required_bytes = sum(self._decoded_bytes(array, key) for array, key in required.values())
        if required_bytes > self.max_bytes:
            raise RuntimeError(
                f"The planned crop batch needs {required_bytes} decoded bytes but the chunk cache "
                f"allows {self.max_bytes}; increase --cache-gb or reduce --batch-size"
            )
        pending: list[tuple[tuple[str, int, int, int], tuple[zarr.Array, ChunkKey]]] = []
        for cache_key, request in required.items():
            cached = self.values.pop(cache_key, None)
            if cached is None:
                pending.append((cache_key, request))
            else:
                self.values[cache_key] = cached
        pending_bytes = sum(self._decoded_bytes(*request) for _, request in pending)
        required_keys = set(required)
        while self.values and self.current_bytes + pending_bytes > self.max_bytes:
            stale_key = next((key for key in self.values if key not in required_keys), None)
            if stale_key is None:
                break
            evicted = self.values.pop(stale_key)
            self.current_bytes -= evicted.nbytes

        executor = self._pool(workers)
        queued = iter(pending)
        active: dict[Future[np.ndarray], tuple[str, int, int, int]] = {}
        for _ in range(min(workers, len(pending))):
            cache_key, request = next(queued)
            active[executor.submit(self._load, *request)] = cache_key
        while active:
            completed, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in completed:
                cache_key = active.pop(future)
                self._store(cache_key, future.result())
                try:
                    next_cache_key, request = next(queued)
                except StopIteration:
                    continue
                active[executor.submit(self._load, *request)] = next_cache_key
        missing = [cache_key for cache_key in required if cache_key not in self.values]
        if missing:
            raise RuntimeError(
                "The planned crop batch exceeds the chunk cache; increase --cache-gb or reduce --batch-size"
            )


class CellCropReader:
    """Assemble centered 2D crops from chunk-aligned cached source reads."""

    def __init__(
        self,
        *,
        labels: zarr.Array,
        channels: Mapping[str, zarr.Array],
        offset_zyx: tuple[int, int, int],
        cache_bytes: int,
    ) -> None:
        self.labels = labels
        self.channels = dict(channels)
        self.offset_zyx = offset_zyx
        self.cache = _ChunkCache(max_bytes=cache_bytes)

    def close(self) -> None:
        self.cache.close()

    @staticmethod
    def _required_chunk_keys(
        array: zarr.Array,
        *,
        z_index: int,
        center_y: int,
        center_x: int,
        size: int,
    ) -> tuple[tuple[int, int, int], ...]:
        if z_index < 0 or z_index >= array.shape[0]:
            return ()
        half = size // 2
        y0, x0 = center_y - half, center_x - half
        y1, x1 = y0 + size, x0 + size
        src_y0, src_x0 = max(0, y0), max(0, x0)
        src_y1, src_x1 = min(array.shape[1], y1), min(array.shape[2], x1)
        if src_y0 >= src_y1 or src_x0 >= src_x1:
            return ()
        chunk_depth, chunk_height, chunk_width = (int(value) for value in array.chunks)
        chunk_z = z_index // chunk_depth
        return tuple(
            (chunk_z, chunk_y, chunk_x)
            for chunk_y in range(src_y0 // chunk_height, (src_y1 - 1) // chunk_height + 1)
            for chunk_x in range(src_x0 // chunk_width, (src_x1 - 1) // chunk_width + 1)
        )

    def plan_cells(
        self,
        cells: list[tuple[int, int, int]],
        *,
        size: int,
    ) -> CellReadPlan:
        """Resolve the exact multi-array chunk footprint before issuing any I/O."""
        requests: OrderedDict[tuple[str, int, int, int], ChunkRequest] = OrderedDict()
        offset_z, offset_y, offset_x = self.offset_zyx
        for plane_z, center_y, center_x in cells:
            for key in self._required_chunk_keys(
                self.labels,
                z_index=plane_z,
                center_y=center_y,
                center_x=center_x,
                size=size,
            ):
                requests.setdefault(("labels", *key), ("labels", self.labels, key))
            for name, array in self.channels.items():
                for key in self._required_chunk_keys(
                    array,
                    z_index=plane_z + offset_z,
                    center_y=center_y + offset_y,
                    center_x=center_x + offset_x,
                    size=size,
                ):
                    requests.setdefault((name, *key), (name, array, key))
        planned = tuple(requests.values())
        return CellReadPlan(
            cells=tuple(cells),
            requests=planned,
            decoded_bytes=sum(self.cache._decoded_bytes(array, key) for _, array, key in planned),
        )

    def prefetch_plan(self, plan: CellReadPlan, *, workers: int) -> None:
        self.cache.prefetch(plan.requests, workers=workers)

    def plan_fits(self, plan: CellReadPlan) -> bool:
        return plan.decoded_bytes <= self.cache.max_bytes

    def prefetch_cells(
        self,
        cells: list[tuple[int, int, int]],
        *,
        size: int,
        workers: int,
    ) -> None:
        self.prefetch_plan(self.plan_cells(cells, size=size), workers=workers)

    def _read_crop(
        self,
        *,
        role: str,
        array: zarr.Array,
        z_index: int,
        center_y: int,
        center_x: int,
        size: int,
    ) -> np.ndarray:
        if size <= 0:
            raise ValueError("Crop size must be positive")
        output = np.zeros((size, size), dtype=array.dtype)
        if z_index < 0 or z_index >= array.shape[0]:
            return output
        half = size // 2
        y0, x0 = center_y - half, center_x - half
        y1, x1 = y0 + size, x0 + size
        src_y0, src_x0 = max(0, y0), max(0, x0)
        src_y1, src_x1 = min(array.shape[1], y1), min(array.shape[2], x1)
        if src_y0 >= src_y1 or src_x0 >= src_x1:
            return output

        chunk_depth, chunk_height, chunk_width = (int(value) for value in array.chunks)
        chunk_z = z_index // chunk_depth
        for chunk_y in range(src_y0 // chunk_height, (src_y1 - 1) // chunk_height + 1):
            for chunk_x in range(src_x0 // chunk_width, (src_x1 - 1) // chunk_width + 1):
                block = self.cache.get(role, array, (chunk_z, chunk_y, chunk_x))
                block_z0 = chunk_z * chunk_depth
                block_y0 = chunk_y * chunk_height
                block_x0 = chunk_x * chunk_width
                copy_y0, copy_y1 = (
                    max(src_y0, block_y0),
                    min(src_y1, block_y0 + block.shape[1]),
                )
                copy_x0, copy_x1 = (
                    max(src_x0, block_x0),
                    min(src_x1, block_x0 + block.shape[2]),
                )
                output[copy_y0 - y0 : copy_y1 - y0, copy_x0 - x0 : copy_x1 - x0] = block[
                    z_index - block_z0,
                    copy_y0 - block_y0 : copy_y1 - block_y0,
                    copy_x0 - block_x0 : copy_x1 - block_x0,
                ]
        return output

    def read_cell(
        self,
        *,
        label: int,
        plane_z: int,
        center_y: int,
        center_x: int,
        size: int,
    ) -> CellCrop:
        labels = self._read_crop(
            role="labels",
            array=self.labels,
            z_index=plane_z,
            center_y=center_y,
            center_x=center_x,
            size=size,
        )
        mask = labels == label
        channel_crops: dict[str, np.ndarray] = {}
        offset_z, offset_y, offset_x = self.offset_zyx
        for name, array in self.channels.items():
            channel_crops[name] = self._read_crop(
                role=name,
                array=array,
                z_index=plane_z + offset_z,
                center_y=center_y + offset_y,
                center_x=center_x + offset_x,
                size=size,
            )
        return CellCrop(
            labels=labels,
            mask=mask,
            neighbor_mask=(labels != 0) & ~mask,
            channels=channel_crops,
        )


def cell_dino_images(
    crop: CellCrop,
    *,
    input_channels: int,
    channel_names_by_target: Mapping[str, tuple[str, ...]],
    highpass_sigma_px: float = 0.0,
    mask_erosion_px: int = 0,
) -> dict[str, np.ndarray]:
    """Render target inputs while preprocessing each unique source channel once."""
    if input_channels not in (4, 5):
        raise ValueError(f"Cell-DINO input width must be 4 or 5, got {input_channels}")
    if highpass_sigma_px < 0:
        raise ValueError("Cell-DINO high-pass sigma must be non-negative")
    for channel_names in channel_names_by_target.values():
        if len(channel_names) > input_channels:
            raise ValueError(f"Too many source channels for Cell-DINO input: {channel_names}")
        if len(set(channel_names)) != len(channel_names):
            raise ValueError(f"Cell-DINO source channels must be unique: {channel_names}")
    effective_mask = cell_dino_mask(crop, mask_erosion_px=mask_erosion_px)
    unique_channel_names = dict.fromkeys(
        name for channel_names in channel_names_by_target.values() for name in channel_names
    )
    scaled_channels: dict[str, np.ndarray] = {}
    for name in unique_channel_names:
        if name not in crop.channels:
            raise ValueError(f"Cell crop is missing requested channel: {name}")
        channel = crop.channels[name]
        if highpass_sigma_px > 0:
            signal = channel.astype(np.float32)
            signal -= gaussian_filter(signal, sigma=highpass_sigma_px, mode="reflect")
            np.maximum(signal, 0, out=signal)
            scaled = np.clip(signal / 256.0, 0, 255).astype(np.uint8)
        else:
            scaled = np.clip(channel / 256.0, 0, 255).astype(np.uint8)
        scaled_channels[name] = scaled
    images: dict[str, np.ndarray] = {}
    for target, channel_names in channel_names_by_target.items():
        image = np.zeros((input_channels, *crop.mask.shape), dtype=np.uint8)
        for channel_index, name in enumerate(channel_names):
            np.copyto(image[channel_index], scaled_channels[name], where=effective_mask)
        images[target] = image
    return images


def cell_dino_mask(crop: CellCrop, *, mask_erosion_px: int) -> np.ndarray:
    if mask_erosion_px < 0:
        raise ValueError("Cell-DINO mask erosion must be non-negative")
    if mask_erosion_px == 0:
        return crop.mask
    return binary_erosion(crop.mask, structure=disk(mask_erosion_px), border_value=0)


def neighbor_cell_intensity(
    crop: CellCrop,
    *,
    channel_name: str,
    max_neighbors: int,
) -> tuple[float, int]:
    """Return the median mean intensity of the nearest visible neighboring cells."""
    if max_neighbors <= 0:
        raise ValueError("max_neighbors must be positive")
    if channel_name not in crop.channels:
        raise ValueError(f"Cell crop is missing requested channel: {channel_name}")
    if not crop.mask.any():
        raise ValueError("Target cell mask is empty")

    target_center = np.argwhere(crop.mask).mean(axis=0)
    channel = crop.channels[channel_name]
    neighbors: list[tuple[float, float]] = []
    for label in np.unique(crop.labels[crop.neighbor_mask]):
        neighbor_mask = crop.labels == label
        center = np.argwhere(neighbor_mask).mean(axis=0)
        distance_squared = float(np.sum((center - target_center) ** 2))
        neighbors.append((distance_squared, float(np.mean(channel[neighbor_mask]))))

    if not neighbors:
        return float("nan"), 0
    neighbors.sort(key=lambda item: item[0])
    selected = neighbors[:max_neighbors]
    return float(np.median([intensity for _, intensity in selected])), len(selected)


def open_review_source(source_dir: Path) -> OpenSource:
    """Open native review arrays after checking manifest identity and array metadata."""
    manifest = json.loads((source_dir / 'source.json').read_text())
    if manifest['schema'] != 'dino-zarr-source-v1' or manifest_digest(manifest) != manifest['source_sha256']:
        raise ValueError('Corrected display source manifest identity differs')
    arrays = {}
    for name, spec in {'labels': manifest['labels'], **manifest['channels']}.items():
        array, resolved = resolve_array(Path(spec['root']), array_path=spec['array'] or None)
        if (metadata_digest(Path(spec['root']), resolved) != spec['metadata_sha256']
                or list(array.shape) != spec['shape'] or list(array.chunks) != spec['chunks']
                or str(array.dtype) != spec['dtype']):
            raise ValueError(f'Corrected display source metadata differs for {name}')
        arrays[name] = array
    return OpenSource(manifest, arrays.pop('labels'), arrays)
