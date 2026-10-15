"""Measure retained raw 750 nm pixels using the established RHU2 nuclear masks."""
from __future__ import annotations

import argparse
import filecmp
import json
from pathlib import Path
import shutil
import time

import anndata as ad
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, find_objects, gaussian_filter, gaussian_laplace, map_coordinates
import tifffile
import zarr

from fishtools.preprocess.tileconfig import TileConfiguration


RAW_ROOT = Path('data/residual_brdu_raw')
OUTPUT = Path('thesis/analysis/shared/residual-brdu/measurements')
CROP = 40
DOWNSAMPLE = 2
ROLES = {'round_0_brdu': 'brdu', 'final_rna_readout': 'final', 'post_readout_edu': 'edu'}


def read_waveform(path: Path):
    with tifffile.TiffFile(path) as tif:
        metadata = json.loads(tif.pages[0].description)
        waveform = json.loads(metadata['waveform'])
        frames = np.flatnonzero(waveform['ilm750']['sequence'])
        if len(frames) != 34 or tif.series[0].shape[-2:] != (2048, 2048):
            raise ValueError(f'Unexpected retained acquisition geometry: {path}')
    return metadata, waveform, frames


def chromatic_pull(coefficients: np.ndarray, displacement: np.ndarray, shape: tuple[int, int]):
    """Return output-to-input XY affine, matching fishtools.chromatic.Affine.

    SimpleITK's composite adds the inter-round translation after the chromatic
    affine; the affine center itself includes that translation.
    """
    matrix = coefficients[:4].reshape(2, 2)
    shift = -np.asarray(displacement, dtype=float)
    center = (np.array(shape[::-1]) - 1) / 2 + shift
    offset = center - matrix @ center + coefficients[-2:] + shift
    return matrix, offset


def raw_mask_coordinates(matrix: np.ndarray, offset: np.ndarray, origin: np.ndarray,
                         shape: tuple[int, int]):
    """Map raw pixel centers to native fused pixel centers without changing ADU."""
    yy, xx = np.indices(shape, dtype=np.float32)
    raw_xy = np.stack([xx.ravel(), yy.ravel()])
    registered = np.linalg.solve(matrix, raw_xy - offset[:, None])
    native = (registered - CROP - .5) / DOWNSAMPLE + origin[:, None]
    return native[::-1].reshape((2, *shape)), registered.reshape((2, *shape))


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 100 or np.std(a[finite]) == 0 or np.std(b[finite]) == 0:
        raise ValueError('Insufficient varying pixels for registration validation')
    return float(np.corrcoef(a[finite].ravel(), b[finite].ravel())[0, 1])


def highpass(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    return image - gaussian_filter(image, 3)


def tile_origin(dataset: str, roi: str, tile: int, raw_root: Path):
    path = raw_root / dataset / 'registration' / f'stitch--{roi}' / 'TileConfiguration.registered.txt'
    config = TileConfiguration.from_file(path).downsample(DOWNSAMPLE).df.to_pandas()
    row = config.loc[config['index'] == tile].iloc[0]
    return row[['x', 'y']].to_numpy(float) - config[['x', 'y']].min().to_numpy(float)


def acquisition_transform(item: dict, raw_root: Path):
    dataset, roi, tile = item['dataset'], str(item['roi']), item['tile']
    source = Path('/working/chaichontat') / dataset / 'analysis'
    chromatic = source / ('output/chromatic' if dataset == '20250929_JaxA3_Coro4' else 'deconv/chromatic')
    coefficients = np.loadtxt(chromatic / '560to750.txt')
    prefix = Path(item['relative_path']).name.rsplit('-', 1)[0]
    if prefix in {'6_14_22', '1_9_17'}:
        codebook = 'cs_base'
    elif dataset == '20250929_JaxA3_Coro4':
        codebook = 'edu'
    else:
        codebook = 'pi'
    path = raw_root / dataset / 'registration' / f'shifts--{roi}+{codebook}' / f'shifts-{tile:04d}.json'
    entry = json.loads(path.read_text())[prefix]
    matrix, offset = chromatic_pull(coefficients, np.array(entry['shifts']), (2048, 2048))
    return matrix, offset, {'chromatic_source': str(chromatic / '560to750.txt'),
                            'shift_source': str(path), 'shift': entry,
                            'pull_matrix_xy': matrix.tolist(), 'pull_offset_xy': offset.tolist()}


def validate_marker(path: Path, frames: np.ndarray, matrix: np.ndarray, offset: np.ndarray,
                    origin: np.ndarray, fused, channel: int, output: Path, stem: str):
    """Compare registered raw marker structure to native fused marker at every Z."""
    # A central window avoids fusion overlaps and the transformed tile boundary.
    size, start = 512, 236
    yy, xx = np.indices((size, size), dtype=np.float32)
    native_xy = np.stack([(xx + start).ravel(), (yy + start).ravel()])
    registered_xy = native_xy * DOWNSAMPLE + CROP + .5
    raw_xy = matrix @ registered_xy + offset[:, None]
    raw_coords = raw_xy[::-1].reshape((2, size, size))
    gx, gy = origin + start
    x0, y0 = int(np.floor(gx)), int(np.floor(gy))
    fixed = np.asarray(fused[:, y0:y0 + size + 2, x0:x0 + size + 2, channel], dtype=np.float32)
    native_coords = np.stack([yy + gy - y0, xx + gx - x0])
    moving, reference = [], []
    with tifffile.TiffFile(path) as tif:
        for z, frame in enumerate(frames):
            plane = tif.pages[int(frame)].asarray()
            moving.append(highpass(map_coordinates(plane.astype(np.float32), raw_coords, order=1,
                                                    mode='constant', cval=np.nan)))
            reference.append(highpass(map_coordinates(fixed[z], native_coords, order=1)))
    moving, reference = np.stack(moving), np.stack(reference)
    # Decimate only the diagnostic correlation computation, after high pass.
    left, right = moving[:, ::4, ::4], reference[:, ::4, ::4]
    centered_left = left.reshape(34, -1)
    centered_right = right.reshape(34, -1)
    centered_left -= centered_left.mean(axis=1, keepdims=True)
    centered_right -= centered_right.mean(axis=1, keepdims=True)
    zcorr = centered_left @ centered_right.T
    zcorr /= np.sqrt((centered_left ** 2).sum(axis=1)[:, None]
                     * (centered_right ** 2).sum(axis=1)[None, :])
    lag_scores = {str(lag): float(np.diag(zcorr, lag).mean()) for lag in range(-4, 5)}
    best_lag = max(lag_scores, key=lag_scores.get)
    report = {'same_z_correlation': float(np.diag(zcorr).mean()), 'best_lag': int(best_lag),
              'z_lag_correlations': lag_scores, 'z_correlations': zcorr.tolist(),
              'source': str(path), 'channel': int(channel),
              'method': 'established XY transform; plane-wise Gaussian high pass sigma=3 native pixels; no Z shift'}
    np.savez_compressed(output / f'{stem}_registration.npz', raw_highpass=moving[:, ::2, ::2],
                        fused_highpass=reference[:, ::2, ::2], z_correlations=zcorr)
    return report


def validate_final_fid(path: Path, displacement: list[float], reference_path: Path,
                       output: Path, stem: str):
    """Test a saved RNA round shift on independent 405 nm registration beads."""
    with tifffile.TiffFile(path) as tif:
        raw = np.maximum(tif.pages[-1].asarray(), tif.pages[-2].asarray()).astype(np.float32)
    reference = tifffile.imread(reference_path).astype(np.float32)
    if reference.shape != (1968, 1968):
        raise ValueError(f'Reference fiducial does not confirm crop40: {reference_path}')
    # fids--ROI stores the reference LoG image used by run_fiducial.
    raw = -gaussian_laplace(raw, 3)
    yy, xx = np.indices(reference.shape, dtype=np.float32)
    before = map_coordinates(raw, [yy + CROP, xx + CROP], order=1)
    after = map_coordinates(raw, [yy + CROP - displacement[1], xx + CROP - displacement[0]], order=1)
    trim = (slice(80, -80), slice(80, -80))
    report = {'fiducial_before_correlation': correlation(before[trim], reference[trim]),
              'fiducial_after_correlation': correlation(after[trim], reference[trim]),
              'reference_source': str(reference_path), 'source': str(path),
              'method': 'saved RNA round XY shift tested on independent maximum of two terminal 405 nm frames; LoG sigma 3; no chromatic correction for 405 nm'}
    np.savez_compressed(output / f'{stem}_final_fiducial.npz', raw_before=before[::4, ::4],
                        raw_after=after[::4, ::4], reference=reference[::4, ::4])
    return report


def background_requests(labels: np.ndarray, selected: pd.DataFrame, x0: int, y0: int):
    """Five native-pixel XY rings around each nuclear mask, excluding every nucleus."""
    lookup = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    lookup[selected.label.to_numpy(int)] = np.arange(1, len(selected) + 1)
    objects = find_objects(lookup[labels], max_label=len(selected))
    requests = [[] for _ in range(labels.shape[0])]
    for index, (row, box) in enumerate(zip(selected.itertuples(), objects)):
        if box is None:
            continue
        zs, ys, xs = box
        ys = slice(max(0, ys.start - 5), min(labels.shape[1], ys.stop + 5))
        xs = slice(max(0, xs.start - 5), min(labels.shape[2], xs.stop + 5))
        patch = labels[zs, ys, xs]
        focal = patch == row.label
        ring = binary_dilation(focal, structure=np.ones((1, 3, 3), dtype=bool), iterations=5)
        ring &= patch == 0
        zz, yy, xx = np.nonzero(ring)
        for z in np.unique(zz):
            keep = zz == z
            requests[int(z + zs.start)].append((index, yy[keep] + ys.start + y0,
                                                xx[keep] + xs.start + x0))
    return requests


def extract_acquisition(path: Path, frames: np.ndarray, matrix: np.ndarray, offset: np.ndarray,
                        origin: np.ndarray, labels: np.ndarray, selected: pd.DataFrame,
                        x0: int, y0: int, requests):
    native_coordinates, registered = raw_mask_coordinates(matrix, offset, origin, (2048, 2048))
    native_coordinates[0] -= y0
    native_coordinates[1] -= x0
    footprint = ((registered[0] >= CROP) & (registered[0] < 2048 - CROP)
                 & (registered[1] >= CROP) & (registered[1] < 2048 - CROP))
    wanted = set(selected.label.astype(int))
    nuclear = {label: [] for label in wanted}
    background = [[] for _ in range(len(selected))]
    with tifffile.TiffFile(path) as tif:
        for z, frame in enumerate(frames):
            raw = tif.pages[int(frame)].asarray()
            native_labels = map_coordinates(labels[z], native_coordinates, order=0, mode='grid-constant', cval=0)
            native_labels[~footprint] = 0
            keep = np.isin(native_labels, selected.label.to_numpy())
            ids, values = native_labels[keep], raw[keep]
            order = np.argsort(ids, kind='stable')
            ids, values = ids[order], values[order]
            unique, starts = np.unique(ids, return_index=True)
            for label, values in zip(unique, np.split(values, starts[1:])):
                nuclear[int(label)].append(values)
            for index, yy, xx in requests[z]:
                native_xy = np.stack([xx - origin[0], yy - origin[1]])
                raw_xy = matrix @ (native_xy * DOWNSAMPLE + CROP + .5) + offset[:, None]
                indices = np.rint(raw_xy).astype(int)
                support = (indices >= 0).all(axis=0) & (indices < 2048).all(axis=0)
                background[index].append(raw[indices[1, support], indices[0, support]])
    measurements = []
    for index, row in enumerate(selected.itertuples()):
        pixels = np.concatenate(nuclear[int(row.label)])
        bg = np.concatenate(background[index]) if background[index] else np.empty(0, np.uint16)
        measurements.append({'median_raw_adu': float(np.median(pixels)), 'raw_voxels': len(pixels),
                             'saturated_voxels': int(np.count_nonzero(pixels == 65535)),
                             'background_raw_adu': float(np.median(bg)) if len(bg) else np.nan,
                             'background_voxels': len(bg)})
    return pd.DataFrame(measurements, index=selected.index)


def covered_native_masks(matrix: np.ndarray, offset: np.ndarray, origin: np.ndarray,
                         labels: np.ndarray, selected: pd.DataFrame, x0: int, y0: int):
    """Count native mask voxels whose whole pixel area lies in acquired support."""
    yy, xx = np.indices(labels.shape[-2:], dtype=float)
    xy = np.stack([(xx + x0 - origin[0]).ravel(), (yy + y0 - origin[1]).ravel()])
    registered = xy * DOWNSAMPLE + CROP + .5
    raw = matrix @ registered + offset[:, None]
    radius = np.abs(matrix).sum(axis=1)[:, None]
    support = ((raw - radius >= 0).all(axis=0) & (raw + radius < 2048).all(axis=0)
               & (registered - 1 >= CROP).all(axis=0)
               & (registered + 1 < 2048 - CROP).all(axis=0)).reshape(labels.shape[-2:])
    ids, counts = np.unique(labels[:, support], return_counts=True)
    return selected.label.map(dict(zip(ids, counts))).fillna(0).astype(int)


def run(raw_root: Path, output: Path, diagnostics_only: bool = False):
    output.mkdir(parents=True, exist_ok=True)
    selections = [item for item in json.loads((raw_root / 'selection.json').read_text())
                  if item['relative_path'].endswith('.tif')]
    items = pd.DataFrame(selections)
    release = ad.read_h5ad('/working/chaichontat/ccf_release_rhu2_map2/cohort/cells.h5ad', backed='r')
    areas = release.obs['area'].copy()
    release.file.close()
    reports, acquisitions, rows = [], [], []
    for (dataset, roi, tile), tile_items in items.groupby(['dataset', 'roi', 'tile'], sort=True):
        roi = str(roi)
        tile = int(tile)
        tick = time.monotonic()
        origin = tile_origin(dataset, roi, tile, raw_root)
        source = Path('/working/chaichontat') / dataset / 'analysis/deconv'
        fused = zarr.open_array(source / f'stitch--{roi}+edu/fused.zarr', mode='r')
        masks = zarr.open_array(source / f'stitch--{roi}+pi/output_segmentation-sam_postproc_s1-2-2_v500.zarr', mode='r')
        cohort = pd.read_parquet(Path('/working/chaichontat/rhu2-barrage-review/spatial-model-v1') / f'{dataset}__{roi}.parquet')
        # Native pixel geometry is 984x984 after crop40 and downsample2.
        selected = cohort.loc[(cohort.x > origin[0] + 50) & (cohort.x < origin[0] + 934)
                              & (cohort.y > origin[1] + 50) & (cohort.y < origin[1] + 934)].copy()
        selected['area'] = areas.reindex(selected.cell).to_numpy()
        if selected.area.isna().any():
            raise ValueError('Selected cortical cell absent from current release')
        x0, y0 = np.floor(origin - 32).astype(int)
        labels = np.asarray(masks[:, y0:y0 + 1048, x0:x0 + 1048])
        ids, counts = np.unique(labels, return_counts=True)
        selected['mask_voxels'] = selected.label.map(dict(zip(ids, counts)))
        selected['coverage_fraction'] = selected.mask_voxels / selected.area
        selected['valid'] = selected.coverage_fraction == 1
        selected['tile'] = tile
        transforms = {}
        tile_reports = []
        for item in tile_items.to_dict('records'):
            role = ROLES[item['role']]
            path = raw_root / item['relative_path']
            metadata, waveform, frames = read_waveform(path)
            matrix, offset, provenance = acquisition_transform(item, raw_root)
            transforms[role] = (path, frames, matrix, offset)
            acquisitions.append({'dataset': dataset, 'roi': roi, 'tile': tile, 'role': role,
                                 'source': str(path), 'power750': waveform['ilm750']['power'],
                                 'exposure_seconds': waveform['exposure_time'], 'stage_z_um': metadata['z'],
                                 'stage_xy_um': metadata['position'], 'z_spacing_um': .4,
                                 'pixel_size_um': 1 / metadata['resolution'], 'frames750': frames.tolist(),
                                 'origin_xy_native': origin.tolist(), **provenance})
            if role != 'final':
                report = validate_marker(path, frames, matrix, offset, origin, fused,
                                         list(fused.attrs['key']).index(role), output,
                                         f'{dataset}_roi{roi}_tile{tile:04d}_{role}')
                message = f'{dataset} roi{roi} tile{tile:04d} {role}: same_z={report["same_z_correlation"]:.3f} best_lag={report["best_lag"]}'
            else:
                report = validate_final_fid(path, provenance['shift']['shifts'],
                                            source / f'fids--{roi}/fids-{tile:04d}.tif',
                                            output, f'{dataset}_roi{roi}_tile{tile:04d}')
                message = f'{dataset} roi{roi} tile{tile:04d} final fid: {report["fiducial_before_correlation"]:.3f}→{report["fiducial_after_correlation"]:.3f}'
            report.update(dataset=dataset, roi=roi, tile=tile, role=role)
            reports.append(report)
            tile_reports.append(report)
            print(message, flush=True)
        (output / 'registration_qc.json').write_text(json.dumps(reports, indent=2) + '\n')
        (output / 'acquisitions.json').write_text(json.dumps(acquisitions, indent=2) + '\n')
        if diagnostics_only:
            continue
        if any(report.get('best_lag', 0) != 0 for report in tile_reports):
            raise ValueError(f'Existing XY-only registration does not establish same-Z correspondence for {dataset}:{roi}:{tile}')
        final_report = next(report for report in tile_reports if report['role'] == 'final')
        if final_report['fiducial_after_correlation'] <= final_report['fiducial_before_correlation']:
            raise ValueError(f'Final RNA saved shift fails independent fiducial validation: {dataset}:{roi}:{tile}')
        requests = background_requests(labels, selected, x0, y0)
        for role, (path, frames, matrix, offset) in transforms.items():
            selected[f'{role}_covered_mask_voxels'] = covered_native_masks(
                matrix, offset, origin, labels, selected, x0, y0)
            selected['valid'] &= selected[f'{role}_covered_mask_voxels'] == selected.area
            measured = extract_acquisition(path, frames, matrix, offset, origin, labels, selected,
                                           x0, y0, requests)
            selected = selected.join(measured.add_prefix(f'{role}_'))
        rows.append(selected)
        selected.to_parquet(output / f'{dataset}_roi{roi}_tile{tile:04d}_cells.parquet', index=False)
        print(f'Finished {dataset}:{roi}:{tile} in {time.monotonic()-tick:.1f}s', flush=True)
    if not diagnostics_only:
        result = pd.concat(rows, ignore_index=True)
        if result.cell.duplicated().any():
            raise ValueError('Selected fields overlap; cell-level analysis needs one field per cell')
        result.to_parquet(output / 'cells.parquet', index=False)
        (output / 'complete.json').write_text(json.dumps({
            'cells': len(result), 'valid_cells': int(result.valid.sum()),
            'method': 'original integer raw750 pixels assigned to nearest native nuclear label through inverse established chromatic/XY transform; no intensity resampling',
            'background': 'five native pixels (1.08um) XY ring in same occupied Z planes; excludes every nuclear label; nearest original raw pixel sample',
            'crop_raw_pixels': CROP, 'native_xy_downsample': DOWNSAMPLE, 'z_shift': 0,
            'camera_offset_subtraction': None, 'power_conversion': None,
            'incomplete_mask_exclusions': int((~result.valid).sum())}, indent=2) + '\n')


def first_round_images(raw_root: Path, output: Path):
    """Export one predetermined matched field from BrdU, first RNA and EdU."""
    from thesis_figures.section_array import section_compass
    from thesis_figures.section_orientation import section_display_transform

    dataset, roi, tile, z = '20250929_JaxA3_Coro4', '4', 324, 17
    output.mkdir(parents=True, exist_ok=True)
    source_root = Path('/mnt/solarplex/raw/D/mermap') / dataset
    relative = Path(dataset) / f'1_9_17--{roi}/1_9_17-{tile:04d}.tif'
    source, target = source_root / relative.relative_to(dataset), raw_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)
    if not filecmp.cmp(source, target, shallow=False):
        raise ValueError('Retrieved first RNA stack differs from its retained source')
    toml_name = '1_9_17-autoexposed.toml'
    toml_target = raw_root / dataset / toml_name
    shutil.copy2(source_root / toml_name, toml_target)
    if not filecmp.cmp(source_root / toml_name, toml_target, shallow=False):
        raise ValueError('Retrieved first RNA waveform differs from source')
    work = Path('/working/chaichontat') / dataset / 'analysis'
    pose_path = work / f'output/ccf-transforms/{roi}/p1_landmarks.json'
    pose = json.loads(pose_path.read_text())
    masks = zarr.open_array(work / f'deconv/stitch--{roi}+pi/output_segmentation-sam_postproc_s1-2-2_v500.zarr', mode='r')
    rotation = section_display_transform(masks.shape[1:], .216, pose).get_matrix()[:2, :2]
    compass, residual_deg = section_compass(pose_path)
    origin = tile_origin(dataset, roi, tile, raw_root)
    center = origin + 491.5
    size, spacing = 926, .108
    yy, xx = np.indices((size, size), dtype=np.float64)
    offsets_um = np.stack([(xx.ravel() - (size - 1) / 2) * spacing,
                           (yy.ravel() - (size - 1) / 2) * spacing])
    native_xy = np.linalg.solve(rotation, offsets_um) / .216 + center[:, None]
    registered = (native_xy - origin[:, None]) * 2 + CROP + .5
    crops, manifests = {}, []
    for key, prefix, role in [('brdu_raw_adu', 'wga_brdu', 'round_0_brdu'),
                              ('first_rna_raw_adu', '1_9_17', 'first_rna_readout'),
                              ('edu_raw_adu', 'edu', 'post_readout_edu')]:
        relative = Path(dataset) / f'{prefix}--{roi}/{prefix}-{tile:04d}.tif'
        path = raw_root / relative
        item = {'dataset': dataset, 'roi': roi, 'tile': tile, 'role': role,
                'relative_path': str(relative)}
        metadata, waveform, frames = read_waveform(path)
        matrix, offset, provenance = acquisition_transform(item, raw_root)
        raw_xy = matrix @ registered + offset[:, None]
        if (raw_xy < 0).any() or (raw_xy >= 2047).any():
            raise ValueError('Predetermined display crop leaves the acquired raw field')
        with tifffile.TiffFile(path) as tif:
            plane = tif.pages[int(frames[z])].asarray().astype(np.float32)
        crops[key] = map_coordinates(plane, raw_xy[::-1].reshape((2, size, size)), order=1)
        manifests.append({'key': key, 'source': str(path), 'retained_source': str(source_root / relative.relative_to(dataset)),
                          'source_bytes': path.stat().st_size, 'raw_frame': int(frames[z]), 'z_index': z,
                          'z_offset_um': waveform['z_positions'][int(frames[z])],
                          'stage_z_um': metadata['z'], 'power750': waveform['ilm750']['power'],
                          'exposure_seconds': waveform['exposure_time'],
                          'raw_pixel_spacing_um': 1 / metadata['resolution'],
                          'display_quantiles_adu': dict(zip(['p1', 'p50', 'p99', 'p99_8'],
                                                          np.percentile(crops[key], [1, 50, 99, 99.8]).tolist())),
                          **provenance})
        if prefix == '1_9_17':
            qc = validate_final_fid(path, provenance['shift']['shifts'],
                                    work / f'deconv/fids--{roi}/fids-{tile:04d}.tif', output,
                                    f'{dataset}_roi{roi}_tile{tile:04d}_first_round')
            if qc['fiducial_after_correlation'] <= qc['fiducial_before_correlation']:
                raise ValueError('First RNA saved shift fails its independent405 fiducial check')
    np.savez_compressed(output / 'paired_crop.npz', **crops)
    label_yx = (native_xy[::-1] - np.floor(native_xy[::-1].min(axis=1))[:, None]).reshape((2, size, size))
    mins = np.floor(native_xy[::-1].min(axis=1)).astype(int)
    maxs = np.ceil(native_xy[::-1].max(axis=1)).astype(int) + 1
    labels = np.asarray(masks[z, mins[0]:maxs[0], mins[1]:maxs[1]])
    sampled = map_coordinates(labels, label_yx, order=0, mode='grid-constant', cval=0)
    occupancy = float(np.mean(sampled != 0))
    if occupancy < .05:
        raise ValueError('Predetermined central microscopy crop has insufficient nuclear coverage')
    manifest = {'dataset': dataset, 'roi': roi, 'tile': tile, 'z_index': z,
                'crop_selection': 'predetermined tile center; not chosen by residual signal',
                'native_center_xy': center.tolist(), 'shape_yx': [size, size],
                'pixel_spacing_um': spacing, 'span_um': size * spacing,
                'pose_source': str(pose_path), 'pose_prior_rotation_deg': pose['prior_rotation_deg'],
                'pose_prior_flip_x': pose['prior_flip_x'], 'pose_residual_rotation_deg': residual_deg,
                'native_to_display_rotation_xy': rotation.tolist(), 'dorsal_up': True,
                'compass_labels': list(compass), 'image_origin': 'upper',
                'display_resampling': 'linear at original 0.108 um spacing; identical physical output grid for all three acquisitions',
                'intensity': 'raw ADU; no background/camera subtraction or power/exposure conversion',
                'nuclear_mask_occupied_fraction': occupancy,
                'first_round_fiducial_qc': qc, 'acquisitions': manifests,
                'retrieval': {'byte_equal': True, 'local_tiff': str(target), 'waveform': str(toml_target)}}
    (output / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Paired raw microscopy: {output}/paired_crop.npz; occupancy={occupancy:.3f}; firstRNAfid afterr={qc["fiducial_after_correlation"]:.3f}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=RAW_ROOT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--diagnostics-only', action='store_true')
    parser.add_argument('--first-round-images', action='store_true')
    args = parser.parse_args()
    if args.first_round_images:
        first_round_images(args.raw_root, args.output)
    else:
        run(args.raw_root, args.output, args.diagnostics_only)
