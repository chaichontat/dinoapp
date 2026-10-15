from pathlib import Path

import numpy as np
import pytest
import zarr

from dino import workspace_inputs


def test_workspace_roots_preserve_default_search_order() -> None:
    assert workspace_inputs.WORKSPACE_ROOT_BASES == (
        Path('/working'), Path('/working/chaichontat'), Path.home() / 'nvme'
    )


def test_workspace_root_uses_first_existing_candidate_and_reports_missing_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    roots = (tmp_path / 'working', tmp_path / 'nvme')
    monkeypatch.setattr(workspace_inputs, 'WORKSPACE_ROOT_BASES', roots)
    (roots[1] / 'DS1').mkdir(parents=True)
    assert workspace_inputs.resolve_workspace_root('DS1') == roots[1] / 'DS1'
    (roots[0] / 'DS1').mkdir(parents=True)
    assert workspace_inputs.resolve_workspace_root('DS1') == roots[0] / 'DS1'
    with pytest.raises(FileNotFoundError) as error:
        workspace_inputs.resolve_workspace_root('missing')
    assert str(error.value) == (
        f"workspace_root not found for dataset 'missing'; checked: {roots[0] / 'missing'}, {roots[1] / 'missing'}"
    )


def test_legacy_cell_ids_support_dataset_namespaces() -> None:
    assert workspace_inputs.cell_id_roi_and_label('2r|39329') == ('2r', 39329)
    assert workspace_inputs.cell_id_roi_and_label('20251001_JaxA3_Coro11:2r|39329') == ('2r', 39329)
    assert workspace_inputs.cell_id_roi_and_label('namespace:dataset:2r|39329') == ('2r', 39329)


@pytest.mark.parametrize('cell', ['2r', '2r|1|2'])
def test_legacy_cell_ids_reject_missing_separator_or_invalid_integer(cell: str) -> None:
    with pytest.raises(ValueError):
        workspace_inputs.cell_id_roi_and_label(cell)


@pytest.mark.parametrize('trailing_channel', [False, True])
@pytest.mark.parametrize(('x_center', 'y_center', 'expected'), [
    (0, 0, [[0, 0, 0], [0, 1, 2], [0, 5, 6]]),
    (3, 2, [[7, 8, 0], [11, 12, 0], [0, 0, 0]]),
    (2, 1, [[2, 3, 4], [6, 7, 8], [10, 11, 12]]),
])
def test_centered_segmentation_labels_clip_pad_and_preserve_all_labels(
    trailing_channel: bool, x_center: int, y_center: int, expected: list[list[int]]
) -> None:
    data = np.arange(1, 13, dtype=np.uint16).reshape(1, 3, 4)
    if trailing_channel:
        data = np.stack((data, np.full_like(data, 99)), axis=-1)
    crop = workspace_inputs.centered_segmentation_labels(
        zarr.array(data), z_index=0, x_center=x_center, y_center=y_center, size=3
    )
    np.testing.assert_array_equal(crop, expected)
    assert crop.dtype == np.uint16


def test_centered_segmentation_labels_preserve_read_errors_and_disjoint_no_read() -> None:
    invalid_layout = zarr.zeros((1, 3, 4, 1, 1))
    with pytest.raises(ValueError, match='Unexpected segmentation array ndim=5'):
        workspace_inputs.centered_segmentation_labels(
            invalid_layout, z_index=0, x_center=1, y_center=1, size=2
        )
    np.testing.assert_array_equal(
        workspace_inputs.centered_segmentation_labels(
            invalid_layout, z_index=99, x_center=20, y_center=20, size=2
        ), np.zeros((2, 2), dtype=np.int32)
    )
    with pytest.raises(IndexError):
        workspace_inputs.centered_segmentation_labels(
            zarr.zeros((1, 3, 4)), z_index=99, x_center=1, y_center=1, size=2
        )
