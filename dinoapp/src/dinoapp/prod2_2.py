"""Prod2-2 paths for the shared native ventricular-field reviewer."""

from pathlib import Path

from dinoapp.native import render_field as render_field
from dinoapp.prod3_2 import FieldSource as CorrectedFieldSource

ROOT = Path('/working/eduseg/prod2-2')
REVIEW_ROOT = ROOT / 'dino-prod2-2'


class FieldSource(CorrectedFieldSource):
    """Review native cells with the recorded correction, surface, and calls."""

    dataset_name = 'Prod2-2'
    dataset_root = ROOT
    review_directory = 'dino-prod2-2'
    reembedding_run = ROOT / 'brdu-dapi-subtraction-v1/reembedding'
    surface_coordinates = ROOT / 'spatial-workflow-v1/vz-coordinates/cells.parquet'
    display_name = 'Prod2-2 native'
    cell_prefix = '234-2-514561638'
    prediction_pointer = REVIEW_ROOT / 'cell-review/current.json'
