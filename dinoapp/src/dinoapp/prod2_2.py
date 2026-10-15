"""Prod2-2 layout for the shared native ventricular-field reviewer."""

from pathlib import Path

from dinoapp.native import render_field as render_field
from dinoapp.prod3_2 import FieldSource as CorrectedFieldSource


class FieldSource(CorrectedFieldSource):
    """Review native cells with the recorded correction, surface, and calls."""

    @property
    def surface_coordinates(self) -> Path:
        return self.root / 'spatial-workflow-v1/vz-coordinates/cells.parquet'
