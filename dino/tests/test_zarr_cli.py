from __future__ import annotations

import sys
from types import SimpleNamespace

from click.testing import CliRunner

from dino import zarr_cli


def test_zarr_cli_dispatches_subcommand_arguments(monkeypatch) -> None:
    calls: list[list[str]] = []

    def command() -> None:
        calls.append(sys.argv.copy())

    monkeypatch.setattr(zarr_cli, "import_module", lambda name: SimpleNamespace(main=command))

    result = CliRunner().invoke(zarr_cli.main, ["prepare", "--workers", "3"])

    assert result.exit_code == 0
    assert calls == [["dino-zarr prepare", "--workers", "3"]]


def test_zarr_cli_help_lists_documented_commands() -> None:
    result = CliRunner().invoke(zarr_cli.main, ["--help"])

    assert result.exit_code == 0
    for command in ("prepare", "barrage", "train", "infer", "periodic"):
        assert command in result.output
