from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from dino import cli


def test_cli_dispatches_subcommand_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def command() -> None:
        calls.append(sys.argv.copy())

    monkeypatch.setattr(cli, "import_module", lambda name: SimpleNamespace(main=command))

    result = CliRunner().invoke(cli.main, ["train", "--seed", "4"])

    assert result.exit_code == 0
    assert calls == [["dino train", "--seed", "4"]]


@pytest.mark.parametrize("args", [["--server.port", "8502"], ["--help"], ["--", "--server.port", "8502"]])
def test_cli_dispatches_label_to_current_interpreter(monkeypatch, args) -> None:
    calls = []
    monkeypatch.setattr(cli, "find_spec", lambda name: object() if name == "dinoapp" else None)
    monkeypatch.setattr(cli.os, "execv", lambda executable, argv: calls.append((executable, argv)))
    result = CliRunner().invoke(cli.main, ["label", *args])
    assert result.exit_code == 0
    forwarded = args[1:] if args[0] == "--" else args
    assert calls == [(sys.executable, [sys.executable, "-m", "dinoapp", *forwarded])]


def test_cli_label_reports_missing_reviewer(monkeypatch) -> None:
    monkeypatch.setattr(cli, "find_spec", lambda name: None)
    result = CliRunner().invoke(cli.main, ["label"])
    assert result.exit_code == 1
    assert "Install dinoapp in this Python environment" in result.output


def test_label_process_preserves_arguments_environment_directory_and_exit_status(tmp_path):
    (tmp_path / "dinoapp.py").write_text(
        "import json, os, sys\n"
        "print(json.dumps([sys.argv[1:], os.environ['DINO_REVIEW_BARRAGE_DIR'], os.getcwd()]))\n"
        "sys.exit(7)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", "from dino.cli import main; main()", "label", "--server.port", "8502"],
        cwd=tmp_path,
        env={**os.environ, "DINO_REVIEW_BARRAGE_DIR": "/review/barrage"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 7, result.stderr
    assert json.loads(result.stdout) == [["--server.port", "8502"], "/review/barrage", str(tmp_path)]


def test_cli_rejects_unknown_command() -> None:
    result = CliRunner().invoke(cli.main, ["unknown"])

    assert result.exit_code == 2
    assert "No such command 'unknown'" in result.output


def test_cli_help_lists_consolidated_commands() -> None:
    result = CliRunner().invoke(cli.main, ["--help"])

    assert result.exit_code == 0
    for command in ("barrage", "clean-labels", "train", "infer", "label", "run", "roc", "periodic"):
        assert command in result.output
