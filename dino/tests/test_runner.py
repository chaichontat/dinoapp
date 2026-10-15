from __future__ import annotations

from pathlib import Path
from typing import Any

from dino import runner


def test_existing_output_is_validated_by_inference(tmp_path: Path, monkeypatch: Any) -> None:
    module = runner
    module.DEFAULT_OUTPUT_DIR = tmp_path
    output = module._output_path_for_dataset("ds1")
    output.write_bytes(b"stale")
    cfg = module.RunnerConfig(
        h5ad_path=tmp_path / "input.h5ad",
        model_path=tmp_path / "model.joblib",
        datasets=("ds1",),
        batch_size=8,
        num_workers=2,
        skip_existing=True,
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(module, "_parse_args", lambda: cfg)
    monkeypatch.setattr(module, "_datasets_to_run", lambda _cfg: ["ds1"])
    monkeypatch.setattr(module.subprocess, "run", lambda command, check: commands.append(command))

    module.main()

    assert len(commands) == 1
    assert "--skip-existing" in commands[0]
