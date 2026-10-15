from __future__ import annotations

import fcntl
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

from dino import clean_labels
from dinoapp.label_store import update_label


def test_backup_preserves_original_bytes(monkeypatch, tmp_path):
    original = b"cell,brdu,edu,note\r\nr|1,0,0,first\r\nr|1,1,0,last\r\n"
    (tmp_path / "labels.csv").write_bytes(original)
    temporary = tmp_path / "labels.csv.tmp"
    temporary.write_bytes(b"existing temporary")
    monkeypatch.setattr("sys.argv", ["clean-labels", "--barrage-dir", str(tmp_path), "--apply"])
    clean_labels.main()
    assert next(tmp_path.glob("labels.csv.bak_*")).read_bytes() == original
    assert temporary.read_bytes() == b"existing temporary"
    assert pd.read_csv(tmp_path / "labels.csv").to_dict("records") == [{"cell": "r|1", "brdu": 1, "edu": 0}]


def test_cleaner_holds_reviewer_lock_and_preserves_reviewer_edit(monkeypatch, tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("cell,brdu,edu\nr|1,0,0\n")
    monkeypatch.setattr("sys.argv", ["clean-labels", "--barrage-dir", str(tmp_path), "--apply"])
    read_done, resume, writer_waiting = threading.Event(), threading.Event(), threading.Event()
    flock = fcntl.flock

    def observed_flock(file, operation):
        if threading.current_thread().name.startswith("reviewer"):
            writer_waiting.set()
        return flock(file, operation)

    monkeypatch.setattr(fcntl, "flock", observed_flock)
    read_csv = pd.read_csv

    def paused_read(*args, **kwargs):
        result = read_csv(*args, **kwargs)
        if threading.current_thread().name.startswith("cleaner"):
            read_done.set()
            assert resume.wait(10), "Cleaner was not released"
        return result

    monkeypatch.setattr(clean_labels.pd, "read_csv", paused_read)
    with (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="cleaner") as cleaner,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="reviewer") as reviewer,
    ):
        cleaning = cleaner.submit(clean_labels.main)
        try:
            assert read_done.wait(10), "Cleaner did not read labels"
            with path.with_suffix(".csv.lock").open("a") as lock:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            editing = reviewer.submit(update_label, path, "r|1", (1, 0))
            assert writer_waiting.wait(10), "Reviewer did not attempt the shared lock"
        finally:
            resume.set()
        cleaning.result(timeout=10)
        editing.result(timeout=10)
    assert read_csv(path)["brdu"].tolist() == [1]


def test_backup_collision_preserves_existing_backup_and_labels(monkeypatch, tmp_path):
    from datetime import datetime

    class FixedTime:
        @staticmethod
        def now():
            return datetime(2026, 1, 1)

    path = tmp_path / "labels.csv"
    original = b"cell,brdu,edu\nr|1,0,0\n"
    path.write_bytes(original)
    backup = tmp_path / "labels.csv.bak_20260101_000000_000000"
    backup.write_bytes(b"previous backup")
    monkeypatch.setattr(clean_labels, "datetime", FixedTime)
    monkeypatch.setattr("sys.argv", ["clean-labels", "--barrage-dir", str(tmp_path), "--apply"])
    with pytest.raises(FileExistsError):
        clean_labels.main()
    assert backup.read_bytes() == b"previous backup"
    assert path.read_bytes() == original

