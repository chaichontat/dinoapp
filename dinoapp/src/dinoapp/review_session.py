"""Per-target review history, annotation persistence, backups, and journals."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st

from dinoapp.label_store import Label
from dinoapp.sources import AnnotationFiles, Edit


def get_context(source_id: str) -> dict[str, Any]:
    contexts = st.session_state.setdefault("review_contexts", {})
    return contexts.setdefault(
        source_id,
        {
            "history": [],
            "galleries": {},
            "cell_queues": {},
            "notice": "",
            "client_token": None,
            "session_id": uuid4().hex[:8],
            "revision": None,
            "widgets": {"seed": 0, "position": 0},
            "event_location": None,
            "event_token": None,
            "workspace": {"queue": None, "located_field": None},
            "native": {"selection": None, "field_token": None,
                       "gallery_states": {}, "gallery_client_token": None, "selections": {}},
        },
    )


def retire_view(context: dict[str, Any]) -> None:
    """Invalidate pending events while preserving the target's stored state."""
    context["event_location"] = None
    context["event_token"] = None
    context["client_token"] = None
    context["native"]["gallery_client_token"] = None


def retire_active_view() -> None:
    target_id = st.session_state.get("review_target_id")
    if target_id is not None:
        retire_view(get_context(target_id))


def activate_target(target_id: str, *, revision: str) -> dict[str, Any]:
    """Bind annotation ownership separately from model and display revisions."""
    context = get_context(target_id)
    changed_target = st.session_state.get("review_target_id") != target_id
    changed_revision = context["revision"] is not None and context["revision"] != revision
    if changed_revision:
        for gallery in context["galleries"].values():
            for future in gallery["prefetches"].values():
                future.cancel()
        for gallery in context["native"]["gallery_states"].values():
            for futures in gallery["prefetches"].values():
                for future in futures:
                    future.cancel()
        context["galleries"].clear()
        context["cell_queues"].clear()
        for key in ("active_field", "field_crop_source", "focus_cell", "crop_return"):
            context["workspace"].pop(key, None)
        context["workspace"].update(queue=None, located_field=None)
        context.pop("navigation", None)
        context["native"]["gallery_states"].clear()
        context["native"]["field_token"] = None
        context["native"]["selection"] = None
        context["native"]["selections"].clear()
    if changed_target or changed_revision:
        retire_view(context)
    context["revision"] = revision
    st.session_state["review_target_id"] = target_id
    return context


def get_active_context() -> dict[str, Any]:
    return get_context(st.session_state["review_target_id"])


def activate_view(context: dict[str, Any], location: tuple[str | int | float, ...]) -> str:
    """Keep rerender tokens stable and rotate them when a view is activated."""
    if context["event_location"] != location:
        context["event_location"] = location
        context["event_token"] = uuid4().hex
        context["client_token"] = None
        context["native"]["gallery_client_token"] = None
    return context["event_token"]


def widget_key(context: dict[str, Any], name: str, default: object) -> str:
    """Restore persistent control values when Streamlit recreates a widget."""
    value = context["widgets"].setdefault(name, default)
    key = f"review:{context['session_id']}:{name}"
    st.session_state.setdefault(key, value)
    return key


def remember_widget(context: dict[str, Any], name: str) -> None:
    context["widgets"][name] = st.session_state[f"review:{context['session_id']}:{name}"]
    retire_view(context)


def navigate(context: dict[str, Any], scope: str, view: str) -> None:
    context["navigation"] = (scope, view)
    retire_view(context)


def capture_queue(context: dict[str, Any], cells: list[str], source: object, *, position: int = 0) -> None:
    if not cells or len(set(cells)) != len(cells) or not 0 <= position < len(cells):
        raise ValueError("A captured queue requires unique cells and a valid position")
    context["workspace"]["queue"] = {
        "cells": list(cells),
        "position": position,
        "queue_id": uuid4().hex,
        "source": source,
        "origin_field": (
            context["workspace"].get("active_field")
            if context["widgets"].get("view") == "Spatial field" else None
        ),
        "return_to": (
            context["widgets"].get("scope", "Dataset cells"),
            context["widgets"].get("view", "Spatial field"),
        ),
    }
    navigate(context, "Captured cells", "Cell review")


def journal(state_dir: Path, session_id: str, record: dict[str, Any]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = {**record, "time_utc": datetime.now(timezone.utc).isoformat()}
    with (state_dir / f"changes-{session_id}.jsonl").open("a") as stream:
        stream.write(json.dumps(payload) + "\n")


def save_assignment(
    annotations: AnnotationFiles,
    state_dir: Path,
    context: dict[str, Any],
    cell: str,
    value: Label,
    *,
    expected: Label | None,
    record: dict[str, Any],
) -> Edit:
    target = annotations.destination(cell)
    state_dir.mkdir(parents=True, exist_ok=True)
    backup = state_dir / f"{target.parent.name}-{context['session_id']}.csv"
    if target.exists() and not backup.exists():
        backup.write_bytes(target.read_bytes())
    edit = annotations.assign(cell, value, expected=expected)
    context["history"].append(edit)
    journal(
        state_dir,
        context["session_id"],
        {
            **record,
            "cell": cell,
            "previous": edit.previous,
            "value": value,
            "path": str(edit.path),
        },
    )
    return edit


def undo_edit(
    annotations: AnnotationFiles,
    context: dict[str, Any],
    state_dir: Path,
    edit: Edit,
) -> None:
    annotations.undo(edit)
    context["history"].pop()
    journal(
        state_dir,
        context["session_id"],
        {
            "action": "undo",
            "cell": edit.cell,
            "previous": edit.value,
            "value": edit.previous,
            "path": str(edit.path),
        },
    )
    context["notice"] = f"Undid the last edit to {edit.cell}."


