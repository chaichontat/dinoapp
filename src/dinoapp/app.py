"""Universal Streamlit review interface for native and indexed cell sources."""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import os
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st
from streamlit.components.v2 import component

from dinoapp.label_store import Label, LabelConflictError
from dinoapp.sources import (
    AnnotationFiles,
    BarrageSource,
    Crop,
    Edit,
    NativeSource,
    ReviewSource,
    state_label,
    toggle_marker,
)


GALLERY_PAGE_SIZE = 24
GALLERY_IDENTITIES: dict[str, Label] = {
    "BrdU− / EdU−": (0, 0),
    "BrdU− / EdU+": (0, 1),
    "BrdU+ / EdU−": (1, 0),
    "BrdU+ / EdU+": (1, 1),
}


@st.cache_resource
def gallery_picker():
    """Render stable crop cards with one-shot marker edit events."""
    return component(
        "universal_marker_correction_gallery",
        html='<div class="gallery" role="group" aria-label="Marker correction cells"></div>',
        css="""
          .gallery { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 0.6rem; }
          .card { appearance: none; width: 100%; padding: 0; overflow: hidden; text-align: left;
                  color: inherit; background: var(--secondary-background-color, #f5f5f5);
                  border: 1px solid rgba(128,128,128,.35); border-radius: .6rem; cursor: pointer; }
          .card:hover { border-color: rgba(128,128,128,.8); }
          .card:focus-visible { outline: 3px solid #ffcc00; outline-offset: 2px; }
          .card:disabled { cursor: default; opacity: .7; }
          .card img { display: block; width: 100%; height: auto; }
          .body { display: grid; gap: .18rem; padding: .55rem .65rem .65rem; }
          .cell { font-weight: 700; }
          .state, .detail { font-size: .82rem; opacity: .78; }
          .action { margin-top: .2rem; font-size: .88rem; font-weight: 650; }
          .changed { color: #008f3a; font-size: .78rem; font-weight: 700; }
          @media (max-width: 1100px) { .gallery { grid-template-columns: repeat(4, minmax(0, 1fr)); } }
          @media (max-width: 800px) { .gallery { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
          @media (max-width: 520px) { .gallery { grid-template-columns: 1fr; } }
        """,
        js="""export default function({data, parentElement, setTriggerValue}) {
          const gallery = parentElement.querySelector('.gallery');
          const textClasses = ['cell', 'state', 'detail', 'changed', 'action'];
          gallery.setAttribute('aria-label', data.galleryLabel);
          const focused = parentElement.ownerDocument.activeElement?.dataset?.itemId;
          function updateCard(button, item) {
            button.dataset.itemId = item.itemId;
            button.dataset.cell = item.cell;
            button.setAttribute('aria-label', item.ariaLabel);
            button.disabled = item.disabled;
            const image = button.querySelector('img');
            if (item.image && image.getAttribute('src') !== item.image) image.src = item.image;
            for (const className of textClasses) {
              let value = item[className];
              if (className === 'changed') value = item.changed ? 'Changed in this view' : '';
              if (className === 'cell') value = item.cellLabel;
              button.querySelector(`.${className}`).textContent = value;
            }
            button.onclick = () => {
              button.disabled = true;
              setTriggerValue('flipped', {
                itemId: item.itemId, cell: item.cell, expected: item.expected,
                value: item.value, token: data.token
              });
            };
          }
          const existing = Array.from(gallery.querySelectorAll('.card'));
          const canPatch = data.patch && existing.length === data.items.length
            && existing.every((button, index) => button.dataset.itemId === data.items[index].itemId);
          if (data.patch && !canPatch) {
            setTriggerValue('needsFull', {token: data.token});
            return;
          }
          if (!canPatch) gallery.replaceChildren();
          for (const [index, item] of data.items.entries()) {
            if (canPatch) {
              updateCard(existing[index], item);
              continue;
            }
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'card';
            const image = document.createElement('img');
            image.src = item.image;
            image.alt = '';
            const body = document.createElement('span');
            body.className = 'body';
            for (const className of textClasses) {
              const line = document.createElement('span');
              line.className = className;
              body.appendChild(line);
            }
            button.append(image, body);
            updateCard(button, item);
            gallery.appendChild(button);
          }
          if (focused) gallery.querySelector(`[data-item-id="${CSS.escape(focused)}"]`)?.focus();
          return () => { for (const button of gallery.querySelectorAll('button')) button.onclick = null; };
        }""",
    )


@st.cache_resource
def gallery_prefetch_executor() -> ThreadPoolExecutor:
    """Keep next-page crop work off the Streamlit rerun path."""
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="dinoapp-gallery")


@st.cache_resource
def get_barrage_source(root: str) -> BarrageSource:
    return BarrageSource(Path(root))


@st.cache_resource
def get_native_modules(
    native_dir: str, dependency_dirs: tuple[str, ...] = ()
) -> tuple[Any, Any]:
    directories = (native_dir, *dependency_dirs)
    for directory in reversed(directories):
        resolved = str(Path(directory).expanduser().resolve())
        if resolved in sys.path:
            sys.path.remove(resolved)
        sys.path.insert(0, resolved)
    native = importlib.import_module("cell_review")
    native_app = importlib.import_module("cell_review_app")
    return native, native_app


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
        },
    )


def source_key(source_id: str) -> str:
    return hashlib.sha256(source_id.encode()).hexdigest()[:16]


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


def new_gallery_state(
    source: ReviewSource,
    annotations: AnnotationFiles,
    *,
    marker: str,
    subset: str,
    dataset: str,
    identity: Label,
) -> dict[str, Any]:
    rows = source.correction_rows(
        annotations.values,
        identity=identity,
        subset=subset,
        dataset=dataset,
    ).copy()
    if rows["cell"].duplicated().any():
        raise ValueError("Gallery cells must be unique")
    cells = rows["cell"].astype(str).tolist()
    return {
        "queue_id": uuid4().hex,
        "marker": marker,
        "subset": subset,
        "dataset": dataset,
        "identity": identity,
        "cells": cells,
        "initial": {cell: annotations.values[cell] for cell in cells},
        "page": 0,
        "pages": {},
        "prefetches": {},
    }


def _load_gallery_crops(
    source: ReviewSource, cells: tuple[str, ...], marker: str
) -> list[Crop]:
    return [source.crop(cell, marker) for cell in cells]


def prefetch_gallery_page(
    source: ReviewSource, gallery: dict[str, Any], page: int
) -> None:
    """Start loading one unvisited gallery page without blocking this render."""
    page_key = (gallery["queue_id"], page)
    prefetches: dict[tuple[str, int], Future[list[Crop]]] = gallery["prefetches"]
    if page_key in gallery["pages"] or page_key in prefetches:
        return
    start = page * GALLERY_PAGE_SIZE
    cells = tuple(gallery["cells"][start : start + GALLERY_PAGE_SIZE])
    if not cells:
        return
    prefetches[page_key] = gallery_prefetch_executor().submit(
        _load_gallery_crops, source, cells, str(gallery["marker"])
    )


def gallery_page_crops(
    source: ReviewSource, gallery: dict[str, Any], page: int
) -> list[Crop]:
    """Return crops in queue order, reusing completed background work."""
    page_key = (gallery["queue_id"], page)
    future = gallery["prefetches"].pop(page_key, None)
    if future is not None:
        return future.result()
    start = page * GALLERY_PAGE_SIZE
    cells = tuple(gallery["cells"][start : start + GALLERY_PAGE_SIZE])
    return _load_gallery_crops(source, cells, str(gallery["marker"]))


def update_gallery_card(
    card: dict[str, Any],
    value: Label | None,
    initial: Label,
) -> None:
    marker = str(card["marker"])
    marker_name = {"brdu": "BrdU", "edu": "EdU"}[marker]
    if value is None:
        card.update(
            state="No saved call",
            changed=True,
            action="Unavailable",
            ariaLabel=f"{card['cell']}, no saved call",
            disabled=True,
            expected=None,
            value=None,
        )
        return
    target = toggle_marker(value, marker)
    index = {"brdu": 0, "edu": 1}[marker]
    action = f"Set {marker_name}{'+' if target[index] else '−'}"
    card.update(
        state=state_label(value),
        changed=value != initial,
        action=action,
        ariaLabel=f"{card['cell']}, {state_label(value)}. {action}",
        disabled=False,
        expected=list(value),
        value=list(target),
    )


def update_resident_cards(
    context: dict[str, Any], cell: str, value: Label | None
) -> None:
    for gallery in context["galleries"].values():
        initial = gallery["initial"].get(cell)
        if initial is None:
            continue
        for cards in gallery["pages"].values():
            for card in cards:
                if card["cell"] == cell:
                    update_gallery_card(card, value, initial)


def undo_edit(
    annotations: AnnotationFiles,
    context: dict[str, Any],
    state_dir: Path,
    edit: Edit,
) -> None:
    annotations.undo(edit)
    context["history"].pop()
    update_resident_cards(context, edit.cell, edit.previous)
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


def render_gallery(
    source: ReviewSource,
    annotations: AnnotationFiles,
    context: dict[str, Any],
    gallery: dict[str, Any],
    *,
    state_dir: Path,
) -> None:
    marker = str(gallery["marker"])
    widget_source = source_key(source.source_id)
    marker_name = {"brdu": "BrdU", "edu": "EdU"}[marker]
    other_name = {"brdu": "EdU", "edu": "BrdU"}[marker]
    cells = gallery["cells"]
    total = len(cells)
    page_count = max(1, (total + GALLERY_PAGE_SIZE - 1) // GALLERY_PAGE_SIZE)
    page = min(int(gallery["page"]), page_count - 1)
    gallery["page"] = page

    st.title(f"{marker_name} correction gallery")
    st.caption(
        f"Click a crop to flip {marker_name} while keeping {other_name} unchanged. "
        "The loaded queue stays fixed until you refresh its matches."
    )
    undo_column, status_column = st.columns([1, 3], vertical_alignment="center")
    history: list[Edit] = context["history"]
    last_edit = history[-1] if history else None
    undo_label = (
        f"Undo last edit · {last_edit.cell.rsplit('|', 1)[-1]}"
        if last_edit is not None
        else "Undo last edit"
    )
    if undo_column.button(
        undo_label,
        disabled=last_edit is None,
        key=f"undo-{widget_source}",
        width="stretch",
    ):
        try:
            undo_edit(annotations, context, state_dir, last_edit)
        except LabelConflictError as error:
            context["notice"] = str(error)
        st.rerun()
    if context["notice"]:
        status_column.info(context["notice"])
    if not total:
        st.info("No cells match this saved state.")
        return

    start = page * GALLERY_PAGE_SIZE
    stop = min(start + GALLERY_PAGE_SIZE, total)
    page_key = (gallery["queue_id"], page)
    if page_key not in gallery["pages"]:
        cards: list[dict[str, Any]] = []
        crops = gallery_page_crops(source, gallery, page)
        for cell, crop in zip(cells[start:stop], crops, strict=True):
            current = annotations.values.get(cell)
            card: dict[str, Any] = {
                "itemId": cell,
                "cell": cell,
                "cellLabel": crop.title,
                "marker": marker,
                "image": "data:image/png;base64,"
                + base64.b64encode(crop.image).decode("ascii"),
                "detail": crop.detail,
            }
            update_gallery_card(card, current, gallery["initial"][cell])
            cards.append(card)
        gallery["pages"][page_key] = cards
    cards = gallery["pages"][page_key]
    prefetch_gallery_page(source, gallery, page + 1)

    view_token = f"{source.source_id}:{gallery['queue_id']}:{page}"
    patch_existing = context["client_token"] == view_token
    items = cards
    if patch_existing:
        items = [{key: value for key, value in card.items() if key != "image"} for card in cards]
    result = gallery_picker()(
        data={
            "items": items,
            "token": view_token,
            "patch": patch_existing,
            "galleryLabel": f"{marker_name} correction cells",
        },
        key=f"gallery-cards-{widget_source}",
        on_flipped_change=lambda: None,
        on_needsFull_change=lambda: None,
    )
    context["client_token"] = view_token
    needs_full = getattr(result, "needsFull", None)
    if needs_full and needs_full.get("token") == view_token:
        context["client_token"] = None
        st.rerun()

    navigation_label, previous, following = st.columns([3, 1, 1])
    navigation_label.write(f"**Cells {start + 1:,}–{stop:,} of {total:,} loaded**")
    if previous.button("Previous", disabled=page == 0, width="stretch"):
        gallery["page"] = page - 1
        context["client_token"] = None
        st.rerun()
    if following.button("Next", disabled=page == page_count - 1, width="stretch"):
        gallery["page"] = page + 1
        context["client_token"] = None
        st.rerun()

    event = result.flipped
    if not event:
        return
    item_identity = str(event.get("itemId", ""))
    card = next((card for card in cards if card["itemId"] == item_identity), None)
    cell_identity = str(card["cell"]) if card is not None else ""
    expected_raw = event.get("expected")
    value_raw = event.get("value")
    expected = tuple(expected_raw) if isinstance(expected_raw, list) else None
    requested = tuple(value_raw) if isinstance(value_raw, list) else None
    current = annotations.values.get(cell_identity)
    displayed = tuple(card["expected"]) if card and card["expected"] is not None else None
    if (
        event.get("token") != view_token
        or card is None
        or event.get("cell") != cell_identity
        or expected != displayed
        or expected != current
    ):
        update_resident_cards(context, cell_identity, current)
        context["notice"] = (
            "That cell changed after this gallery was rendered; no edit was saved."
        )
        st.rerun()
    target = toggle_marker(current, marker)
    if requested != target:
        context["notice"] = "Ignored an invalid gallery edit; no label was changed."
        st.rerun()
    try:
        edit = save_assignment(
            annotations,
            state_dir,
            context,
            cell_identity,
            target,
            expected=current,
            record={
                "action": f"gallery_toggle_{marker}",
                "source_id": source.source_id,
                "queue_id": gallery["queue_id"],
            },
        )
    except LabelConflictError as error:
        context["notice"] = str(error)
    else:
        update_resident_cards(context, cell_identity, target)
        context["notice"] = f"Saved {cell_identity} as {state_label(target)} in {edit.path}."
    st.rerun()


def review_marker_gallery(
    source: ReviewSource,
    annotations: AnnotationFiles,
    context: dict[str, Any],
    marker: str,
    *,
    state_dir: Path,
) -> None:
    marker_name = {"brdu": "BrdU", "edu": "EdU"}[marker]
    widget_source = source_key(source.source_id)
    with st.sidebar:
        st.header(f"{marker_name} correction gallery")
        subset = st.selectbox("Examples", source.subsets, key=f"subset-{widget_source}")
        dataset = st.selectbox("Dataset", source.datasets, key=f"dataset-{widget_source}")
        identity_name = st.selectbox(
            "Saved state", tuple(GALLERY_IDENTITIES), key=f"identity-{widget_source}"
        )
        st.caption("Matches use current saved labels. The queue stays fixed while you edit.")
    identity = GALLERY_IDENTITIES[identity_name]
    signature = (source.source_id, marker, subset, dataset, identity)
    if st.sidebar.button("Refresh matches", width="stretch"):
        discarded = context["galleries"].pop(signature, None)
        if discarded is not None:
            for future in discarded["prefetches"].values():
                future.cancel()
        context["client_token"] = None
        st.rerun()
    if signature not in context["galleries"]:
        context["galleries"][signature] = new_gallery_state(
            source,
            annotations,
            marker=marker,
            subset=subset,
            dataset=dataset,
            identity=identity,
        )
    render_gallery(
        source,
        annotations,
        context,
        context["galleries"][signature],
        state_dir=state_dir,
    )


def review_barrage_cell(
    source: ReviewSource,
    annotations: AnnotationFiles,
    context: dict[str, Any],
    *,
    state_dir: Path,
) -> None:
    filter_values: dict[str, Label | None | str] = {
        "Unlabeled": None,
        "All cells": "all",
        **GALLERY_IDENTITIES,
    }
    widget_source = source_key(source.source_id)
    with st.sidebar:
        st.header("Cell review")
        dataset = st.selectbox(
            "Dataset", source.datasets, key=f"cell-dataset-{widget_source}"
        )
        filter_name = st.selectbox(
            "Saved state", tuple(filter_values), key=f"cell-filter-{widget_source}"
        )
        st.caption("Choose one explicit BrdU/EdU identity for each displayed cell.")
    signature = (source.source_id, dataset, filter_name)
    queues = context["cell_queues"]
    if st.sidebar.button("Refresh cells", width="stretch"):
        queues.pop(signature, None)
        st.rerun()
    if signature not in queues:
        rows = source.review_rows(dataset=dataset)
        wanted = filter_values[filter_name]
        if wanted is None:
            rows = rows[~rows["cell"].isin(annotations.values)]
        elif wanted != "all":
            rows = rows[
                rows["cell"].map(
                    lambda cell: annotations.values.get(str(cell)) == wanted
                )
            ]
        queues[signature] = {
            "cells": rows["cell"].astype(str).tolist(),
            "position": 0,
        }
    queue = queues[signature]
    cells: list[str] = queue["cells"]
    st.title("Cell review")
    st.caption("Assign the displayed cell directly from its real combined crop.")
    if not cells:
        st.info("No cells match this filter.")
        return
    position = min(int(queue["position"]), len(cells) - 1)
    queue["position"] = position
    cell = cells[position]
    crop = source.crop(cell, "combined")
    current = annotations.values.get(cell)

    previous, count, following, undo_column = st.columns([1, 2, 1, 1])
    if previous.button("Previous", disabled=position == 0, width="stretch"):
        queue["position"] = position - 1
        st.rerun()
    count.write(f"**Cell {position + 1:,} of {len(cells):,}**")
    if following.button(
        "Next", disabled=position == len(cells) - 1, width="stretch"
    ):
        queue["position"] = position + 1
        st.rerun()
    history: list[Edit] = context["history"]
    if undo_column.button(
        "Undo last edit",
        disabled=not history,
        key=f"cell-undo-{widget_source}",
        width="stretch",
    ):
        edit = history[-1]
        try:
            undo_edit(annotations, context, state_dir, edit)
        except LabelConflictError as error:
            context["notice"] = str(error)
        st.rerun()

    image_column, action_column = st.columns([3, 1], gap="large")
    with image_column:
        st.subheader(crop.title)
        st.image(crop.image, width="stretch")
        st.caption(f"{cell} · {crop.detail}")
    with action_column:
        st.subheader("Saved identity")
        st.write(state_label(current) if current is not None else "No saved call")
        for name, value in GALLERY_IDENTITIES.items():
            if st.button(
                name,
                key=f"assign-{widget_source}-{cell}-{value}",
                type="primary" if current == value else "secondary",
                width="stretch",
            ):
                try:
                    edit = save_assignment(
                        annotations,
                        state_dir,
                        context,
                        cell,
                        value,
                        expected=current,
                        record={
                            "action": "assign",
                            "source_id": source.source_id,
                            "dataset_filter": dataset,
                        },
                    )
                except LabelConflictError as error:
                    context["notice"] = str(error)
                else:
                    update_resident_cards(context, cell, value)
                    context["notice"] = (
                        f"Saved {cell} as {state_label(value)} in {edit.path}."
                    )
                    queue["position"] = min(position + 1, len(cells) - 1)
                st.rerun()
        if context["notice"]:
            st.info(context["notice"])


def load_native_source(
    native_dir: str,
    label_root: Path,
    dependency_dirs: tuple[str, ...] = (),
) -> tuple[NativeSource, Any, Any]:
    native, native_app = get_native_modules(native_dir, dependency_dirs)
    field_source = native_app.get_source(native.FieldSource)
    return NativeSource(field_source, label_root=label_root), native, native_app


def native_field_mode(source: NativeSource, native: Any, native_app: Any) -> None:
    """Reuse the prod2 field renderer while the shared app owns navigation."""
    for key, value in (
        ("position", 0),
        ("seed", 0),
        ("selection", None),
        ("history", []),
        ("session_id", uuid4().hex[:8]),
        ("notice", ""),
    ):
        st.session_state.setdefault(key, value)
    with st.sidebar:
        st.header("Field")
        seed = int(
            st.number_input(
                "Seed",
                min_value=0,
                max_value=2**31 - 1,
                step=1,
                key="seed",
                on_change=native_app.reset_position,
            )
        )
        st.number_input("Position", min_value=0, step=1, key="position")
        previous, following = st.columns(2)
        previous.button(
            "Previous",
            on_click=native_app.move_position,
            args=(-1,),
            disabled=st.session_state.position == 0,
            width="stretch",
        )
        following.button(
            "Next", on_click=native_app.move_position, args=(1,), width="stretch"
        )
        width = float(
            st.select_slider("Field width (µm)", options=[100, 150, 200, 250, 300], value=200)
        )
        z_um = float(
            st.number_input(
                "Z (µm)",
                value=float(source.field_source.default_z),
                step=float(source.field_source.scale[0]),
                format="%.1f",
            )
        )
        border_alpha = st.slider("Cell border opacity", 0.0, 1.0, 0.35, 0.05)
        boundary_alpha = st.slider("Boundary opacity", 0.0, 1.0, 0.45, 0.05)
        right_view = st.selectbox("Right panel view", ("Cell identities", "Model uncertainty"))
    field = native_app.get_field(
        native.FieldSource, seed, st.session_state.position, z_um, width
    )
    if st.session_state.get("field_token") != field.token:
        st.session_state.field_token = field.token
        st.session_state.selection = None
    annotations = native.AnnotationFiles(
        source.annotation_paths, source.default_annotation_path
    )
    calls = source.field_source.field_calls(field, annotations)
    image = native.render_field(
        field,
        calls,
        mode=right_view,
        selected=None,
        border_alpha=border_alpha,
        boundary_alpha=boundary_alpha,
        contrast=source.field_source.contrast,
        uncertainty=source.field_source.field_uncertainty(field),
        uncertainty_log_vmax=source.field_source.uncertainty_log_vmax,
    )
    native_app.review_field(
        source.field_source,
        field,
        source.annotation_paths,
        source.label_root / "cell-review",
        image,
        right_view,
    )


def launch() -> None:
    """Start the universal reviewer through Streamlit."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ],
        check=True,
    )


def main() -> None:
    st.set_page_config(page_title="Cell review", layout="wide")
    source_options = ("Barrage", "Prod2 native")
    configured_source = os.environ.get("DINO_REVIEW_SOURCE", "Barrage")
    source_index = (
        source_options.index(configured_source) if configured_source in source_options else 0
    )
    with st.sidebar:
        source_kind = st.selectbox(
            "Review source", source_options, index=source_index
        )
    native = native_app = None
    if source_kind == "Barrage":
        default_root = os.environ.get(
            "DINO_REVIEW_BARRAGE_DIR",
            os.environ.get("DINO_BARRAGE_DIR", "output/brdu_edu_barrage"),
        )
        root = Path(st.sidebar.text_input("Barrage directory", default_root)).resolve()
        try:
            source: ReviewSource = get_barrage_source(str(root))
        except (FileNotFoundError, ValueError) as error:
            st.error(str(error))
            st.stop()
        modes = (
            "EdU correction gallery",
            "BrdU correction gallery",
            "Cell review",
        )
        state_dir = root / "cell-review"
    else:
        native_dir = st.sidebar.text_input(
            "Native app directory",
            os.environ.get(
                "DINO_NATIVE_REVIEW_DIR",
                "/working/eduseg/20260808-prod2/full-run/vz-rates",
            ),
        )
        label_root = Path(
            st.sidebar.text_input(
                "Native label root",
                os.environ.get(
                    "DINO_REVIEW_LABEL_ROOT",
                    "/working/eduseg/20260808-prod2/full-run2",
                ),
            )
        ).resolve()
        default_native_pythonpath = str(Path(native_dir).resolve().parent / "svz-cut")
        native_dependency_dirs = tuple(
            path
            for path in os.environ.get(
                "DINO_NATIVE_PYTHONPATH", default_native_pythonpath
            ).split(os.pathsep)
            if path
        )
        try:
            source, native, native_app = load_native_source(
                native_dir, label_root, native_dependency_dirs
            )
        except (FileNotFoundError, ImportError, ValueError) as error:
            st.error(str(error))
            st.stop()
        modes = (
            "Ventricular fields",
            "Training miscalls",
            "EdU correction gallery",
            "BrdU correction gallery",
        )
        state_dir = label_root / "cell-review"

    with st.sidebar:
        review_mode = st.selectbox("Review mode", modes)
        st.caption(f"Source: {source.display_name}")
    if review_mode == "Ventricular fields":
        native_field_mode(source, native, native_app)
        return
    if review_mode == "Training miscalls":
        native_app.review_training_miscalls(
            source.field_source,
            source.annotation_paths,
            state_dir,
        )
        return
    annotations = AnnotationFiles(
        source.annotation_paths, source.default_annotation_path
    )
    context = get_context(source.source_id)
    if review_mode == "Cell review":
        review_barrage_cell(
            source,
            annotations,
            context,
            state_dir=state_dir,
        )
        return
    marker = "edu" if review_mode.startswith("EdU") else "brdu"
    review_marker_gallery(
        source,
        annotations,
        context,
        marker,
        state_dir=state_dir,
    )


if __name__ == "__main__":
    main()
