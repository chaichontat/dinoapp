"""Universal Streamlit review interface for native and indexed cell sources."""

from __future__ import annotations

import base64
import importlib
import os
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any
from uuid import uuid4

import dino_markers
import streamlit as st
from dino_markers import (
    ContractMismatchError,
    Label,
    MarkerConfig,
    SavedFilter,
    matches_saved,
    state_label,
    toggle_marker,
)
from streamlit.components.v2 import component

from dinoapp.label_store import LabelConflictError
from dinoapp.review_session import (
    EDIT_REJECTED,
    activate_target,
    activate_view,
    capture_queue,
    cell_draft,
    draft_label,
    navigate,
    get_context,
    remember_widget,
    retire_active_view,
    save_assignment,
    saved_filter_controls,
    undo_edit as persist_undo,
    widget_key,
)
from dinoapp.sources import (
    AnnotationFiles,
    BarrageSource,
    Crop,
    Edit,
    NativeSource,
    ReviewSource,
)


GALLERY_PAGE_SIZE = 24
CELL_OPTIONS = ("Unsaved", "Saved", "All cells")


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
def get_barrage_source(root: str, marker_revision: str) -> BarrageSource:
    if (Path(root) / "display.json").is_file():
        from dinoapp.corrected_barrage import CorrectedBarrageSource

        return CorrectedBarrageSource(Path(root))
    return BarrageSource(Path(root))


NATIVE_SOURCES = {
    "Prod2": "dinoapp.native",
    "Prod3": "dinoapp.prod3",
    "Prod2-2": "dinoapp.prod2_2",
    "Prod3-2": "dinoapp.prod3_2",
}


@st.cache_resource
def get_native_modules(dataset: str) -> tuple[Any, Any]:
    if dataset not in NATIVE_SOURCES:
        raise ValueError(f"Unknown native dataset: {dataset}")
    return importlib.import_module(NATIVE_SOURCES[dataset]), importlib.import_module("dinoapp.native_app")


def new_gallery_state(
    source: ReviewSource,
    annotations: AnnotationFiles,
    *,
    marker: str,
    subset: str,
    dataset: str,
    identity: SavedFilter,
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
    *,
    config: MarkerConfig,
) -> None:
    marker = str(card["marker"])
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
    contract = config.contract
    target = toggle_marker(contract, value, marker)
    action = f"Set {config.label(marker)}{'+' if target[contract.markers.index(marker)] else '−'}"
    card.update(
        state=state_label(config, value),
        changed=value != initial,
        action=action,
        ariaLabel=f"{card['cell']}, {state_label(config, value)}. {action}",
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
                    update_gallery_card(card, value, initial, config=gallery["config"])


def undo_edit(
    annotations: AnnotationFiles,
    context: dict[str, Any],
    state_dir: Path,
    edit: Edit,
) -> None:
    persist_undo(annotations, context, state_dir, edit)
    context["drafts"].pop(edit.cell, None)
    update_resident_cards(context, edit.cell, edit.previous)


def render_gallery(
    source: ReviewSource,
    annotations: AnnotationFiles,
    context: dict[str, Any],
    gallery: dict[str, Any],
    *,
    state_dir: Path,
) -> None:
    marker = str(gallery["marker"])
    config = source.config
    gallery["config"] = config
    marker_name = config.label(marker)
    others = ", ".join(config.label(key) for key in config.display_markers if key != marker)
    cells = gallery["cells"]
    total = len(cells)
    page_count = max(1, (total + GALLERY_PAGE_SIZE - 1) // GALLERY_PAGE_SIZE)
    page = min(int(gallery["page"]), page_count - 1)
    gallery["page"] = page

    view_token = activate_view(context, ("gallery", source.source_id, gallery["queue_id"], page))
    st.title(f"{marker_name} correction gallery")
    st.caption(
        f"Click a crop to flip {marker_name}"
        + (f" while keeping {others} unchanged. " if others else ". ")
        + "The loaded queue stays fixed until you refresh its matches."
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
        key=f"undo-{context['session_id']}:{view_token}",
        width="stretch",
    ):
        try:
            undo_edit(annotations, context, state_dir, last_edit)
        except EDIT_REJECTED as error:
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
            card: dict[str, Any] = {
                "itemId": cell,
                "cell": cell,
                "cellLabel": crop.title,
                "marker": marker,
                "image": "data:image/png;base64,"
                + base64.b64encode(crop.image).decode("ascii"),
                "detail": crop.detail,
            }
            cards.append(card)
        gallery["pages"][page_key] = cards
    cards = gallery["pages"][page_key]
    for card in cards:
        update_gallery_card(
            card, annotations.values.get(card["cell"]), gallery["initial"][card["cell"]], config=config
        )
    prefetch_gallery_page(source, gallery, page + 1)

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
        key=f"gallery-cards-{context['session_id']}",
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
    if previous.button("Previous", disabled=page == 0, width="stretch", key=f"gallery-prev:{view_token}"):
        gallery["page"] = page - 1
        context["client_token"] = None
        st.rerun()
    if following.button("Next", disabled=page == page_count - 1, width="stretch", key=f"gallery-next:{view_token}"):
        gallery["page"] = page + 1
        context["client_token"] = None
        st.rerun()

    if "combined" in source.crop_markers:
        cells_to_inspect = [card["cell"] for card in cards]
        chosen = st.selectbox("Cell to inspect", cells_to_inspect, key=f"gallery-open-choice:{view_token}")
        if st.button("Review cell", key=f"gallery-open-cell:{view_token}"):
            capture_queue(context, cells_to_inspect, source, position=cells_to_inspect.index(chosen))
            st.rerun(scope="app")

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
    target = toggle_marker(config.contract, current, marker)
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
    except ContractMismatchError as error:
        context["notice"] = str(error)
        st.error(str(error))
        return
    else:
        update_resident_cards(context, cell_identity, target)
        context["notice"] = f"Saved {cell_identity} as {state_label(config, target)} in {edit.path}."
    st.rerun()


def review_marker_gallery(
    source: ReviewSource,
    annotations: AnnotationFiles,
    context: dict[str, Any],
    marker: str,
    *,
    state_dir: Path,
) -> None:
    config = source.config
    with st.sidebar:
        st.header(f"{config.label(marker)} correction gallery")
        subset = st.selectbox("Examples", source.subsets, key=widget_key(context, "gallery_subset", source.subsets[0]), on_change=remember_widget, args=(context, "gallery_subset"))
        dataset = st.selectbox("Dataset", source.datasets, key=widget_key(context, "gallery_dataset", source.datasets[0]), on_change=remember_widget, args=(context, "gallery_dataset"))
        identity = saved_filter_controls(context, config, "gallery_saved")
        st.caption("Matches use current saved labels. The queue stays fixed while you edit.")
    signature = (source.source_id, config.contract, marker, subset, dataset, identity)
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
    queue: dict[str, Any] | None = None,
) -> None:
    config = source.config
    contract = config.contract
    dataset = "All datasets"
    if queue is None:
        with st.sidebar:
            st.header("Cell review")
            dataset = st.selectbox(
                "Dataset", source.datasets, key=widget_key(context, "cell_dataset", source.datasets[0]), on_change=remember_widget, args=(context, "cell_dataset")
            )
            cells_option = st.selectbox(
                "Cells", CELL_OPTIONS, key=widget_key(context, "cell_filter", CELL_OPTIONS[0]), on_change=remember_widget, args=(context, "cell_filter")
            )
            saved_filter: SavedFilter = (None,) * len(contract.markers)
            if cells_option == "Saved":
                saved_filter = saved_filter_controls(context, config, "cell_saved")
            st.caption(
                f"Set {', '.join(config.label(key) for key in config.display_markers)} "
                "for each displayed cell, then save."
            )
        signature = (source.source_id, contract, dataset, cells_option, saved_filter)
        queues = context["cell_queues"]
        if st.sidebar.button("Refresh cells", width="stretch"):
            queues.pop(signature, None)
            st.rerun()
        if signature not in queues:
            rows = source.review_rows(dataset=dataset)
            if cells_option == "Unsaved":
                rows = rows[~rows["cell"].isin(annotations.values)]
            elif cells_option == "Saved":
                rows = rows[
                    rows["cell"].map(
                        lambda cell: str(cell) in annotations.values
                        and matches_saved(annotations.values[str(cell)], saved_filter)
                    )
                ]
            queues[signature] = {
                "cells": rows["cell"].astype(str).tolist(),
                "position": 0,
                "queue_id": uuid4().hex,
            }
        queue = queues[signature]
    cells: list[str] = queue["cells"]
    position = min(int(queue["position"]), max(len(cells) - 1, 0))
    view_token = activate_view(context, ("cell", source.source_id, queue["queue_id"], position))
    st.title("Cell review")
    st.caption("Assign the displayed cell directly from its real combined crop.")
    if not cells:
        st.info("No cells match this filter.")
        return
    queue["position"] = position
    cell = cells[position]
    crop = source.crop(cell, "combined")
    current = annotations.values.get(cell)

    return_to = queue.get("return_to")
    if return_to is not None and st.button("Return to previous view", key=f"cell-return:{view_token}"):
        if queue.get("origin_field") is not None:
            context["workspace"]["located_field"] = queue["origin_field"]
        navigate(context, *return_to)
        st.rerun()
    if isinstance(source, NativeSource) or context["workspace"].get("spatial_source") is not None:
        if st.button("Show in field", key=f"cell-locate:{view_token}"):
            width = float(context["widgets"].get("field_width", 200))
            if isinstance(source, NativeSource):
                reader = source.field_source
                located = source.field_at_cell(cell, width)
            else:
                reader, label = context["workspace"]["spatial_source"].resolve_cell(cell)
                located = reader.field_at_cell(label, width)
            context["workspace"]["located_field"] = (reader, located)
            context["workspace"]["focus_cell"] = cell
            context["workspace"]["crop_return"] = (context["widgets"]["scope"], "Cell review")
            navigate(context, "Dataset cells", "Spatial field")
            st.rerun()

    previous, count, following, undo_column = st.columns([1, 2, 1, 1])
    if previous.button("Previous", disabled=position == 0, width="stretch", key=f"cell-prev:{view_token}"):
        queue["position"] = position - 1
        st.rerun()
    count.write(f"**Cell {position + 1:,} of {len(cells):,}**")
    if following.button(
        "Next", disabled=position == len(cells) - 1, width="stretch", key=f"cell-next:{view_token}"
    ):
        queue["position"] = position + 1
        st.rerun()
    if undo_column.button(
        "Undo last edit",
        disabled=not context["history"],
        key=f"cell-undo-{context['session_id']}:{view_token}",
        width="stretch",
    ):
        edit = context["history"][-1]
        try:
            undo_edit(annotations, context, state_dir, edit)
        except EDIT_REJECTED as error:
            context["notice"] = str(error)
        st.rerun()

    draft = cell_draft(context, cell, current, contract.markers)
    image_column, action_column = st.columns([3, 1], gap="large")
    with image_column:
        st.subheader(crop.title)
        st.image(crop.image, width="stretch")
        st.caption(f"{cell} · {crop.detail}")
    with action_column:
        st.subheader("Saved identity")
        st.write(state_label(config, current) if current is not None else "No saved call")
        for key in config.display_markers:
            draft["values"][key] = st.segmented_control(
                config.label(key),
                (0, 1),
                format_func=lambda value: "+" if value else "−",
                default=draft["values"][key],
                key=f"cell-mark:{context['session_id']}:{cell}:{key}:{view_token}:{draft['expected']}",
            )
        value = draft_label(contract.markers, draft)
        if st.button(
            "Save and next",
            disabled=value is None,
            type="primary",
            key=f"cell-save:{context['session_id']}:{cell}:{view_token}",
            width="stretch",
        ):
            advance = value == current
            if advance:
                context["notice"] = f"{cell} is unchanged; nothing was written."
            else:
                try:
                    edit = save_assignment(
                        annotations,
                        state_dir,
                        context,
                        cell,
                        value,
                        expected=draft["expected"],
                        record={
                            "action": "assign",
                            "source_id": source.source_id,
                            "dataset_filter": dataset,
                        },
                    )
                except LabelConflictError as error:
                    context["notice"] = str(error)
                    st.rerun()
                except ContractMismatchError as error:
                    context["notice"] = str(error)
                else:
                    advance = True
                    update_resident_cards(context, cell, value)
                    context["notice"] = f"Saved {cell} as {state_label(config, value)} in {edit.path}."
            if advance:
                del context["drafts"][cell]
                queue["position"] = min(position + 1, len(cells) - 1)
                st.rerun()
        if context["notice"]:
            st.info(context["notice"])


def native_field_mode(source: NativeSource, native: Any, native_app: Any, *, marker_revision: str) -> None:
    """Use the packaged spatial renderer with the selected dataset."""
    context = get_context(source.target_id)
    located_view = context["workspace"]["located_field"] is not None
    with st.sidebar:
        if located_view:
            st.caption("Viewing a located cell. Return to sampled fields to change the sampling cursor.")
        st.header("Field")
        seed = int(
            st.number_input(
                "Seed",
                min_value=0,
                max_value=2**31 - 1,
                disabled=located_view,
                step=1,
                key=widget_key(context, "seed", 0),
                on_change=native_app.change_field_widget, args=(context, "seed"),
            )
        )
        position = int(st.number_input("Position", min_value=0, step=1, disabled=located_view, key=widget_key(context, "position", 0), on_change=remember_widget, args=(context, "position")))
        previous, following = st.columns(2)
        previous.button(
            "Previous",
            on_click=native_app.move_position,
            args=(-1, context), key=f"field-prev:{context['session_id']}",
            disabled=located_view or position == 0,
            width="stretch",
        )
        following.button(
            "Next", disabled=located_view, on_click=native_app.move_position, args=(1, context), width="stretch", key=f"field-next:{context['session_id']}"
        )
        width = float(
            st.select_slider("Field width (µm)", disabled=located_view, options=[100, 150, 200, 250, 300], key=widget_key(context, "field_width", 200), on_change=remember_widget, args=(context, "field_width"))
        )
        z_um = float(
            st.number_input(
                "Z (µm)", disabled=located_view,
                key=widget_key(context, "field_z", float(source.field_source.default_z)), on_change=remember_widget, args=(context, "field_z"),
                step=float(source.field_source.scale[0]),
                format="%.1f",
            )
        )
        border_alpha = st.slider("Cell border opacity", 0.0, 1.0, step=0.05, key=widget_key(context, "field_border_alpha", 0.35), on_change=remember_widget, args=(context, "field_border_alpha"))
        boundary_alpha = st.slider("Boundary opacity", 0.0, 1.0, step=0.05, key=widget_key(context, "field_boundary_alpha", 0.45), on_change=remember_widget, args=(context, "field_boundary_alpha"))
        right_view = st.selectbox("Right panel view", ("Cell identities", "Model uncertainty"), key=widget_key(context, "field_view", "Cell identities"), on_change=remember_widget, args=(context, "field_view"))
    located = context["workspace"]["located_field"]
    if located is None:
        field = native_app.get_field(native.FieldSource, marker_revision, seed, position, z_um, width)
    else:
        _, field = located
        if st.button("Return to sampled fields"):
            context["workspace"]["located_field"] = None
            navigate(context, "Dataset cells", "Spatial field")
            st.rerun()
        if context["workspace"].get("crop_return") and st.button("Return to cell review"):
            navigate(context, *context["workspace"]["crop_return"])
            st.rerun()
    context["workspace"]["field_crop_source"] = source
    context["workspace"]["active_field"] = (source.field_source, field)
    annotations = AnnotationFiles(
        source.annotation_paths, source.default_annotation_path, contract=source.config.contract
    )
    calls = source.field_source.field_calls(field, annotations)
    image = native.render_field(
        field,
        calls,
        config=source.config,
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
    configured_dataset = os.environ.get("DINO_NATIVE_DATASET", "Prod2")
    target_options = ("Crop directory", *NATIVE_SOURCES)
    configured_target = configured_dataset if os.environ.get("DINO_REVIEW_SOURCE") == "Native fields" else "Crop directory"
    if configured_target not in target_options:
        st.error(f"Unknown review target: {configured_target}")
        st.stop()
    with st.sidebar:
        target = st.selectbox("Review target", target_options, index=target_options.index(configured_target), on_change=retire_active_view)
    native = native_app = None
    if target == "Crop directory":
        default_root = os.environ.get("DINO_REVIEW_BARRAGE_DIR", os.environ.get("DINO_BARRAGE_DIR", "output/brdu_edu_barrage"))
        root = Path(st.sidebar.text_input("Barrage directory", default_root, on_change=retire_active_view)).expanduser().resolve()
        revision_of_markers = dino_markers.marker_revision((root,))
        try:
            source: ReviewSource = get_barrage_source(str(root), revision_of_markers)
        except (FileNotFoundError, ValueError) as error:
            retire_active_view()
            st.error(str(error))
            st.stop()
        state_dir = root / "cell-review"
        spatial_path = root / "spatial.json"
    else:
        try:
            native, native_app = get_native_modules(target)
            label_root = Path(st.sidebar.text_input("Review state directory", os.environ.get("DINO_REVIEW_LABEL_ROOT", str(native.REVIEW_ROOT)), key=f"native-label-root:{target}", on_change=retire_active_view)).expanduser().resolve()
            revision_of_markers = dino_markers.marker_revision(native.FieldSource.annotation_directories())
            source = NativeSource(
                native_app.get_source(native.FieldSource, revision_of_markers), label_root=label_root
            )
        except (FileNotFoundError, ImportError, ValueError) as error:
            retire_active_view()
            st.error(str(error))
            st.stop()
        state_dir = label_root / "cell-review"
    config = source.config
    context = activate_target(
        source.target_id,
        revision=f"{source.source_id}:markers:{revision_of_markers}",
        contract=config.contract,
    )
    context["workspace"]["review_source"] = source
    spatial = isinstance(source, NativeSource)
    if target == "Crop directory" and spatial_path.is_file():
        from dinoapp.rhu2_app import get_source as get_spatial_source
        context["workspace"]["spatial_source"] = get_spatial_source(str(root), revision_of_markers)
        spatial = True
    gallery_views = {
        f"{config.label(key)} correction gallery": key for key in source.crop_markers if key != "combined"
    }
    scopes = ["Dataset cells"]
    if isinstance(source, NativeSource):
        scopes.append("Training miscalls")
    if context["workspace"]["queue"] is not None:
        scopes.append("Captured cells")
    pending = context.pop("navigation", None)
    if pending:
        for name, value in zip(("scope", "view"), pending, strict=True):
            context["widgets"][name] = value
            st.session_state[f"review:{context['session_id']}:{name}"] = value
    if context["widgets"].get("scope") not in scopes:
        context["widgets"]["scope"] = scopes[0]
        st.session_state[f"review:{context['session_id']}:scope"] = scopes[0]
    with st.sidebar:
        scope = st.selectbox("Review scope", scopes, key=widget_key(context, "scope", scopes[0]), on_change=remember_widget, args=(context, "scope"))
        if scope == "Training miscalls":
            views = ["Correction gallery"]
        elif scope == "Captured cells":
            views = ["Cell review"]
        else:
            views = []
            if spatial:
                views.append("Spatial field")
            if "combined" in source.crop_markers:
                views.append("Cell review")
            views.extend(gallery_views)
        if pending is None and context["workspace"].get("active_scope") != scope:
            restored = context["workspace"].setdefault("scope_views", {}).get(scope, views[0] if views else None)
            context["widgets"]["view"] = restored
            st.session_state[f"review:{context['session_id']}:view"] = restored
        if not views:
            st.error("This target has no supported review crops or spatial fields.")
            st.stop()
        if context["widgets"].get("view") not in views:
            context["widgets"]["view"] = views[0]
            st.session_state[f"review:{context['session_id']}:view"] = views[0]
        view = st.selectbox("Review view", views, key=widget_key(context, "view", views[0]), on_change=remember_widget, args=(context, "view"))
        context["workspace"]["active_scope"] = scope
        context["workspace"].setdefault("scope_views", {})[scope] = view
        st.caption(f"Target: {source.display_name}")
        if isinstance(source, NativeSource):
            st.caption(source.field_source.classifier_caption)
    if view == "Spatial field":
        if isinstance(source, NativeSource):
            native_field_mode(source, native, native_app, marker_revision=revision_of_markers)
        else:
            from dinoapp.rhu2_app import spatial_mode
            spatial_mode(root, target_id=source.target_id)
        return
    if scope == "Training miscalls":
        native_app.review_training_miscalls(source.field_source, source.annotation_paths, state_dir)
        return
    annotations = AnnotationFiles(source.annotation_paths, source.default_annotation_path, contract=config.contract)
    if view == "Cell review":
        queue = context["workspace"]["queue"] if scope == "Captured cells" else None
        review_barrage_cell(
            queue["source"] if queue is not None else source,
            annotations, context, state_dir=state_dir, queue=queue,
        )
        return
    review_marker_gallery(source, annotations, context, gallery_views[view], state_dir=state_dir)


if __name__ == "__main__":
    main()
