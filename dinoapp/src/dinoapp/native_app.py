"""Streamlit cell correction beside the ventricular boundary."""

from __future__ import annotations

import base64
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import numpy as np
import streamlit as st
from streamlit.components.v2 import component

from dinoapp.native import (
    AnnotationFiles,
    FieldSource,
    IDENTITIES,
    pick_cell,
    prefetch_brdu_crop,
    prefetch_edu_crop,
    toggle_brdu,
    toggle_edu,
    training_gallery,
    training_miscalls,
)
from dinoapp.label_store import LabelConflictError
from dinoapp.review_session import (
    activate_view,
    get_active_context,
    remember_widget,
    retire_view,
    widget_key,
    save_assignment as persist_assignment,
    undo_edit as persist_undo,
)


IDENTITY_BACKGROUNDS = (
    "#404040",
    "#008f3a",
    "#ff00ff",
    "#ffffff",
    "#777777",
)
IDENTITY_FOREGROUNDS = ("#ffffff", "#ffffff", "#ffffff", "#202020", "#ffffff")


@st.cache_resource(hash_funcs={type: id})
def get_source(source_type: type[FieldSource]) -> FieldSource:
    """Invalidate native readers when a code reload replaces their source class."""
    return source_type()


@st.cache_data(max_entries=6, show_spinner=False, hash_funcs={type: id})
def get_field(
    source_type: type[FieldSource],
    seed: int,
    position: int,
    z_um: float,
    width_um: float,
):
    return get_source(source_type).field(seed, position, z_um, width_um)


@st.cache_resource
def image_picker():
    # The installed Streamlit component API supplies one-shot click events.
    # Only image data is dynamic; markup and JavaScript are fixed trusted code.
    return component(
        "native_cell_picker",
        html='<div class="field"><img alt="Select a segmented cell" draggable="false" /><svg viewBox="0 0 1 1" preserveAspectRatio="none"></svg></div>',
        css=".field { position: relative; } img { width: 100%; height: auto; display: block; cursor: crosshair; } svg { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }",
        js="""export default function({data, parentElement, setTriggerValue}) {
          const image = parentElement.querySelector('img');
          if (image.getAttribute('src') !== data.image) image.src = data.image;
          const svg = parentElement.querySelector('svg');
          const ns = 'http://www.w3.org/2000/svg';
          svg.replaceChildren();
          for (const mark of data.marks) {
            const group = document.createElementNS(ns, 'g');
            group.setAttribute('data-cell', String(mark.label));
            group.setAttribute('data-color', mark.color);
            for (const [color, width] of [['#000000', 4], [mark.color, 2]]) {
              for (const sign of [-1, 1]) {
                const line = document.createElementNS(ns, 'line');
                line.setAttribute('x1', mark.x - 0.009);
                line.setAttribute('x2', mark.x + 0.009);
                line.setAttribute('y1', mark.y - sign * 0.009);
                line.setAttribute('y2', mark.y + sign * 0.009);
                line.setAttribute('stroke', color);
                line.setAttribute('stroke-width', width);
                line.setAttribute('vector-effect', 'non-scaling-stroke');
                group.appendChild(line);
              }
            }
            svg.appendChild(group);
          }
          if (data.selection) {
            const circle = document.createElementNS(ns, 'circle');
            circle.setAttribute('cx', data.selection.x);
            circle.setAttribute('cy', data.selection.y);
            circle.setAttribute('r', '0.012');
            circle.setAttribute('fill', 'none');
            circle.setAttribute('stroke', '#ffcc00');
            circle.setAttribute('stroke-width', '1');
            circle.setAttribute('stroke-opacity', '0.7');
            circle.setAttribute('vector-effect', 'non-scaling-stroke');
            svg.appendChild(circle);
          }
          image.alt = data.label;
          image.onclick = (event) => {
            const rect = image.getBoundingClientRect();
            setTriggerValue('clicked', {
              x: (event.clientX - rect.left) / rect.width,
              y: (event.clientY - rect.top) / rect.height,
              token: data.token
            });
          };
          return () => { image.onclick = null; };
        }""",
    )


@st.cache_resource
def gallery_picker():
    """Render stable, keyboard-accessible crop cards with one-shot edit events."""
    return component(
        "marker_correction_gallery",
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
          gallery.setAttribute('aria-label', data.galleryLabel);
          const focused = parentElement.ownerDocument.activeElement?.dataset?.itemId;
          function updateCard(button, item) {
            button.dataset.itemId = item.itemId;
            button.dataset.cell = item.cell;
            button.setAttribute('aria-label', item.ariaLabel);
            button.disabled = item.disabled;
            const image = button.querySelector('img');
            if (item.image && image.getAttribute('src') !== item.image) image.src = item.image;
            for (const className of ['cell', 'state', 'detail', 'changed', 'action']) {
              const value = className === 'changed'
                ? (item.changed ? 'Changed in this view' : '')
                : item[className];
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
          if (!canPatch) {
            gallery.replaceChildren();
          }
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
            for (const className of ['cell', 'state', 'detail', 'changed', 'action']) {
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
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="edu-gallery")


def reset_position(context: dict) -> None:
    context["widgets"]["position"] = 0
    retire_view(context)
    st.session_state[widget_key(context, "position", 0)] = 0


def move_position(offset: int, context: dict) -> None:
    position = max(0, int(context["widgets"]["position"]) + offset)
    context["widgets"]["position"] = position
    retire_view(context)
    st.session_state[widget_key(context, "position", 0)] = position


def change_field_widget(context: dict, name: str) -> None:
    remember_widget(context, name)
    if name in {"seed", "rhu2_branch_index"}:
        reset_position(context)


def save_assignment(
    annotations: AnnotationFiles,
    state_dir: Path,
    cell: str,
    value: tuple[int, int],
    *,
    expected: tuple[int, int] | None,
    record: dict,
):
    edit = persist_assignment(
        annotations, state_dir, get_active_context(),
        cell, value, expected=expected, record=record,
    )
    update_resident_gallery_cards(cell, value)
    return edit


def undo_edit(annotations: AnnotationFiles, state_dir: Path, edit) -> None:
    persist_undo(annotations, get_active_context(), state_dir, edit)
    update_resident_gallery_cards(edit.cell, edit.previous)


GALLERY_IDENTITIES = {
    "BrdU− / EdU−": (0, 0),
    "BrdU− / EdU+": (0, 1),
    "BrdU+ / EdU−": (1, 0),
    "BrdU+ / EdU+": (1, 1),
}
GALLERY_PAGE_SIZE = 24


def refresh_gallery_queue(signature: tuple) -> None:
    """Discard only the explicitly refreshed gallery queue."""
    gallery = get_active_context()["native"]["gallery_states"].pop(signature, None)
    prefetches = gallery["prefetches"] if gallery is not None else {}
    for futures in prefetches.values():
        for future in futures:
            future.cancel()
    get_active_context()["native"]["gallery_client_token"] = None


def gallery_target(card: dict, value: tuple[int, int]) -> tuple[int, int]:
    """Return the card's next valid assignment."""
    return {"brdu": toggle_brdu, "edu": toggle_edu}[card["marker"]](value)


def update_gallery_card(
    card: dict, value: tuple[int, int] | None, initial: tuple[int, int]
) -> None:
    """Update one resident card without rebuilding its encoded crop."""
    if value is None:
        card.update(state="No saved call", changed=True, action="Unavailable",
                    ariaLabel=f"{card['label']}, no saved call", disabled=True,
                    expected=None, value=None)
        return
    target = gallery_target(card, value)
    marker_name = {"brdu": "BrdU", "edu": "EdU"}[card["marker"]]
    marker_index = {"brdu": 0, "edu": 1}[card["marker"]]
    action = f"Set {marker_name}{'+' if target[marker_index] else '−'}"
    state = f"BrdU{'+' if value[0] else '−'} / EdU{'+' if value[1] else '−'}"
    aria_label = f"{card['label']}, {state}. {action}"
    card.update(
        state=state,
        changed=value != initial,
        action=action,
        ariaLabel=aria_label,
        disabled=False,
        expected=list(value),
        value=list(target),
    )


def update_resident_gallery_cards(cell: str, value: tuple[int, int] | None) -> None:
    """Patch a saved value into any loaded page containing that cell."""
    for gallery in get_active_context()["native"]["gallery_states"].values():
        initial = gallery["initial"].get(cell)
        if initial is not None:
            for page in gallery["pages"].values():
                for card in page["cards"]:
                    if card["cell"] == cell:
                        update_gallery_card(card, value, initial)


def new_gallery_state(
    rows, annotations: AnnotationFiles, *, kind: str, crop_kind: str
) -> dict:
    """Create one stable, session-resident gallery queue."""
    rows = rows.copy()
    if kind == "miscall":
        rows["item_id"] = rows["cell"].astype(str) + ":" + rows["audit_marker"]
    else:
        rows["item_id"] = rows["cell"].astype(str)
    if rows["item_id"].duplicated().any():
        raise ValueError("Gallery item identifiers must be unique")
    item_ids = rows["item_id"].tolist()
    cells = rows["cell"].astype(str).unique().tolist()
    return {
        "kind": kind,
        "crop_kind": crop_kind,
        "item_ids": item_ids,
        "initial": {cell: annotations.values[cell] for cell in cells},
        "rows": rows.set_index("item_id"),
        "queue_id": uuid4().hex,
        "page": 0,
        "pages": {},
        "prefetches": {},
    }


def review_gallery(
    source: FieldSource,
    annotations: AnnotationFiles,
    state_dir: Path,
    gallery: dict,
    *,
    title: str,
    caption: str,
    group_label: str,
    empty_message: str,
) -> None:
    """Render, persist, and undo one resident paginated crop gallery."""
    context = get_active_context()
    item_ids = gallery["item_ids"]
    total = len(item_ids)
    page_count = max(1, (total + GALLERY_PAGE_SIZE - 1) // GALLERY_PAGE_SIZE)
    page = min(int(gallery["page"]), page_count - 1)
    gallery["page"] = page
    prefetches = gallery["prefetches"]
    current_key = (gallery["queue_id"], page)
    current_prefetch = prefetches.get(current_key)
    if current_prefetch is not None:
        for future in current_prefetch:
            future.result()

    view_token = activate_view(context, ("native-gallery", gallery["queue_id"], page))
    st.title(title)
    st.caption(caption)
    undo_column, status_column = st.columns([1, 3], vertical_alignment="center")
    last_edit = context["history"][-1] if context["history"] else None
    undo_label = (
        f"Undo last edit · {last_edit.cell.rsplit('|', 1)[-1]}"
        if last_edit is not None
        else "Undo last edit"
    )
    if undo_column.button(
        undo_label,
        disabled=last_edit is None,
        key=f"gallery-undo:{view_token}",
        width="stretch",
    ):
        try:
            undo_edit(annotations, state_dir, last_edit)
        except LabelConflictError as exc:
            context["notice"] = str(exc)
        st.rerun()
    if context["notice"]:
        status_column.info(context["notice"])

    if not total:
        st.info(empty_message)
        return

    start = page * GALLERY_PAGE_SIZE
    stop = min(start + GALLERY_PAGE_SIZE, total)
    page_item_ids = item_ids[start:stop]

    crop_cells = {"edu": source.edu_cell_crop, "brdu": source.brdu_cell_crop}
    prefetch_crops = {"edu": prefetch_edu_crop, "brdu": prefetch_brdu_crop}

    pages = gallery["pages"]
    if current_key not in pages:
        cards = []
        missing = []
        for item_id in page_item_ids:
            row = gallery["rows"].loc[item_id]
            cell = str(row["cell"])
            current = annotations.values.get(cell)
            if current is None:
                missing.append(cell)
                continue
            label = int(cell.rsplit("|", 1)[1])
            crop_kind = (
                str(row["audit_marker"])
                if gallery["crop_kind"] == "audit_marker"
                else gallery["crop_kind"]
            )
            crop, plane_z = crop_cells[crop_kind](label)
            card = {
                "kind": gallery["kind"],
                "itemId": item_id,
                "cell": cell,
                "label": f"Cell {label}",
                "image": "data:image/png;base64,"
                + base64.b64encode(crop).decode("ascii"),
            }
            if gallery["kind"] == "miscall":
                marker_name = {"brdu": "BrdU", "edu": "EdU"}[crop_kind]
                card.update(
                    marker=crop_kind,
                    detail=(
                        f"{marker_name} miscall · Model {marker_name}"
                        f"{'+' if row[f'pred_{crop_kind}'] else '−'} · "
                        f"{str(row['split']).capitalize()} · {row['tranche']} · "
                        f"p({marker_name}+) {row[f'p_{crop_kind}']:.3f} · "
                        f"Z {source.origin[0] + plane_z * source.scale[0]:.1f} µm"
                    ),
                )
            else:
                card.update(
                    marker=gallery["marker"],
                    detail=(
                        f"{str(row['split']).capitalize()} · {row['tranche']} · "
                        f"Z {source.origin[0] + plane_z * source.scale[0]:.1f} µm"
                    ),
                )
            update_gallery_card(card, current, gallery["initial"][cell])
            cards.append(card)
        pages[current_key] = {"cards": cards, "missing": missing}
    page_data = pages[current_key]
    cards = page_data["cards"]
    for card in cards:
        update_gallery_card(card, annotations.values.get(card["cell"]), gallery["initial"][card["cell"]])
    if page_data["missing"]:
        st.warning(
            f"{len(page_data['missing'])} queued cell(s) no longer have a saved label. "
            "Refresh matches to remove them."
        )

    next_page = page + 1
    if next_page < page_count:
        next_key = (gallery["queue_id"], next_page)
        if next_key not in prefetches:
            next_start = next_page * GALLERY_PAGE_SIZE
            next_item_ids = item_ids[next_start : next_start + GALLERY_PAGE_SIZE]
            executor = gallery_prefetch_executor()
            prefetches[next_key] = tuple(
                executor.submit(
                    prefetch_crops[
                        str(gallery["rows"].loc[item_id, "audit_marker"])
                        if gallery["crop_kind"] == "audit_marker"
                        else gallery["crop_kind"]
                    ],
                    source,
                    str(gallery["rows"].loc[item_id, "cell"]),
                )
                for item_id in next_item_ids
            )

    patch_existing = context["native"]["gallery_client_token"] == view_token
    crop_source = context["workspace"].get("review_source")
    if crop_source is not None:
        from dinoapp.review_session import capture_queue
        cells = list(dict.fromkeys(card["cell"] for card in cards))
        chosen = st.selectbox("Cell to inspect", cells, key=f"miscall-open-choice:{view_token}")
        if st.button("Review cell", key=f"miscall-open-cell:{view_token}"):
            capture_queue(context, cells, crop_source, position=cells.index(chosen))
            st.rerun(scope="app")
    items = cards
    if patch_existing:
        items = [
            {key: value for key, value in card.items() if key != "image"}
            for card in cards
        ]
    result = gallery_picker()(
        data={
            "items": items,
            "token": view_token,
            "patch": patch_existing,
            "galleryLabel": group_label,
        },
        key="gallery-cards",
        on_flipped_change=lambda: None,
        on_needsFull_change=lambda: None,
    )
    context["native"]["gallery_client_token"] = view_token
    needs_full = getattr(result, "needsFull", None)
    if needs_full and needs_full.get("token") == view_token:
        context["native"]["gallery_client_token"] = None
        st.rerun()

    navigation_label, previous, following = st.columns([3, 1, 1])
    navigation_label.write(f"**Cells {start + 1:,}–{stop:,} of {total:,} loaded**")
    if previous.button("Previous", disabled=page == 0, width="stretch", key=f"gallery-prev:{view_token}"):
        gallery["page"] = page - 1
        st.rerun()
    if following.button("Next", disabled=page == page_count - 1, width="stretch", key=f"gallery-next:{view_token}"):
        gallery["page"] = page + 1
        st.rerun()

    event = result.flipped
    if not event:
        return

    item_id = str(event.get("itemId", ""))
    expected_raw = event.get("expected")
    value_raw = event.get("value")
    expected = tuple(expected_raw) if isinstance(expected_raw, list) else None
    value = tuple(value_raw) if isinstance(value_raw, list) else None
    card = next((card for card in cards if card["itemId"] == item_id), None)
    cell = str(card["cell"]) if card is not None else ""
    displayed = tuple(card["expected"]) if card is not None and card["expected"] is not None else None
    current = annotations.values.get(cell)
    if (
        event.get("token") != view_token
        or event.get("cell") != cell
        or current is None
        or expected != displayed
        or expected != current
    ):
        context["notice"] = (
            "That cell changed after this gallery view was rendered; no edit was saved."
        )
        st.rerun()
    target = gallery_target(card, current)
    if value != target:
        context["notice"] = (
            "Ignored an invalid gallery edit; no label was changed."
        )
        st.rerun()

    row = gallery["rows"].loc[item_id]
    if gallery["kind"] == "miscall":
        record = {
            "action": f"miscall_toggle_{card['marker']}",
            "split": str(row["split"]),
            "model_call": int(row[f"pred_{card['marker']}"]),
            "probability": float(row[f"p_{card['marker']}"]),
        }
    else:
        record = {
            "action": f"gallery_toggle_{gallery['marker']}",
            "split": gallery["split"],
            "gallery_identity": gallery["identity"],
        }
    try:
        edit = save_assignment(
            annotations,
            state_dir,
            cell,
            target,
            expected=current,
            record={**record, "queue_id": gallery["queue_id"]},
        )
    except LabelConflictError as exc:
        context["notice"] = str(exc)
    else:
        update_resident_gallery_cards(cell, target)
        context["notice"] = (
            f"Saved {cell} as {IDENTITIES[2 * target[0] + target[1]]} "
            f"in {edit.path.parent.name}."
        )
    st.rerun()


def review_marker_gallery(
    source: FieldSource,
    paths: tuple[Path, ...],
    state_dir: Path,
    marker: str,
) -> None:
    """Rapidly toggle one marker in a stable snapshot of labeled cell crops."""
    marker_name = {"brdu": "BrdU", "edu": "EdU"}[marker]
    other_name = {"brdu": "EdU", "edu": "BrdU"}[marker]
    annotations = AnnotationFiles(paths, source.default_annotation_path)
    with st.sidebar:
        st.header(f"{marker_name} correction gallery")
        split_name = st.selectbox(
            "Examples",
            ("Training", "Validation", "All labeled"),
            index=2,
            key=widget_key(get_active_context(), "gallery_split", 'All labeled'), on_change=remember_widget, args=(get_active_context(), "gallery_split"),
        )
        identity_name = st.selectbox(
            "Saved state",
            tuple(GALLERY_IDENTITIES),
            key=widget_key(get_active_context(), "gallery_identity", 'BrdU− / EdU−'), on_change=remember_widget, args=(get_active_context(), "gallery_identity"),
        )
        st.caption(
            "Matches use the current saved labels. The queue stays fixed while you edit."
        )
    split = {"Training": "training", "Validation": "validation", "All labeled": "all"}[
        split_name
    ]
    identity = GALLERY_IDENTITIES[identity_name]
    signature = (marker, split, identity)
    with st.sidebar:
        if st.button("Refresh matches", width="stretch"):
            refresh_gallery_queue(signature)
            st.rerun()

    gallery_states = get_active_context()["native"]["gallery_states"]
    if signature not in gallery_states:
        matches = training_gallery(
            source.training_predictions,
            annotations.values,
            split=split,
            identity=identity,
        )
        gallery = new_gallery_state(
            matches, annotations, kind="toggle", crop_kind=marker
        )
        gallery.update(marker=marker, split=split, identity=identity)
        gallery_states[signature] = gallery
    review_gallery(
        source,
        annotations,
        state_dir,
        gallery_states[signature],
        title=f"{marker_name} correction gallery",
        caption=(
            f"Click a crop to flip {marker_name} while keeping {other_name} unchanged. "
            "Each filter keeps its loaded crops until you refresh its matches."
        ),
        group_label=f"{marker_name} correction cells",
        empty_message="No cells match this saved state.",
    )


def review_training_miscalls(
    source: FieldSource,
    paths: tuple[Path, ...],
    state_dir: Path,
) -> None:
    """Correct ranked model disagreements from a resident crop gallery."""
    annotations = AnnotationFiles(paths, source.default_annotation_path)
    with st.sidebar:
        st.header("Training miscalls")
        split_name = st.selectbox(
            "Examples",
            ("Training", "Validation", "All labeled"),
            index=2,
            key=widget_key(get_active_context(), "miscall_split", 'All labeled'), on_change=remember_widget, args=(get_active_context(), "miscall_split"),
        )
        target_name = st.selectbox(
            "Disagreement", ("Either marker", "BrdU", "EdU"), key=widget_key(get_active_context(), "miscall_target", 'Either marker'), on_change=remember_widget, args=(get_active_context(), "miscall_target")
        )
        identity_name = st.selectbox(
            "Saved state",
            tuple(GALLERY_IDENTITIES),
            key=widget_key(get_active_context(), "miscall_identity", 'BrdU− / EdU−'), on_change=remember_widget, args=(get_active_context(), "miscall_identity"),
        )
        st.caption(
            "Ranked by the probability assigned against the current saved label."
        )
    split = {"Training": "training", "Validation": "validation", "All labeled": "all"}[
        split_name
    ]
    target = {"Either marker": "either", "BrdU": "brdu", "EdU": "edu"}[target_name]
    identity = GALLERY_IDENTITIES[identity_name]
    signature = ("miscall", split, target, identity)
    with st.sidebar:
        if st.button("Refresh matches", width="stretch"):
            refresh_gallery_queue(signature)
            st.rerun()

    gallery_states = get_active_context()["native"]["gallery_states"]
    if signature not in gallery_states:
        scored = training_miscalls(
            source.training_predictions,
            annotations.values,
            split=split,
            target=target,
            identity=identity,
        )
        gallery = new_gallery_state(
            scored, annotations, kind="miscall", crop_kind="audit_marker"
        )
        gallery.update(split=split, target=target, identity=identity)
        gallery_states[signature] = gallery
    review_gallery(
        source,
        annotations,
        state_dir,
        gallery_states[signature],
        title="Training miscall gallery",
        caption=(
            "Each marker disagreement is a separate card, so a cell can appear twice. "
            "Click a card to flip only its displayed marker; the queue stays fixed until refreshed."
        ),
        group_label="Training miscall cells",
        empty_message="No model/annotation disagreements match these filters.",
    )


def select_field_cell(source, field, paths, state_dir, *, event_token: str):
    """Apply the one-shot click before the fragment renders its selected crop."""
    context = get_active_context()
    event = st.session_state.get("picker-right-panel", {}).get("clicked")
    if not event or event.get("token") != event_token or context["event_token"] != event_token:
        return
    annotations = AnnotationFiles(paths, source.default_annotation_path)
    seed = field.seed
    calls = source.field_calls(field, annotations)
    label = pick_cell(field.mask, float(event["x"]), float(event["y"]))
    if label is None:
        context["native"]["selection"] = None
        context["notice"] = "No segmented cell at that pixel."
    else:
        cell = source.cell_id(label)
        index = int((field.ids == label).nonzero()[0][0])
        identity = int(calls[index])
        context["native"]["selection"] = {
            "label": label,
            "expected": annotations.values.get(cell),
            "x": float(event["x"]),
            "y": float(event["y"]),
        }
        if cell not in annotations.values and identity < 4:
            value = (identity // 2, identity % 2)
            try:
                save_assignment(
                    annotations,
                    state_dir,
                    cell,
                    value,
                    expected=None,
                    record={
                        "action": "confirm",
                        "seed": seed,
                        "position": field.position,
                        "z_um": field.z_um,
                        "z_index": field.z_index,
                        "center_xy_um": field.center_xy_um.tolist(),
                    },
                )
            except LabelConflictError as exc:
                context["native"]["selection"] = None
                context["notice"] = str(exc)
            else:
                context["native"]["selection"]["expected"] = value
                context["notice"] = (
                    f"Confirmed cell {label} as {IDENTITIES[identity]} in "
                    f"{annotations.destination(cell).parent.name}."
                )
        else:
            context["notice"] = ""


@st.fragment
def review_field(
    source, field, paths, state_dir, image, right_view, *,
    crop_plane_caption: Callable[[int], str] | None = None,
    selected_caption: Callable[[int], str] | None = None,
    show_seed: bool = True,
    render_cell_crop: Callable[[int], tuple[bytes, int]] | None = None,
):
    """Save edits and update vector markers without rerendering the field images."""
    context = get_active_context()
    geometry = f"{field.token}:{field.origin_xy_um.tolist()}:{field.scale_xy_um.tolist()}:{field.center_xy_um.tolist()}"
    if context["native"]["field_token"] != geometry:
        if context["native"]["selection"] is None:
            context["native"]["selections"].pop(context["native"]["field_token"], None)
        else:
            context["native"]["selections"][context["native"]["field_token"]] = context["native"]["selection"]
        context["native"]["field_token"] = geometry
        context["native"]["selection"] = context["native"]["selections"].get(geometry)
    previous_token = context["event_token"]
    field_location = ("field", geometry, right_view)
    selection_cell = source.cell_id(context["native"]["selection"]["label"]) if context["native"]["selection"] else ""
    selection_expected = str(context["native"]["selection"]["expected"]) if context["native"]["selection"] else ""
    view_token = activate_view(context, (*field_location, selection_cell, selection_expected))
    select_field_cell(source, field, paths, state_dir, event_token=view_token)
    annotations = AnnotationFiles(paths, source.default_annotation_path)
    if previous_token != view_token and context["native"]["selection"] is not None:
        context["native"]["selection"]["expected"] = annotations.values.get(source.cell_id(context["native"]["selection"]["label"]))
    focus = context["workspace"].pop("focus_cell", None)
    if focus is not None:
        labels = [int(label) for label in field.ids if label > 0 and source.cell_id(int(label)) == focus]
        if len(labels) != 1:
            raise ValueError(f"Located field does not contain {focus}")
        context["native"]["selection"] = {"label": labels[0], "expected": annotations.values.get(focus)}
    selection_cell = source.cell_id(context["native"]["selection"]["label"]) if context["native"]["selection"] else ""
    selection_expected = str(context["native"]["selection"]["expected"]) if context["native"]["selection"] else ""
    view_token = activate_view(context, (*field_location, selection_cell, selection_expected))
    seed = field.seed
    calls = source.field_calls(field, annotations)
    selection = context["native"]["selection"]
    selected = selection["label"] if selection else None
    z_caption = f"Z {field.z_um:.1f} µm" if field.z_um is not None else f"Z plane {field.z_index}"
    st.caption(
        (f"Seed {seed} · Position {field.position} · " if show_seed and seed is not None else "")
        + f"{z_caption} · center X {field.center_xy_um[0]:.1f}, Y {field.center_xy_um[1]:.1f} µm"
    )
    with st.container(height=112, border=False):
        if selected is None:
            st.info("Select a segmented cell in the field.")
        else:
            cell = source.cell_id(selected)
            selected_index = int((field.ids == selected).nonzero()[0][0])
            identity = IDENTITIES[int(calls[selected_index])]
            label_source = (
                "saved annotation" if cell in annotations.values else "model call"
            )
            st.markdown(f"**Cell {selected} · {identity}** ({label_source})")
            info = field.boundary_info.get(selected, {})
            membership = "no boundary assignment"
            if info.get("included_60") is True:
                membership = "included at 60% and 90%"
            elif info.get("included_90") is True:
                membership = "added at 90%"
            elif info.get("included_90") is False:
                membership = "excluded at both"
            distance = info.get("r_vz_um")
            if distance is not None:
                membership += f" · ventricular distance {distance:.1f} µm"
            st.caption(selected_caption(selected) if selected_caption else membership)
        if context["notice"]:
            st.write(context["notice"])
    actions = st.columns([1, 1, 1, 1, 1])
    choices = [
        ("EdU only", (0, 1)),
        ("BrdU only", (1, 0)),
        ("Dual positive", (1, 1)),
        ("Neither", (0, 0)),
    ]
    editable = selected is not None and (
        source.cell_id(selected) in annotations.values
        or (selected < len(source.calls) and source.calls[selected] < 4)
    )
    for column, (name, value) in zip(actions[:4], choices, strict=True):
        if column.button(name, disabled=not editable, width="stretch", key=f"field-assign:{view_token}:{value}"):
            cell = source.cell_id(selected)
            try:
                edit = save_assignment(
                    annotations,
                    state_dir,
                    cell,
                    value,
                    expected=selection["expected"],
                    record={
                        "action": "assign",
                        "seed": seed,
                        "position": field.position,
                        "z_um": field.z_um,
                        "z_index": field.z_index,
                        "center_xy_um": field.center_xy_um.tolist(),
                    },
                )
            except LabelConflictError as exc:
                context["native"]["selection"] = None
                context["notice"] = str(exc)
                st.rerun(scope="fragment")
            context["native"]["selection"]["expected"] = value
            context["notice"] = (
                f"Saved cell {selected} as {name} in {edit.path.parent.name}."
            )
            st.rerun(scope="fragment")
    if actions[-1].button(
        "Undo last edit", disabled=not context["history"], width="stretch", key=f"field-undo:{view_token}"
    ):
        edit = context["history"][-1]
        try:
            undo_edit(annotations, state_dir, edit)
        except LabelConflictError as exc:
            context["notice"] = str(exc)
            st.rerun(scope="fragment")
        if selected is not None and edit.cell == source.cell_id(selected):
            context["native"]["selection"]["expected"] = edit.previous
        context["notice"] = f"Undid the last edit to {edit.cell}."
        st.rerun(scope="fragment")
    crop_source = context["workspace"].get("field_crop_source")
    if crop_source is not None:
        from dinoapp.review_session import capture_queue
        selected_action, field_action = st.columns(2)
        if selected_action.button("Review selected cell", disabled=not editable, key=f"field-open-cell:{view_token}"):
            capture_queue(context, [source.cell_id(selected)], crop_source)
            st.rerun(scope="app")
        eligible = [source.cell_id(int(label)) for label in field.ids if label > 0 and label < len(source.calls) and source.calls[int(label)] < 4]
        if field_action.button("Review cells in this field", disabled=not eligible, key=f"field-open-queue:{view_token}"):
            capture_queue(context, eligible, crop_source)
            st.rerun(scope="app")
    marks = []
    colors = ["#404040", "#00ff00", "#ff00ff", "#ffffff", "#999999"]
    # The fixed image stays unchanged while X marks show every saved annotation.
    for mark_index, raw_label in enumerate(field.ids):
        label = int(raw_label)
        if source.cell_id(label) not in annotations.values:
            continue
        yy, xx = np.nonzero(field.mask == label)
        if len(xx):
            marks.append(
                {
                    "label": label,
                    "x": float((xx.mean() + 0.5) / field.mask.shape[1]),
                    "y": float((yy.mean() + 0.5) / field.mask.shape[0]),
                    "color": colors[int(calls[mark_index])],
                }
            )
    selected_point = (
        {"x": selection["x"], "y": selection["y"]}
        if selection and "x" in selection
        else None
    )
    left, right = st.columns(2)
    picker = image_picker()
    with left:
        st.subheader("Selected cell")
        if selected is None:
            st.caption("Click a cell in the field to inspect its widest Z slice.")
        elif not editable:
            st.caption("This segmented cell has no saved call or review crop geometry.")
        else:
            crop, plane_z = (render_cell_crop or source.cell_crop)(selected)
            st.markdown(
                f"""
                <div data-testid="selected-label-state" data-state="{identity}"
                     style="background: {IDENTITY_BACKGROUNDS[int(calls[selected_index])]};
                            color: {IDENTITY_FOREGROUNDS[int(calls[selected_index])]};
                            border: 1px solid #b8b8b8; border-radius: 0.5rem;
                            padding: 0.55rem 0.75rem; margin-bottom: 0.25rem; line-height: 1.15;">
                  <span style="font-size: 0.8rem; font-weight: 600; letter-spacing: 0.08em;">LABEL STATE</span><br>
                  <span style="font-size: 1.55rem; font-weight: 750;">{identity}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(label_source.capitalize())
            st.image(
                crop,
                caption=(f"Cell {selected} · {crop_plane_caption(plane_z)}" if crop_plane_caption else
                         f"Cell {selected} · largest-area slice · Z {source.origin[0] + plane_z * source.scale[0]:.1f} µm (plane {plane_z})"),
                width="stretch",
            )
    with right:
        st.subheader(
            "EdU / BrdU · cell identities"
            if right_view == "Cell identities"
            else "Combined model uncertainty"
        )
        if right_view == "Model uncertainty":
            st.caption(
                "Yellower cells are closer to either fitted decision threshold. "
                "Colors show log1p(max inverse margin), capped at the global 99.9th percentile."
            )
        picker(
            data={
                "image": image,
                "marks": marks,
                "selection": selected_point,
                "token": view_token,
                "label": (
                    "Cell identity field"
                    if right_view == "Cell identities"
                    else "Model uncertainty field"
                ),
            },
            key="picker-right-panel",
            on_clicked_change=lambda: None,
        )
