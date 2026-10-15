"""A component click renders the requested crop without rendering the old cell."""
import json

import pytest
from streamlit.components.v2.bidi_component.main import _make_trigger_id
from streamlit.testing.v1 import AppTest


def selection_app(annotation_path):
    import io
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np
    from PIL import Image
    import streamlit as st
    from dinoapp import native_app
    from dinoapp.review_session import activate_target
    from dinoapp.native import Field

    activate_target('selection-test', revision='selection-test')
    if 'rendered_cells' not in st.session_state:
        st.session_state.rendered_cells = []
        native_app.get_active_context()['native']['selection'] = {'label': 1, 'expected': (0, 0)}

    def crop(label):
        st.session_state.rendered_cells.append(label)
        output = io.BytesIO()
        Image.new('RGB', (8, 8), (label * 40, 0, 0)).save(output, format='PNG')
        return output.getvalue(), 0

    def calls(field, annotations):
        result = np.array([4, 0, 3], dtype=np.uint8)
        for index, label in enumerate(field.ids):
            manual = annotations.values.get(f'test|{label}')
            if manual is not None:
                result[index] = 2 * manual[0] + manual[1]
        return result

    path = Path(annotation_path)
    source = SimpleNamespace(default_annotation_path=path, cell_id=lambda x: f'test|{x}',
                             field_calls=calls, calls=np.array([4, 0, 3]), cell_crop=crop,
                             origin=np.zeros(3), scale=np.ones(3))
    mask = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.uint32)
    field = Field({}, mask, np.array([0, 1, 2]), mask, mask > 0, np.zeros(2),
                  np.ones(2), np.zeros(2), 0., 0, 0, 0, [], {})
    state = native_app.get_active_context()['native']
    if state['field_token'] is None and state['selection'] is not None:
        state['field_token'] = f'{field.token}:{field.origin_xy_um.tolist()}:{field.scale_xy_um.tolist()}:{field.center_xy_um.tolist()}'
    native_app.review_field(source, field, (path,), path.parent, '', 'Cell identities')
    st.session_state.test_field_token = native_app.get_active_context()['event_token']


def send_click(app, *, x, token):
    # AppTest does not expose a CCv2 event helper; submit the real trigger wire format.
    states = app._tree.get_widget_states()
    component_id = app.get('bidi_component')[0].proto.id
    states.widgets.add(id=component_id, json_value='{}')
    event = states.widgets.add(id=_make_trigger_id(component_id, 'events'))
    event.json_trigger_value = json.dumps([{'event': 'clicked', 'value': {'x': x, 'y': .5, 'token': token}}])
    app._run(states)


@pytest.fixture
def selection_review(tmp_path):
    from dinoapp.native_app import image_picker
    image_picker.clear()
    path = tmp_path / 'labels.csv'
    path.write_text('cell,brdu,edu\ntest|1,0,0\ntest|2,1,1\n')
    app = AppTest.from_function(selection_app, args=(str(path),)).run(timeout=20)
    assert not app.exception
    app.session_state.rendered_cells = []
    return app, path


def test_click_renders_only_new_selected_crop(selection_review):
    app, path = selection_review
    before = path.read_bytes()
    send_click(app, x=5 / 6, token=app.session_state.test_field_token)
    assert app.session_state.rendered_cells == [2]
    assert not app.exception
    assert app.session_state.review_contexts['selection-test']['native']['selection']['label'] == 2
    assert path.read_bytes() == before
    assert any('Dual positive' in item.value for item in app.markdown)


@pytest.mark.parametrize('stale', [False, True])
def test_background_and_stale_clicks_preserve_selection_contract(selection_review, stale):
    app, path = selection_review
    before = path.read_bytes()
    send_click(app, x=1 / 6, token='old-field' if stale else app.session_state.test_field_token)
    assert app.session_state.review_contexts['selection-test']['native']['selection'] == ({'label': 1, 'expected': (0, 0)} if stale else None)
    assert app.session_state.rendered_cells == ([1] if stale else [])
    assert path.read_bytes() == before


def test_new_cell_is_confirmed_once_before_its_crop(selection_review):
    app, path = selection_review
    path.write_text('cell,brdu,edu\ntest|1,0,0\n')
    send_click(app, x=5 / 6, token=app.session_state.test_field_token)
    assert not app.exception
    assert app.session_state.rendered_cells == [2]
    assert app.session_state.review_contexts['selection-test']['native']['selection']['expected'] == (1, 1)
    assert len(app.session_state.review_contexts['selection-test']['history']) == 1
    assert 'test|2,1,1' in path.read_text()
    app.run()
    assert not app.exception
    assert len(app.session_state.review_contexts['selection-test']['history']) == 1


def test_confirmation_conflict_clears_selection_without_rendering_old_crop(selection_review, monkeypatch):
    from dinoapp import native_app
    from dinoapp.label_store import LabelConflictError

    app, path = selection_review
    path.write_text('cell,brdu,edu\ntest|1,0,0\n')
    before = path.read_bytes()

    def conflicting_assignment(*args, **kwargs):
        raise LabelConflictError('concurrent annotation')

    monkeypatch.setattr(native_app, 'save_assignment', conflicting_assignment)
    send_click(app, x=5 / 6, token=app.session_state.test_field_token)
    assert not app.exception
    assert app.session_state.review_contexts['selection-test']['native']['selection'] is None
    assert app.session_state.review_contexts['selection-test']['notice'] == 'concurrent annotation'
    assert app.session_state.rendered_cells == []
    assert path.read_bytes() == before


def test_conflicted_field_undo_retains_shared_history(selection_review, monkeypatch):
    from dinoapp.label_store import update_label

    app, path = selection_review
    path.write_text('cell,brdu,edu\ntest|1,0,0\n')
    send_click(app, x=5 / 6, token=app.session_state.test_field_token)
    history = app.session_state.review_contexts['selection-test']['history']
    assert len(history) == 1
    edit = history[-1]
    journal_path, = path.parent.glob('changes-*.jsonl')
    journal_before = journal_path.read_bytes()
    update_label(path, 'test|2', (0, 1), expected=(1, 1))
    before = path.read_bytes()
    from dinoapp import native_app

    # AppTest button events run the entire script; adapt only the framework rerun.
    rerun = native_app.st.rerun
    with monkeypatch.context() as framework:
        framework.setattr(native_app.st, 'rerun', lambda **kwargs: rerun())
        next(button for button in app.button if button.label == 'Undo last edit').click().run()
    assert not app.exception
    assert app.session_state.review_contexts['selection-test']['history'] == [edit]
    assert path.read_bytes() == before
    assert journal_path.read_bytes() == journal_before
    assert 'changed' in app.session_state.review_contexts['selection-test']['notice'].lower()


def switching_selection_app(root):
    import io
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np
    from PIL import Image
    import streamlit as st
    from dinoapp import native_app
    from dinoapp.review_session import activate_target, remember_widget, widget_key
    from dinoapp.native import Field

    target = st.selectbox('Target', ('A', 'B'))
    context = activate_target(target, revision=target)
    view = st.selectbox('View', ('Field', 'Gallery'), key=widget_key(context, 'test-view', 'Field'),
                        on_change=remember_widget, args=(context, 'test-view'))
    annotation_path = str(Path(root) / f'{target}.csv')
    if 'rendered_cells' not in st.session_state:
        st.session_state.rendered_cells = []
        native_app.get_active_context()['native']['selection'] = {'label': 1, 'expected': (0, 0)}

    def crop(label):
        st.session_state.rendered_cells.append(label)
        output = io.BytesIO()
        Image.new('RGB', (8, 8), (label * 40, 0, 0)).save(output, format='PNG')
        return output.getvalue(), 0

    def calls(field, annotations):
        result = np.array([4, 0, 3], dtype=np.uint8)
        for index, label in enumerate(field.ids):
            manual = annotations.values.get(f'{target}|{label}')
            if manual is not None:
                result[index] = 2 * manual[0] + manual[1]
        return result

    path = Path(annotation_path)
    source = SimpleNamespace(default_annotation_path=path, cell_id=lambda x: f'{target}|{x}',
                             field_calls=calls, calls=np.array([4, 0, 3]), cell_crop=crop,
                             origin=np.zeros(3), scale=np.ones(3))
    if view == 'Gallery':
        import pandas as pd
        from dinoapp.sources import AnnotationFiles

        source.edu_cell_crop = crop
        source.brdu_cell_crop = crop
        annotations = AnnotationFiles((path,), path)
        galleries = context['native']['gallery_states']
        if 'switch-test' not in galleries:
            rows = pd.DataFrame({'cell': list(annotations.values), 'split': 'training', 'tranche': 'cortex'})
            galleries['switch-test'] = native_app.new_gallery_state(rows, annotations, kind='toggle', crop_kind='edu')
            galleries['switch-test'].update(marker='edu', split='training', identity=(0, 0))
        native_app.review_gallery(source, annotations, path.parent, galleries['switch-test'],
                                  title='Gallery', caption='', group_label='Cells', empty_message='Empty')
        st.session_state.test_gallery_token = context['event_token']
        return
    position = int(st.number_input('Position', min_value=0, step=1,
                                  key=widget_key(context, 'position', 0),
                                  on_change=remember_widget, args=(context, 'position')))
    mask = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.uint32)
    field = Field({}, mask, np.array([0, 1, 2]), mask, mask > 0, np.zeros(2),
                  np.ones(2), np.zeros(2), 0., 0, 0, position, [], {})
    state = native_app.get_active_context()['native']
    if state['field_token'] is None and state['selection'] is not None:
        state['field_token'] = f'{field.token}:{field.origin_xy_um.tolist()}:{field.scale_xy_um.tolist()}:{field.center_xy_um.tolist()}'
    native_app.review_field(source, field, (path,), path.parent, '', 'Cell identities')
    st.session_state.test_field_token = native_app.get_active_context()['event_token']



def test_pending_field_click_cannot_cross_targets(tmp_path):
    from dinoapp.native_app import image_picker

    image_picker.clear()
    for target in ('A', 'B'):
        (tmp_path / f'{target}.csv').write_text(f'cell,brdu,edu\n{target}|1,0,0\n')
    review = AppTest.from_function(switching_selection_app, args=(str(tmp_path),)).run(timeout=20)
    assert not review.exception
    old_token = review.session_state.test_field_token
    review.selectbox[0].select('B').run()
    assert not review.exception
    before = (tmp_path / 'B.csv').read_bytes()
    send_click(review, x=5 / 6, token=old_token)
    assert not review.exception
    assert (tmp_path / 'B.csv').read_bytes() == before
    assert review.session_state.review_contexts['B']['history'] == []


@pytest.mark.parametrize('revisit', ['view', 'target'])
def test_revisited_field_restores_controls_and_rejects_old_click(tmp_path, revisit):
    from dinoapp.native_app import image_picker, gallery_picker

    image_picker.clear()
    gallery_picker.clear()
    for target in ('A', 'B'):
        (tmp_path / f'{target}.csv').write_text(f'cell,brdu,edu\n{target}|1,0,0\n')
    review = AppTest.from_function(switching_selection_app, args=(str(tmp_path),)).run(timeout=20)
    review.number_input[0].set_value(7).run()
    old_token = review.session_state.test_field_token
    if revisit == 'view':
        next(widget for widget in review.selectbox if widget.label == 'View').select('Gallery').run()
        next(widget for widget in review.selectbox if widget.label == 'View').select('Field').run()
    else:
        review.selectbox[0].select('B').run()
        assert review.number_input[0].value == 0
        review.selectbox[0].select('A').run()
    assert not review.exception
    assert review.number_input[0].value == 7
    assert review.session_state.test_field_token != old_token
    before = (tmp_path / 'A.csv').read_bytes()
    send_click(review, x=5 / 6, token=old_token)
    assert not review.exception
    assert (tmp_path / 'A.csv').read_bytes() == before
    fresh_token = review.session_state.test_field_token
    send_click(review, x=5 / 6, token=fresh_token)
    assert not review.exception
    assert 'A|2,1,1' in (tmp_path / 'A.csv').read_text()
    assert len(review.session_state.review_contexts['A']['history']) == 1
    selected_token = review.session_state.test_field_token
    assert selected_token != fresh_token
    review.run()
    assert review.session_state.test_field_token == selected_token
    assert len(review.session_state.review_contexts['A']['history']) == 1


def test_revisited_gallery_rejects_old_flip_without_rebuilding_queue(tmp_path):
    from dinoapp.native_app import image_picker, gallery_picker

    image_picker.clear()
    gallery_picker.clear()
    path = tmp_path / 'A.csv'
    path.write_text('cell,brdu,edu\nA|1,0,0\n')
    review = AppTest.from_function(switching_selection_app, args=(str(tmp_path),)).run(timeout=20)
    next(widget for widget in review.selectbox if widget.label == 'View').select('Gallery').run()
    old_token = review.session_state.test_gallery_token
    context = review.session_state.review_contexts['A']
    queue_id = context['native']['gallery_states']['switch-test']['queue_id']
    next(widget for widget in review.selectbox if widget.label == 'View').select('Field').run()
    next(widget for widget in review.selectbox if widget.label == 'View').select('Gallery').run()
    assert not review.exception
    assert review.session_state.test_gallery_token != old_token
    before = path.read_bytes()
    states = review._tree.get_widget_states()
    component_id = review.get('bidi_component')[0].proto.id
    states.widgets.add(id=component_id, json_value='{}')
    event = states.widgets.add(id=_make_trigger_id(component_id, 'events'))
    event.json_trigger_value = json.dumps([{'event': 'flipped', 'value': {
        'itemId': 'A|1', 'cell': 'A|1', 'expected': [0, 0], 'value': [0, 1], 'token': old_token,
    }}])
    review._run(states)
    assert not review.exception
    assert path.read_bytes() == before
    context = review.session_state.review_contexts['A']
    assert context['native']['gallery_states']['switch-test']['queue_id'] == queue_id
    assert context['history'] == []


def send_delayed_field_button(app, button_id, monkeypatch):
    from dinoapp import native_app

    states = app._tree.get_widget_states()
    matching = next((state for state in states.widgets if state.id == button_id), None)
    if matching is None:
        states.widgets.add(id=button_id, bool_value=True)
    else:
        matching.bool_value = True
    rerun = native_app.st.rerun
    with monkeypatch.context() as framework:
        framework.setattr(native_app.st, 'rerun', lambda **kwargs: rerun())
        app._run(states)


@pytest.mark.parametrize("return_to_first", [False, True])
def test_delayed_assignment_for_previous_field_selection_cannot_edit_new_cell(selection_review, monkeypatch, return_to_first):
    app, path = selection_review
    old_id = next(button for button in app.button if button.label == "EdU only").proto.id
    before = path.read_bytes()
    send_click(app, x=5 / 6, token=app.session_state.test_field_token)
    assert app.session_state.review_contexts["selection-test"]["native"]["selection"]["label"] == 2
    if return_to_first:
        send_click(app, x=.5, token=app.session_state.test_field_token)
        assert app.session_state.review_contexts["selection-test"]["native"]["selection"]["label"] == 1
    send_delayed_field_button(app, old_id, monkeypatch)
    assert not app.exception
    assert path.read_bytes() == before
    assert not app.session_state.review_contexts["selection-test"]["history"]


def test_same_field_cell_reselection_rejects_action_rendered_before_external_edit(selection_review, monkeypatch):
    from dinoapp.label_store import update_label
    app, path = selection_review
    old_id = next(button for button in app.button if button.label == "EdU only").proto.id
    update_label(path, "test|1", (1, 1), expected=(0, 0))
    before = path.read_bytes()
    send_click(app, x=.5, token=app.session_state.test_field_token)
    assert app.session_state.review_contexts["selection-test"]["native"]["selection"]["expected"] == (1, 1)
    send_delayed_field_button(app, old_id, monkeypatch)
    assert not app.exception
    assert path.read_bytes() == before
    assert not app.session_state.review_contexts["selection-test"]["history"]
