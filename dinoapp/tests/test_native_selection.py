"""A component click renders the requested crop without rendering the old cell."""
import json

import pytest
from dino_markers import DEFAULT_CONTRACT
from streamlit.components.v2.bidi_component.main import _make_trigger_id
from streamlit.testing.v1 import AppTest


def selection_app(annotation_path):
    import io
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np
    from PIL import Image
    import streamlit as st
    from dino_markers import default_config
    from dinoapp import native_app
    from dinoapp.review_session import activate_target
    from dinoapp.native import Field

    config = default_config()
    activate_target('selection-test', revision='selection-test', contract=config.contract)
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
    source = SimpleNamespace(config=config, default_annotation_path=path, cell_id=lambda x: f'test|{x}',
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
    update_label(path, 'test|2', (0, 1), contract=DEFAULT_CONTRACT, expected=(1, 1))
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
    from dino_markers import default_config
    from dinoapp import native_app
    from dinoapp.review_session import activate_target, remember_widget, widget_key
    from dinoapp.native import Field

    config = default_config()
    target = st.selectbox('Target', ('A', 'B'))
    context = activate_target(target, revision=target, contract=config.contract)
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
    source = SimpleNamespace(config=config, default_annotation_path=path, cell_id=lambda x: f'{target}|{x}',
                             field_calls=calls, calls=np.array([4, 0, 3]), cell_crop=crop,
                             origin=np.zeros(3), scale=np.ones(3))
    if view == 'Gallery':
        import pandas as pd
        from dinoapp.sources import AnnotationFiles

        source.marker_crop = lambda label, marker: crop(label)
        annotations = AnnotationFiles((path,), path, contract=config.contract)
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
    old_id = next(button for button in app.button if button.label == "Save").proto.id
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
    old_id = next(button for button in app.button if button.label == "Save").proto.id
    update_label(path, "test|1", (1, 1), contract=DEFAULT_CONTRACT, expected=(0, 0))
    before = path.read_bytes()
    send_click(app, x=.5, token=app.session_state.test_field_token)
    assert app.session_state.review_contexts["selection-test"]["native"]["selection"]["expected"] == (1, 1)
    send_delayed_field_button(app, old_id, monkeypatch)
    assert not app.exception
    assert path.read_bytes() == before
    assert not app.session_state.review_contexts["selection-test"]["history"]


def three_marker_field_app(root, initial):
    import io
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np
    from PIL import Image
    import streamlit as st
    from dino_markers import decode_call, encode_call, load_config
    from dinoapp import native_app
    from dinoapp.native import Field
    from dinoapp.review_session import activate_target

    root = Path(root)
    config = load_config(root / 'markers.json')
    contract = config.contract
    activate_target('three-markers', revision='three-markers', contract=contract)
    if 'rendered_cells' not in st.session_state:
        st.session_state.rendered_cells = []
        native_app.get_active_context()['native']['selection'] = initial

    def crop(label):
        st.session_state.rendered_cells.append(label)
        output = io.BytesIO()
        Image.new('RGB', (8, 8), (label * 40, 0, 0)).save(output, format='PNG')
        return output.getvalue(), 0

    uncalled = 2 ** len(contract.markers)
    model = np.array([uncalled, uncalled, encode_call(contract, (0, 1, 1))], dtype=np.uint8)

    def calls(field, annotations):
        result = model[field.ids].copy()
        for index, label in enumerate(field.ids):
            manual = annotations.values.get(f'test|{label}')
            if manual is not None:
                result[index] = encode_call(contract, manual)
        return result

    path = root / 'labels.csv'
    source = SimpleNamespace(config=config, default_annotation_path=path, cell_id=lambda x: f'test|{x}',
                             field_calls=calls, calls=model, cell_crop=crop,
                             origin=np.zeros(3), scale=np.ones(3))
    mask = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.uint32)
    field = Field({}, mask, np.array([0, 1, 2]), mask, mask > 0, np.zeros(2),
                  np.ones(2), np.zeros(2), 0., 0, 0, 0, [], {})
    state = native_app.get_active_context()['native']
    if state['field_token'] is None and state['selection'] is not None:
        state['field_token'] = f'{field.token}:{field.origin_xy_um.tolist()}:{field.scale_xy_um.tolist()}:{field.center_xy_um.tolist()}'
    native_app.review_field(source, field, (path,), root / 'review', '', 'Cell identities')
    st.session_state.test_field_token = native_app.get_active_context()['event_token']
    st.session_state.test_contract = contract
    st.session_state.test_decoded = decode_call(contract, int(model[2]))


@pytest.fixture
def three_marker_field(tmp_path):
    return open_three_marker_field(tmp_path, {'label': 1, 'expected': (1, 0, 1)})


def open_three_marker_field(tmp_path, initial):
    from dinoapp.native_app import image_picker

    image_picker.clear()
    markers = {
        'version': 1,
        'reference': {'key': 'dapi', 'label': 'DAPI'},
        'markers': [
            {'key': 'edu', 'label': 'EdU', 'color': '#00ff00'},
            {'key': 'brdu', 'label': 'BrdU', 'color': '#ff00ff'},
            {'key': 'pcna', 'label': 'PCNA', 'color': '#00ffff'},
        ],
        'presentation': {
            'crop_panels': ['dapi', 'edu', 'brdu', 'pcna'],
            'negative_color': [0.25, 0.25, 0.25],
            'uncalled_color': '#999999',
            'badge_colors': [
                {'state': {'edu': 1, 'brdu': 0, 'pcna': 1}, 'background': '#123456', 'foreground': '#fedcba'},
            ],
        },
    }
    (tmp_path / 'markers.json').write_text(json.dumps(markers))
    path = tmp_path / 'labels.csv'
    path.write_text('cell,edu,brdu,pcna\ntest|1,1,0,1\n')
    app = AppTest.from_function(three_marker_field_app, args=(str(tmp_path), initial)).run(timeout=20)
    assert not app.exception
    return app, path


def click_field_button(app, label, monkeypatch):
    from dinoapp import native_app

    # AppTest button events run the entire script; adapt only the framework rerun.
    rerun = native_app.st.rerun
    with monkeypatch.context() as framework:
        framework.setattr(native_app.st, 'rerun', lambda **kwargs: rerun())
        next(button for button in app.button if button.label == label).click().run()
    assert not app.exception


def test_field_panel_offers_per_marker_controls_and_badges_follow_the_config(three_marker_field):
    app, path = three_marker_field
    assert [widget.label for widget in app.segmented_control] == ['EdU', 'BrdU', 'PCNA']
    assert [widget.value for widget in app.segmented_control] == [1, 0, 1]
    badge = next(item.value for item in app.markdown if 'selected-label-state' in item.value)
    assert 'background: #123456' in badge and 'color: #fedcba' in badge
    assert 'EdU + PCNA only' in badge
    send_click(app, x=5 / 6, token=app.session_state.test_field_token)
    assert not app.exception
    badge = next(item.value for item in app.markdown if 'selected-label-state' in item.value)
    assert 'data-state="BrdU + PCNA only"' in badge
    assert 'background: #ff00ff' in badge or 'background: #00ffff' not in badge
    assert path.read_text() == 'cell,edu,brdu,pcna\ntest|1,1,0,1\ntest|2,0,1,1\n'
    assert app.session_state.review_contexts['three-markers']['history'][-1].value == (0, 1, 1)


def test_field_save_writes_every_marker_at_once_and_an_unchanged_draft_writes_nothing(three_marker_field, monkeypatch):
    app, path = three_marker_field
    context = app.session_state.review_contexts['three-markers']
    before = path.read_bytes()
    click_field_button(app, 'Save', monkeypatch)
    assert path.read_bytes() == before and not context['history']
    next(widget for widget in app.segmented_control if widget.label == 'BrdU').set_value(1).run()
    assert not app.exception
    assert path.read_bytes() == before
    assert context['drafts']['test|1'] == {'expected': (1, 0, 1), 'values': {'edu': 1, 'brdu': 1, 'pcna': 1}}
    click_field_button(app, 'Save', monkeypatch)
    assert path.read_text() == 'cell,edu,brdu,pcna\ntest|1,1,1,1\n'
    assert context['history'][-1].previous == (1, 0, 1)
    assert context['native']['selection']['expected'] == (1, 1, 1)
    assert context['drafts'] == {'test|1': {'expected': (1, 1, 1), 'values': {'edu': 1, 'brdu': 1, 'pcna': 1}}}
    journal_path, = path.parent.glob('review/changes-*.jsonl')
    assert json.loads(journal_path.read_text().splitlines()[0])['value'] == [1, 1, 1]


def test_unsaved_field_cell_with_a_model_call_starts_unset_and_saves_once_every_marker_is_chosen(tmp_path, monkeypatch):
    app, path = open_three_marker_field(tmp_path, {'label': 2, 'expected': None})
    context = app.session_state.review_contexts['three-markers']
    assert [widget.value for widget in app.segmented_control] == [None, None, None]
    assert next(button for button in app.button if button.label == 'Save').disabled
    badge = next(item.value for item in app.markdown if 'selected-label-state' in item.value)
    assert 'data-state="BrdU + PCNA only"' in badge and 'background: #' in badge
    for label, value in (('EdU', 0), ('BrdU', 1)):
        next(widget for widget in app.segmented_control if widget.label == label).set_value(value).run()
        assert next(button for button in app.button if button.label == 'Save').disabled
    next(widget for widget in app.segmented_control if widget.label == 'PCNA').set_value(1).run()
    assert not next(button for button in app.button if button.label == 'Save').disabled
    assert 'test|2' not in path.read_text()
    click_field_button(app, 'Save', monkeypatch)
    assert path.read_text() == 'cell,edu,brdu,pcna\ntest|1,1,0,1\ntest|2,0,1,1\n'
    assert context['history'][-1].previous is None
    assert context['drafts'] == {'test|2': {'expected': (0, 1, 1), 'values': {'edu': 0, 'brdu': 1, 'pcna': 1}}}


THREE_MARKER_CONFIG = {
    'version': 1,
    'reference': {'key': 'dapi', 'label': 'DAPI'},
    'markers': [
        {'key': 'edu', 'label': 'EdU', 'color': '#00ff00'},
        {'key': 'brdu', 'label': 'BrdU', 'color': '#ff00ff'},
        {'key': 'pcna', 'label': 'PCNA', 'color': '#00ffff'},
    ],
    'presentation': {
        'crop_panels': ['dapi', 'edu', 'brdu', 'pcna'],
        'negative_color': [0.25, 0.25, 0.25],
        'uncalled_color': '#999999',
    },
}


def send_flip(review, payload):
    states = review._tree.get_widget_states()
    component_id = review.get('bidi_component')[0].proto.id
    states.widgets.add(id=component_id, json_value='{}')
    event = states.widgets.add(id=_make_trigger_id(component_id, 'events'))
    event.json_trigger_value = json.dumps([{'event': 'flipped', 'value': payload}])
    review._run(states)


def native_gallery_app(root, mode):
    import io
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd
    from PIL import Image
    import streamlit as st
    from dino_markers import load_config
    from dinoapp import native_app
    from dinoapp.review_session import activate_target

    root = Path(root)
    config = load_config(root / 'markers.json')
    activate_target('native-gallery', revision='native-gallery', contract=config.contract)

    def crop(label, marker):
        output = io.BytesIO()
        Image.new('RGB', (8, 8), (label * 3, 0, 0)).save(output, format='PNG')
        return output.getvalue(), 0

    cells = [f'test|{label}' for label in range(1, 26)]
    predictions = pd.DataFrame({
        'cell': cells, 'label': range(1, 26), 'split': 'training', 'tranche': 'cortex',
        'p_edu': 0.1, 'p_brdu': 0.9, 'p_pcna': 0.2, 'pred_edu': 0, 'pred_brdu': 1, 'pred_pcna': 0,
    })
    source = SimpleNamespace(config=config, default_annotation_path=root / 'labels.csv',
                             training_predictions=predictions, marker_crop=crop,
                             origin=np.zeros(3), scale=np.ones(3))
    if mode == 'miscalls':
        native_app.review_training_miscalls(source, (root / 'labels.csv',), root / 'review')
    else:
        native_app.review_marker_gallery(source, (root / 'labels.csv',), root / 'review', 'pcna')
    st.session_state.test_token = native_app.get_active_context()['event_token']


@pytest.fixture
def native_gallery(tmp_path, monkeypatch):
    from dinoapp import native_app

    native_app.gallery_picker.clear()
    (tmp_path / 'markers.json').write_text(json.dumps(THREE_MARKER_CONFIG))
    (tmp_path / 'labels.csv').write_text(
        'cell,edu,brdu,pcna\n' + ''.join(f'test|{label},1,0,1\n' for label in range(1, 26))
    )
    requests = {'miscalls': [], 'gallery': [], 'prefetch': []}

    def rows(predictions, live_labels, *, contract, split, identity, target='absent'):
        kind = 'gallery' if target == 'absent' else 'miscalls'
        requests[kind].append(dict(contract=contract, split=split, identity=identity, target=target))
        selected = predictions[predictions['cell'].isin(live_labels)].copy()
        if kind == 'miscalls':
            selected['audit_marker'] = 'pcna'
        return selected.reset_index(drop=True)

    monkeypatch.setattr(native_app, 'training_gallery', rows)
    monkeypatch.setattr(native_app, 'training_miscalls', rows)
    monkeypatch.setattr(native_app, 'prefetch_marker_crop', lambda source, cell, marker: requests['prefetch'].append((cell, marker)))
    return tmp_path, requests


def test_marker_gallery_filters_per_marker_and_a_card_flips_only_its_marker(native_gallery):
    root, requests = native_gallery
    app = AppTest.from_function(native_gallery_app, args=(str(root), 'gallery')).run(timeout=20)
    assert not app.exception
    assert [widget.label for widget in app.sidebar.selectbox] == ['Examples', 'EdU', 'BrdU', 'PCNA']
    assert [widget.value for widget in app.sidebar.selectbox][1:] == ['Any', 'Any', 'Any']
    assert requests['gallery'][-1]['identity'] == (None, None, None)
    assert requests['gallery'][-1]['contract'].markers == ('edu', 'brdu', 'pcna')
    app.sidebar.selectbox[1].select('EdU+').run()
    app.sidebar.selectbox[2].select('BrdU−').run()
    assert requests['gallery'][-1]['identity'] == (1, 0, None)
    context = app.session_state.review_contexts['native-gallery']
    (gallery,) = [g for key, g in context['native']['gallery_states'].items() if key[-1] == (1, 0, None)]
    cards = gallery['pages'][(gallery['queue_id'], 0)]['cards']
    card = cards[0]
    assert (card['state'], card['action'], card['expected'], card['value']) == (
        'EdU+ / BrdU− / PCNA+', 'Set PCNA−', [1, 0, 1], [1, 0, 0])
    send_flip(app, {'itemId': card['itemId'], 'cell': card['cell'], 'expected': card['expected'],
                    'value': card['value'], 'token': app.session_state.test_token})
    assert not app.exception
    rows = (root / 'labels.csv').read_text().splitlines()
    assert rows[1] == 'test|1,1,0,0' and rows[2] == 'test|10,1,0,1'
    assert context['notice'].startswith('Saved test|1 as EdU only in')
    journal_path, = (root / 'review').glob('changes-*.jsonl')
    record = json.loads(journal_path.read_text().splitlines()[0])
    assert (record['action'], record['value'], record['gallery_identity']) == ('gallery_toggle_pcna', [1, 0, 0], [1, 0, None])


def test_gallery_prefetches_the_next_page_through_the_generic_marker_crop(native_gallery):
    root, requests = native_gallery
    app = AppTest.from_function(native_gallery_app, args=(str(root), 'gallery')).run(timeout=20)
    assert not app.exception
    assert requests['prefetch'] == [('test|25', 'pcna')]


def test_miscalls_name_the_disagreeing_marker_and_flip_only_that_marker(native_gallery):
    root, requests = native_gallery
    app = AppTest.from_function(native_gallery_app, args=(str(root), 'miscalls')).run(timeout=20)
    assert not app.exception
    labels = [widget.label for widget in app.sidebar.selectbox]
    assert labels == ['Examples', 'Disagreement', 'EdU', 'BrdU', 'PCNA']
    disagreement = app.sidebar.selectbox[1]
    assert disagreement.options == ['Any marker', 'EdU', 'BrdU', 'PCNA'] and disagreement.value == 'Any marker'
    assert requests['miscalls'][-1]['target'] is None
    disagreement.select('PCNA').run()
    app.sidebar.selectbox[3].select('BrdU−').run()
    assert (requests['miscalls'][-1]['target'], requests['miscalls'][-1]['identity']) == ('pcna', (None, 0, None))
    context = app.session_state.review_contexts['native-gallery']
    (gallery,) = [g for key, g in context['native']['gallery_states'].items() if key[1:] == ('all', 'pcna', (None, 0, None))]
    card = gallery['pages'][(gallery['queue_id'], 0)]['cards'][0]
    assert card['marker'] == 'pcna' and 'PCNA miscall · Model PCNA− · ' in card['detail'] and 'p(PCNA+) 0.200' in card['detail']
    send_flip(app, {'itemId': card['itemId'], 'cell': card['cell'], 'expected': card['expected'],
                    'value': card['value'], 'token': app.session_state.test_token})
    assert not app.exception
    assert (root / 'labels.csv').read_text().splitlines()[1] == 'test|1,1,0,0'
    journal_path, = (root / 'review').glob('changes-*.jsonl')
    record = json.loads(journal_path.read_text().splitlines()[0])
    assert (record['action'], record['model_call'], record['probability']) == ('miscall_toggle_pcna', 0, 0.2)


def test_two_marker_miscalls_offer_either_marker(tmp_path, monkeypatch):
    import pandas as pd
    from dinoapp import native_app

    (tmp_path / 'markers.json').write_text(json.dumps({**THREE_MARKER_CONFIG, 'markers': THREE_MARKER_CONFIG['markers'][:2],
                                                       'presentation': {**THREE_MARKER_CONFIG['presentation'],
                                                                        'crop_panels': ['dapi', 'edu', 'brdu']}}))
    (tmp_path / 'labels.csv').write_text('cell,edu,brdu\n')
    native_app.gallery_picker.clear()
    monkeypatch.setattr(native_app, 'training_miscalls', lambda *args, **kwargs: pd.DataFrame(
        columns=['cell', 'audit_marker', 'split', 'tranche']))
    app = AppTest.from_function(native_gallery_app, args=(str(tmp_path), 'miscalls')).run(timeout=20)
    assert not app.exception
    assert app.sidebar.selectbox[1].options == ['Either marker', 'EdU', 'BrdU']
