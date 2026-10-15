import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


def shared_history_app(root):
    from pathlib import Path

    import streamlit as st

    from dinoapp import app, native_app
    from dinoapp.review_session import activate_target
    from dinoapp.sources import AnnotationFiles

    root = Path(root)
    original = root / 'original' / 'labels.csv'
    default = root / 'fresh' / 'labels.csv'
    paths = (original, default)
    state_dir = root / 'review'
    activate_target('shared-target', revision='shared-target')
    context = app.get_context('shared-target')
    if st.button('Confirm field'):
        native_app.save_assignment(
            AnnotationFiles(paths, default), state_dir, 'test|2', (1, 1),
            expected=None, record={'action': 'confirm'},
        )
    if st.button('Correct gallery'):
        app.save_assignment(
            AnnotationFiles(paths, default), state_dir, context, 'test|1', (0, 1),
            expected=(0, 0), record={'action': 'gallery_toggle_edu'},
        )
    if st.button('Undo cell'):
        app.undo_edit(AnnotationFiles(paths, default), context, state_dir, context['history'][-1])
    st.session_state.shared_history_size = len(context['history'])


def test_field_and_gallery_share_cell_review_history(tmp_path: Path):
    for directory in ('original', 'fresh'):
        (tmp_path / directory).mkdir()
    original = tmp_path / 'original' / 'labels.csv'
    original.write_text('cell,brdu,edu\ntest|1,0,0\n')
    review = AppTest.from_function(shared_history_app, args=(str(tmp_path),)).run(timeout=20)
    assert not review.exception
    review.button[0].click().run()
    assert not review.exception
    assert review.session_state.shared_history_size == 1
    review.button[1].click().run()
    assert not review.exception
    assert review.session_state.shared_history_size == 2
    review.button[2].click().run()
    assert not review.exception
    assert review.session_state.shared_history_size == 1
    assert original.read_text() == 'cell,brdu,edu\ntest|1,0,0\n'

    review.button[2].click().run()
    assert not review.exception
    assert review.session_state.shared_history_size == 0
    assert (tmp_path / 'fresh' / 'labels.csv').read_text() == 'cell,brdu,edu\n'
    journal_path, = (tmp_path / 'review').glob('changes-*.jsonl')
    records = [json.loads(line) for line in journal_path.read_text().splitlines()]
    assert [record['action'] for record in records] == ['confirm', 'gallery_toggle_edu', 'undo', 'undo']
    assert [record['cell'] for record in records] == ['test|2', 'test|1', 'test|1', 'test|2']
    assert [Path(record['path']).parent.name for record in records] == ['fresh', 'original', 'original', 'fresh']
    backup, = (tmp_path / 'review').glob('original-*.csv')
    assert backup.read_text() == 'cell,brdu,edu\ntest|1,0,0\n'


def test_conflicted_save_and_undo_preserve_history_and_journal(tmp_path, monkeypatch):
    from dinoapp.label_store import LabelConflictError, load_labels, update_label
    from dinoapp.review_session import get_context, save_assignment, undo_edit
    from dinoapp.sources import AnnotationFiles
    import streamlit as st

    monkeypatch.setattr(st, 'session_state', {})
    path = tmp_path / 'labels.csv'
    update_label(path, 'test|1', (0, 0))
    context = get_context('target')
    state_dir = tmp_path / 'review'
    annotations = AnnotationFiles((path,), path)
    edit = save_assignment(annotations, state_dir, context, 'test|1', (1, 0),
                           expected=(0, 0), record={'action': 'assign'})
    journal_path, = state_dir.glob('changes-*.jsonl')
    journal_before = journal_path.read_bytes()
    update_label(path, 'test|1', (0, 1), expected=(1, 0))
    with pytest.raises(LabelConflictError):
        save_assignment(annotations, state_dir, context, 'test|1', (1, 1),
                        expected=(1, 0), record={'action': 'assign'})
    with pytest.raises(LabelConflictError):
        undo_edit(annotations, context, state_dir, edit)
    assert context['history'] == [edit]
    assert journal_path.read_bytes() == journal_before
    assert load_labels(path)[['brdu', 'edu']].values.tolist() == [[0, 1]]


def pending_gallery_app(annotation_path):
    import io
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd
    from PIL import Image
    import streamlit as st
    from dinoapp import native_app
    from dinoapp.review_session import activate_target
    from dinoapp.sources import AnnotationFiles

    activate_target('gallery-target', revision='gallery-target')
    path = Path(annotation_path)
    annotations = AnnotationFiles((path,), path)
    if 'queued_gallery' not in st.session_state:
        rows = pd.DataFrame({'cell': ['test|1'], 'split': ['training'], 'tranche': ['cortex']})
        st.session_state.queued_gallery = native_app.new_gallery_state(
            rows, annotations, kind='toggle', crop_kind='edu',
        )
        st.session_state.queued_gallery.update(marker="edu", split="training", identity=(0, 0))
    output = io.BytesIO()
    Image.new('RGB', (8, 8)).save(output, format='PNG')
    source = SimpleNamespace(edu_cell_crop=lambda label: (output.getvalue(), 0),
                             brdu_cell_crop=lambda label: (output.getvalue(), 0),
                             origin=np.zeros(3), scale=np.ones(3))
    native_app.review_gallery(source, annotations, path.parent, st.session_state.queued_gallery,
                              title='Gallery', caption='', group_label='Cells', empty_message='Empty')


def send_gallery_flip(review, payload):
    from streamlit.components.v2.bidi_component.main import _make_trigger_id

    states = review._tree.get_widget_states()
    component_id = review.get('bidi_component')[0].proto.id
    states.widgets.add(id=component_id, json_value='{}')
    event = states.widgets.add(id=_make_trigger_id(component_id, 'events'))
    event.json_trigger_value = json.dumps([{'event': 'flipped', 'value': payload}])
    review._run(states)


def test_pending_gallery_flip_after_annotation_removal_is_rejected(tmp_path):
    from dinoapp import native_app
    from dinoapp.label_store import update_label

    native_app.gallery_picker.clear()
    path = tmp_path / 'labels.csv'
    update_label(path, 'test|1', (0, 0))
    review = AppTest.from_function(pending_gallery_app, args=(str(path),)).run(timeout=20)
    assert not review.exception
    queue_id = review.session_state.queued_gallery['queue_id']
    update_label(path, 'test|1', None, expected=(0, 0))
    before = path.read_bytes()
    send_gallery_flip(review, {
        'itemId': 'test|1', 'cell': 'test|1', 'expected': [0, 0], 'value': [0, 1],
        'token': review.session_state.review_contexts['gallery-target']['event_token'],
    })
    assert not review.exception
    assert path.read_bytes() == before
    assert review.session_state.review_contexts['gallery-target']['history'] == []
    assert 'no edit was saved' in review.session_state.review_contexts['gallery-target']['notice']
    gallery = review.session_state.queued_gallery
    assert gallery['queue_id'] == queue_id
    assert gallery['item_ids'] == ['test|1']
    card, = gallery['pages'][(queue_id, 0)]['cards']
    assert card['disabled'] is True
    assert card['expected'] is None


def test_model_revision_cancels_queues_without_moving_annotation_history(tmp_path, monkeypatch):
    from concurrent.futures import Future
    import streamlit as st
    from dinoapp.review_session import activate_target, activate_view, save_assignment
    from dinoapp.sources import AnnotationFiles

    monkeypatch.setattr(st, 'session_state', {})
    path = tmp_path / 'labels.csv'
    context = activate_target('target', revision='model-1:display-1')
    edit = save_assignment(AnnotationFiles((path,), path), tmp_path / 'review', context,
                           'cell|1', (1, 0), expected=None, record={'action': 'assign'})
    journal_path, = (tmp_path / 'review').glob('changes-*.jsonl')
    before = (path.read_bytes(), journal_path.read_bytes(), context['session_id'])
    future, native_future = Future(), Future()
    context['galleries']['queue'] = {'prefetches': {'page': future}}
    context['native']['gallery_states']['queue'] = {'prefetches': {'page': (native_future,)}}
    context['cell_queues']['queue'] = {'cells': ['cell|1'], 'position': 0}
    token = activate_view(context, ('gallery', 'queue', 0))
    assert activate_view(context, ('gallery', 'queue', 0)) == token
    assert activate_target('target', revision='model-1:display-1') is context
    assert activate_view(context, ('gallery', 'queue', 0)) == token
    assert not future.cancelled()
    context['workspace'].update(queue={'cells': ['target|1']}, active_field=object(),
                                focus_cell='target|1', field_crop_source=object())
    assert activate_target('target', revision='model-2:display-1') is context
    assert context['history'] == [edit]
    assert (path.read_bytes(), journal_path.read_bytes(), context['session_id']) == before
    assert context['galleries'] == context['cell_queues'] == context['native']['gallery_states'] == {}
    assert context['workspace']['queue'] is None
    assert not {'active_field', 'field_crop_source', 'focus_cell'} & context['workspace'].keys()
    assert future.cancelled() and native_future.cancelled()
    assert activate_view(context, ('gallery', 'queue', 0)) != token


def test_unavailable_target_round_trip_retires_pending_gallery_click(tmp_path, monkeypatch):
    from PIL import Image
    from dinoapp import app

    Image.new('RGB', (8, 8)).save(tmp_path / 'crop.png')
    (tmp_path / 'index.csv').write_text('cell,rank,image_path\nA|1,0,crop.png\n')
    path = tmp_path / 'labels.csv'
    path.write_text('cell,brdu,edu\nA|1,0,0\n')
    monkeypatch.setenv('DINO_REVIEW_SOURCE', 'Barrage')
    monkeypatch.setenv('DINO_REVIEW_BARRAGE_DIR', str(tmp_path))
    app.st.cache_resource.clear()
    review = AppTest.from_file(app.__file__).run(timeout=20)
    assert not review.exception
    next(widget for widget in review.selectbox if widget.label == "Review view").select("EdU correction gallery").run()
    assert not review.exception
    context, = review.session_state.review_contexts.values()
    old_token = context['event_token']
    queue_id = next(iter(context['galleries'].values()))['queue_id']
    directory = next(widget for widget in review.text_input if widget.label == 'Barrage directory')
    directory.set_value(str(tmp_path / 'missing')).run()
    assert not review.exception
    assert review.error
    next(widget for widget in review.text_input if widget.label == 'Barrage directory').set_value(str(tmp_path)).run()
    assert not review.exception
    before = path.read_bytes()
    send_gallery_flip(review, {
        'itemId': 'A|1', 'cell': 'A|1', 'expected': [0, 0], 'value': [0, 1], 'token': old_token,
    })
    assert not review.exception
    assert path.read_bytes() == before
    context, = review.session_state.review_contexts.values()
    assert context['history'] == []
    assert context['event_token'] != old_token
    assert next(iter(context['galleries'].values()))['queue_id'] == queue_id
