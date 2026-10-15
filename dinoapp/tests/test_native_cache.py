"""Cache ownership across Streamlit module reloads."""

import sys
from types import ModuleType

from dinoapp.native_app import get_field, get_source
from dinoapp import native_app
from dinoapp.review_session import activate_target


def test_switching_native_dataset_isolates_edit_state(monkeypatch):
    state = {}
    monkeypatch.setattr(native_app.st, "session_state", state)
    activate_target("prod2", revision="prod2")
    native_app.get_active_context()["native"]["selection"] = {"label": 12}
    state["review_contexts"]["prod2"]["history"] = ["prod2 edit"]
    native_app.get_active_context()["widgets"]["position"] = 4
    activate_target("prod2", revision="prod2")
    assert state["review_contexts"]["prod2"]["history"] == ["prod2 edit"]
    assert native_app.get_active_context()["widgets"]["position"] == 4

    activate_target("prod3", revision="prod3")
    assert native_app.get_active_context()["native"]["selection"] is None
    assert state["review_contexts"]["prod3"]["history"] == []
    assert state["review_contexts"]["prod2"]["history"] == ["prod2 edit"]
    assert native_app.get_active_context()["widgets"]["position"] == 0
    assert native_app.get_active_context()["native"]["gallery_states"] == {}


def test_sidebar_cache_miss_uses_reloaded_source_class(monkeypatch):
    def module_version():
        module = ModuleType("review_reload")
        exec(
            "class Field: pass\n"
            "class Source:\n"
            "    def field(self, *args):\n"
            "        result = Field()\n"
            "        result.settings = args\n"
            "        return result\n",
            module.__dict__,
        )
        monkeypatch.setitem(sys.modules, module.__name__, module)
        return module

    get_source.clear()
    get_field.clear()
    first = module_version()
    initial = get_field(first.Source, 0, 0, 694.7, 200.0)
    assert isinstance(initial, first.Field)
    second = module_version()
    changed = get_field(second.Source, 1, 0, 695.3, 150.0)
    assert isinstance(changed, second.Field)
    assert changed.settings == (1, 0, 695.3, 150.0)
    assert get_source(first.Source) is not get_source(second.Source)
    cached = get_field(second.Source, 1, 0, 695.3, 150.0)
    assert cached.settings == changed.settings
    assert cached is not changed
    get_source.clear()
    get_field.clear()


def test_returning_to_native_target_restores_field_and_queue_state(monkeypatch):
    state = {}
    monkeypatch.setattr(native_app.st, 'session_state', state)
    activate_target('A', revision='A')
    native_app.get_active_context()['widgets']['position'] = 7
    native_app.get_active_context()['widgets']['seed'] = 19
    native_app.get_active_context()['native']['selection'] = {'label': 12, 'expected': (1, 0)}
    native_app.get_active_context()['native']['gallery_states'] = {'queue': {'prefetches': {}, 'page': 3}}
    activate_target('B', revision='B')
    assert native_app.get_active_context()['native']['selection'] is None
    assert native_app.get_active_context()['native']['gallery_states'] == {}
    activate_target('A', revision='A')
    assert native_app.get_active_context()['widgets']['position'] == 7
    assert native_app.get_active_context()['widgets']['seed'] == 19
    assert native_app.get_active_context()['native']['selection'] == {'label': 12, 'expected': (1, 0)}
    assert native_app.get_active_context()['native']['gallery_states']['queue']['page'] == 3
