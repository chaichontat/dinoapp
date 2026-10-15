import dino_markers


def test_every_exported_name_resolves():
    for name in dino_markers.__all__:
        assert hasattr(dino_markers, name)
