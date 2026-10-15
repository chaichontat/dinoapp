import dino_markers


def test_every_public_name_is_exported():
    expected = {
        "RGB", "Label", "SavedFilter", "MAX_MARKERS", "CONFIG_FILENAME", "CONTRACT_FIELD", "RESERVED_KEYS",
        "Channel", "Marker", "BadgeColor", "Presentation", "MarkerContract", "MarkerConfig", "DEFAULT_CONTRACT",
        "parse_config", "load_config", "default_config", "config_to_json", "require_same_contract",
        "MarkerConfigError", "ContractMismatchError", "MissingContractError", "UnsupportedArtifactError",
        "ArtifactKind", "PREPARED_SOURCE", "BARRAGE", "FEATURE_CACHE", "ZARR_MODEL", "INTENSITY_MODEL",
        "JOINT_MODEL", "INFERENCE", "CORTEX_RESCORE", "SPATIAL_REVIEW", "MODEL_KINDS", "bundle_contract",
        "derive_bundle", "CONTRACT_FILES", "declared_contract", "contract_of", "resolve_directory",
        "save_config", "marker_revision", "CLASSIFIER_METRICS", "BARRAGE_STATS", "probability_column",
        "prediction_column", "oof_probability_column", "saved_column", "miscall_column",
        "rhu2_probability_column", "image_path_column", "cutoff_key", "model_key", "scaler_key",
        "metric_key", "positive_count_key", "stat_column", "intensity_column", "validate_label",
        "call_weights", "encode_call", "decode_call", "uncalled_code", "toggle_marker", "matches_saved",
        "state_label", "identity_name", "parse_color", "hex_color", "identity_color", "badge_colors",
    }
    assert expected <= set(dino_markers.__all__)
    for name in expected:
        assert hasattr(dino_markers, name)


def test_constants():
    assert dino_markers.CONFIG_FILENAME == "markers.json"
    assert dino_markers.CONTRACT_FIELD == "marker_contract"
