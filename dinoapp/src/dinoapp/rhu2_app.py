"""RHU2 spatial review using the shared field picker and annotation writer."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dino_markers import marker_revision

from dinoapp import native, native_app
from dinoapp.rhu2 import FieldCropSource, SpatialSource
from dinoapp.sources import AnnotationFiles
from dinoapp.review_session import get_context, navigate, remember_widget, widget_key


@st.cache_resource
def get_source(root: str, marker_revision: str) -> SpatialSource:
    return SpatialSource(Path(root))


def spatial_mode(root: Path, *, target_id: str) -> None:
    source = get_source(str(root), marker_revision((root,)))
    config = source.config
    context = get_context(target_id)
    annotations = AnnotationFiles((root / 'labels.csv',), root / 'labels.csv', contract=config.contract)
    annotated_counts = source.branch_annotation_counts(tuple(sorted(annotations.values)))
    branch_order = sorted(
        (index for index, branch in enumerate(source.branches) if branch['cells'] > 0),
        key=lambda index: (annotated_counts[index], source.branches[index]['key']),
    )
    if context['widgets'].get('rhu2_branch_index') not in branch_order:
        context['widgets']['rhu2_branch_index'] = branch_order[0]
        native_app.reset_position(context)
    located_view = context["workspace"]["located_field"] is not None
    with st.sidebar:
        st.header('CCF branch fields')
        branch_index = st.selectbox(
            'ccf_adjusted ROI', options=branch_order, disabled=located_view, key=widget_key(context, 'rhu2_branch_index', branch_order[0]),
            format_func=lambda index: (
                f"{source.branches[index]['dataset']} · ROI {source.branches[index]['roi']} · "
                f"{source.branches[index]['ccf_adjusted']} · {annotated_counts[index]:,} manual annotations"
            ), on_change=native_app.change_field_widget, args=(context, 'rhu2_branch_index'),
        )
        branch = source.branches[branch_index]
        position = int(st.number_input('Position', min_value=0, step=1, disabled=located_view, key=widget_key(context, 'position', 0), on_change=remember_widget, args=(context, 'position')))
        previous, following = st.columns(2)
        previous.button('Previous', on_click=native_app.move_position, args=(-1, context), key=f"rhu2-prev:{context['session_id']}",
                        disabled=located_view or position == 0, width='stretch')
        following.button('Next', disabled=located_view, on_click=native_app.move_position, args=(1, context), key=f"rhu2-next:{context['session_id']}", width='stretch')
        width = float(st.select_slider('Field width (µm)', disabled=located_view, options=[100, 150, 200, 250, 300], key=widget_key(context, 'field_width', 200), on_change=remember_widget, args=(context, 'field_width')))
        border_alpha = st.slider('Cell border opacity', 0.0, 1.0, step=0.05, key=widget_key(context, 'field_border_alpha', .35), on_change=remember_widget, args=(context, 'field_border_alpha'))
        view = st.selectbox('Right panel view', ('Cell identities', 'Model uncertainty'), key=widget_key(context, 'field_view', 'Cell identities'), on_change=remember_widget, args=(context, 'field_view'))
        percentiles = {
            marker: st.number_input(
                f'{config.label(marker)} vmax percentile', min_value=0.01, max_value=100.0,
                step=0.1, format='%.2f', key=widget_key(context, f'rhu2_{marker}_vmax_percentile', 99.8),
                on_change=remember_widget, args=(context, f'rhu2_{marker}_vmax_percentile'),
                help=f'Upper display limit from {config.label(marker)} pixels in the current field.',
            )
            for marker in config.contract.markers
        }
    if context["workspace"]["located_field"] is None:
        with st.spinner("Loading native ROI field…"):
            roi, cell_index = source.sample_branch(branch_index, position)
            field = roi.sampled_field(cell_index, branch_index, position, width)
    else:
        roi, field = context["workspace"]["located_field"]
        if st.button("Return to sampled fields"):
            context["workspace"]["located_field"] = None
            navigate(context, "Dataset cells", "Spatial field")
            st.rerun()
        if context["workspace"].get("crop_return") and st.button("Return to cell review"):
            navigate(context, *context["workspace"]["crop_return"])
            st.rerun()
    st.header('RHU2 spatial review')
    with st.expander('Manual annotation counts'):
        st.button('Refresh annotation counts')
        counts = pd.DataFrame([
            {'Sample': item['dataset'], 'ROI': item['roi'], 'Manual annotations': annotated_counts[index]}
            for index, item in enumerate(source.branches) if item['cells'] > 0
        ])
        samples, rois = st.tabs(['Samples', 'Acquisition ROIs'])
        with samples:
            st.dataframe(counts.groupby('Sample', as_index=False)['Manual annotations'].sum()
                         .sort_values(['Manual annotations', 'Sample']), hide_index=True)
        with rois:
            st.dataframe(counts.groupby(['Sample', 'ROI'], as_index=False)['Manual annotations'].sum()
                         .sort_values(['Manual annotations', 'Sample', 'ROI']), hide_index=True)
    st.subheader(roi.display_name if context["workspace"]["located_field"] is not None else f"{roi.display_name} · {branch['ccf_adjusted']}")
    prefix = "" if located_view else f"{branch['cells']:,} cortical cohort cells in this branch · "
    st.caption(prefix + f"{int((field.ids > 0).sum()):,} in this field · 20 µm scale bar")
    contrast = dict(roi.contrast)
    for marker, percentile in percentiles.items():
        # One raw intensity unit keeps empty fields well defined with vmin=0.
        contrast[marker] = (contrast[marker][0], max(1., float(np.percentile(field.images[marker], percentile))))
    display_max = tuple(contrast[key][1] for key in config.contract.channel_keys)
    context["workspace"]["active_field"] = (roi, field)
    context["workspace"]["field_crop_source"] = FieldCropSource(roi, display_max, target_id, context["revision"])
    image = native.render_field(
        field, roi.field_calls(field, annotations), config=config, mode=view, selected=None,
        border_alpha=border_alpha, boundary_alpha=0, contrast=contrast,
        uncertainty=roi.field_uncertainty(field), uncertainty_log_vmax=roi.uncertainty_log_vmax,
    )
    native_app.review_field(
        roi, field, roi.annotation_paths, root / 'cell-review', image, view,
        crop_plane_caption=lambda plane: f'largest-area slice · Z plane {plane}',
        selected_caption=roi.probability_caption, show_seed=False,
        render_cell_crop=lambda label: roi.cell_crop(label, display_max),
    )
