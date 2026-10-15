"""RHU2 spatial review using the shared field picker and annotation writer."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from dinoapp import native, native_app
from dinoapp.rhu2 import FieldCropSource, SpatialSource
from dinoapp.sources import AnnotationFiles
from dinoapp.review_session import get_context, navigate, remember_widget, widget_key


@st.cache_resource
def get_source(root: str, manifest_mtime_ns: int) -> SpatialSource:
    return SpatialSource(Path(root))


def spatial_mode(root: Path, *, target_id: str) -> None:
    source = get_source(str(root), (root / 'spatial.json').stat().st_mtime_ns)
    context = get_context(target_id)
    annotations = AnnotationFiles((root / 'labels.csv',), root / 'labels.csv')
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
        if located_view:
            st.caption("Viewing a located cell. Return to sampled fields to change the branch cursor.")
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
        brdu_percentile = st.number_input('BrdU vmax percentile', min_value=0.01, max_value=100.0,
                                          step=0.1, format='%.2f', key=widget_key(context, 'rhu2_brdu_vmax_percentile', 99.8), on_change=remember_widget, args=(context, 'rhu2_brdu_vmax_percentile'),
                                          help='Upper display limit from BrdU pixels in the current field.')
        edu_percentile = st.number_input('EdU vmax percentile', min_value=0.01, max_value=100.0,
                                         step=0.1, format='%.2f', key=widget_key(context, 'rhu2_edu_vmax_percentile', 99.8), on_change=remember_widget, args=(context, 'rhu2_edu_vmax_percentile'),
                                         help='Upper display limit from EdU pixels in the current field.')
        st.caption(f'{len(branch_order)} cortical CCF branches, fewest manual annotations first. '
                   'Counts include your saved barrage and spatial annotations. '
                   'Next and Previous browse reproducible fields within the selected branch.')
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
        st.caption('Saved cortical annotations, including the migrated barrage. '
                   'Acquisition ROI counts combine its ccf_adjusted branches.')
    st.subheader(roi.display_name if context["workspace"]["located_field"] is not None else f"{roi.display_name} · {branch['ccf_adjusted']}")
    prefix = "" if located_view else f"{branch['cells']:,} cortical cohort cells in this branch · "
    st.caption(prefix + f"{int((field.ids > 0).sum()):,} in this field · 20 µm scale bar")
    st.caption('Only cortical cohort cells are colored and selectable. '
               'Cells outside the cohort remain visible in the raw image.')
    contrast = roi.contrast.copy()
    for index, marker, percentile in [(1, 'BrdU', brdu_percentile), (2, 'EdU', edu_percentile)]:
        # One raw intensity unit keeps empty fields well defined with vmin=0.
        contrast[index, 1] = max(1., float(np.percentile(field.images[marker], percentile)))
    context["workspace"]["active_field"] = (roi, field)
    context["workspace"]["field_crop_source"] = FieldCropSource(
        roi, tuple(float(value) for value in contrast[:, 1]), target_id, context["revision"],
    )
    image = native.render_field(
        field, roi.field_calls(field, annotations), mode=view, selected=None,
        border_alpha=border_alpha, boundary_alpha=0, contrast=contrast,
        uncertainty=roi.field_uncertainty(field), uncertainty_log_vmax=roi.uncertainty_log_vmax,
    )
    native_app.review_field(
        roi, field, roi.annotation_paths, root / 'cell-review', image, view,
        crop_plane_caption=lambda plane: f'largest-area slice · Z plane {plane}',
        selected_caption=roi.probability_caption, show_seed=False,
        render_cell_crop=lambda label: roi.cell_crop(label, tuple(float(value) for value in contrast[:, 1])),
    )
