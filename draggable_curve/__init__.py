"""
Draggable curve -- a custom Streamlit component that lets a user click and
drag any point on a line chart to adjust it, with nearby points shifting
too (linear falloff within `radius` slots), rather than creating a single
isolated spike.

Setup, first time only:
    cd draggable_curve/frontend
    npm install
    npm run build          # produces frontend/dist, which _RELEASE=True reads

While actively developing the frontend, set _RELEASE = False below and run
`npm run dev` in a separate terminal instead of `npm run build` -- this
points the component at the Vite dev server so changes hot-reload.
"""

import os
import streamlit.components.v1 as components

_RELEASE = True

if not _RELEASE:
    _component_func = components.declare_component(
        "draggable_curve",
        url="http://localhost:5173",  # Vite's default dev server port
    )
else:
    _parent_dir = os.path.dirname(os.path.abspath(__file__))
    _build_dir = os.path.join(_parent_dir, "frontend", "dist")
    _component_func = components.declare_component("draggable_curve", path=_build_dir)


def draggable_curve(values, labels=None, radius=8, height=360, key=None):
    """Interactive draggable curve. Drag any point to adjust it; every point
    within `radius` slots on either side shifts too, by a linearly-decaying
    fraction of the same delta, so dragging deforms a smooth stretch of the
    curve instead of creating a single-slot spike.

    Parameters
    ----------
    values : list[float]
        The current values to display and edit, e.g. 96 forecast values.
    labels : list[str], optional
        X-axis labels, one per value (e.g. "00:00", "00:15", ...). Defaults
        to plain integer indices if not provided.
    radius : int
        How many slots on each side of the dragged point are affected.
        Points more than `radius` slots away are never touched by a drag.
    height : int
        Chart height in pixels.
    key : str, optional
        Streamlit widget key, for when multiple instances appear on one page.

    Returns
    -------
    list[float]
        The values as of the most recent drag-release. Identical to the
        input `values` if nothing has been dragged yet in this session --
        check for that if you need to distinguish "untouched" from "dragged
        back to the original value".
    """
    values = [float(v) for v in values]
    if labels is None:
        labels = [str(i) for i in range(len(values))]

    component_value = _component_func(
        values=values,
        labels=list(labels),
        radius=int(radius),
        height=int(height),
        key=key,
        default=values,
    )
    return component_value
