import streamlit as st
from overlay import generate_overlay_image, OverlayOptions

st.set_page_config(page_title="GPX Overlay Maker", layout="wide")
st.title("GPX → Social Overlay Image (Track + Elevation + Stats)")

with st.sidebar:
    st.header("Overlay Options")

    # Units
    unit_system = st.radio("Units", options=["imperial", "metric"], index=0, horizontal=True)

    # Title / subtitle
    show_title = st.checkbox("Show Title", value=True)
    title = st.text_input("Title", "My Run")
    title_fs = st.number_input("Title Font Size", value=48, step=1, min_value=8, max_value=200)

    show_subtitle = st.checkbox("Show Sub Title", value=True)
    subtitle = st.text_input("Sub Title", "Marathon Training")
    subtitle_fs = st.number_input("Sub Title Font Size", value=28, step=1, min_value=8, max_value=200)

    st.markdown("---")

    # Track sizing
    sizing_mode = st.radio("Track sizing", options=["fit_width", "fixed_height", "fixed_width"], index=0, horizontal=True)
    track_height_px = st.number_input("Track height (px) [if fixed_height]", value=400, min_value=50, max_value=5000, step=10)
    track_width_px = st.number_input("Track width (px) [if fixed_width]", value=1200, min_value=50, max_value=10000, step=10)

    st.markdown("---")

    # Elevation graph
    show_elev_graph = st.checkbox("Show Elevation Graph", value=True)
    elev_sizing_mode = st.radio("Elevation sizing", options=["fit_track_width", "fixed_height", "fixed_width"], index=0, horizontal=True)
    elev_height_px = st.number_input("Elevation height (px) [if fixed_height]", value=240, min_value=50, max_value=5000, step=10)
    elev_width_px = st.number_input("Elevation width (px) [if fixed_width]", value=1200, min_value=50, max_value=10000, step=10)

    show_graph_label_x = st.checkbox("Show X label", value=True)
    show_graph_label_y = st.checkbox("Show Y label", value=True)
    show_x_ticks = st.checkbox("Show X ticks", value=True)
    show_y_ticks = st.checkbox("Show Y ticks", value=True)
    show_x_ticklabels = st.checkbox("Show X tick labels", value=True)
    show_y_ticklabels = st.checkbox("Show Y tick labels", value=True)
    show_graph_axes_lines = st.checkbox("Show axes lines", value=True)
    axes_fs = st.number_input("Graph Axes Font Size", value=14, step=1, min_value=6, max_value=72)
    grid = st.checkbox("Show elevation grid", value=False)

    # Peak options
    st.subheader("Peak options")
    show_peak = st.checkbox("Show peak elevation", value=True)
    show_peak_marker = st.checkbox("Show peak marker", value=True)
    show_peak_text = st.checkbox("Show peak text", value=True)

    st.markdown("---")

    # Run info panel
    show_run_info = st.checkbox("Show Run Information", value=True)
    info_fs = st.number_input("Run Info Font Size", value=24, step=1, min_value=6, max_value=120)

    # Run info font family/style/weight
    family_choice = st.selectbox("Run Info Font Family", options=["sans-serif", "serif", "monospace"], index=0)
    style_choice = st.selectbox("Run Info Font Style", options=["normal", "italic"], index=0)
    weight_choice = st.selectbox("Run Info Font Weight", options=["normal", "bold", "300", "400", "500", "600", "700", "800", "900"], index=1)

    show_location = st.checkbox("Show: Location", value=True)
    label_location = st.checkbox("Label: Location", value=True)

    show_distance = st.checkbox(f"Show: Distance ({'miles' if unit_system=='imperial' else 'kilometers'})", value=True)
    label_distance = st.checkbox("Label: Distance", value=True)

    show_elev_gain = st.checkbox(f"Show: Elevation ({'feet' if unit_system=='imperial' else 'meters'})", value=True)
    label_elev_gain = st.checkbox("Label: Elevation", value=True)

    show_time = st.checkbox("Show: Time", value=True)
    label_time = st.checkbox("Label: Time", value=True)

    show_temperature = st.checkbox("Show: Temp (°F)", value=True)
    label_temperature = st.checkbox("Label: Temp", value=True)
    temp_f = st.number_input("Temperature (°F, optional)", value=68, step=1)

    st.markdown("---")

    # Canvas / style
    width = st.number_input("Image width (px)", value=1920, step=10, min_value=320, max_value=10000)
    height = st.number_input("Image height (px)", value=1080, step=10, min_value=320, max_value=10000)
    dpi = st.number_input("DPI", value=150, step=10, min_value=72, max_value=600)
    transparent = st.checkbox("Transparent background (for overlay)", value=True)

st.write("Upload a `.gpx` file. We'll draw a 2D overhead track, optionally plot elevation, and stamp stats.")
uploaded = st.file_uploader("GPX File", type=["gpx"])

if uploaded is not None:
    opts = OverlayOptions(
        # units
        unit_system=unit_system,

        # titles
        title=title,
        subtitle=subtitle,
        show_title=show_title,
        show_subtitle=show_subtitle,

        # track sizing
        track_sizing_mode=sizing_mode,
        track_height_px=int(track_height_px),
        track_width_px=int(track_width_px),

        # elevation sizing
        elev_sizing_mode=elev_sizing_mode,
        elev_height_px=int(elev_height_px),
        elev_width_px=int(elev_width_px),

        # graph
        show_elev_graph=show_elev_graph,
        show_graph_label_x=show_graph_label_x,
        show_graph_label_y=show_graph_label_y,
        show_x_ticks=show_x_ticks,
        show_y_ticks=show_y_ticks,
        show_x_ticklabels=show_x_ticklabels,
        show_y_ticklabels=show_y_ticklabels,
        show_graph_axes_lines=show_graph_axes_lines,
        grid=grid,

        # peak
        show_peak=show_peak,
        show_peak_marker=show_peak_marker,
        show_peak_text=show_peak_text,

        # run info block
        show_run_info=show_run_info,

        # fields + labels
        show_location=show_location,
        label_location=label_location,

        show_distance=show_distance,
        label_distance=label_distance,

        show_elev_gain=show_elev_gain,
        label_elev_gain=label_elev_gain,

        show_time=show_time,
        label_time=label_time,

        show_temperature=show_temperature,
        label_temperature=label_temperature,
        temperature_f=float(temp_f) if show_temperature else None,

        # canvas
        width_px=int(width),
        height_px=int(height),
        transparent_bg=transparent,
        dpi=int(dpi),

        # fonts
        title_fontsize=int(title_fs),
        subtitle_fontsize=int(subtitle_fs),
        axes_fontsize=int(axes_fs),
        info_fontsize=int(info_fs),
        info_fontfamily=family_choice,
        info_fontstyle=style_choice,
        info_fontweight=weight_choice,
    )

    if st.button("Generate Overlay Image"):
        png_bytes = generate_overlay_image(uploaded.read(), opts)
        st.image(png_bytes, caption="Overlay Preview", use_container_width=True)
        st.download_button("⬇️ Download PNG", data=png_bytes, file_name="gpx_overlay.png", mime="image/png")

st.markdown("---")
st.caption("Tip: Use a transparent background and place this PNG on top of your video/photo in your editor.")