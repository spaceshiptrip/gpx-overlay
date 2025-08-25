import streamlit as st
from overlay import generate_overlay_image, OverlayOptions

st.set_page_config(page_title="GPX Overlay Maker", layout="wide")
st.title("GPX → Social Overlay Image (Track + Elevation + Stats)")

with st.sidebar:
    st.header("Overlay Options")

    # Units
    unit_system = st.radio("Units", options=["imperial", "metric"], index=0, horizontal=True)

    # Title / subtitle toggles
    show_title = st.checkbox("Show Title", value=True)
    title = st.text_input("Title", "My Run")
    title_fs = st.number_input("Title Font Size", value=48, step=1, min_value=8, max_value=200)

    show_subtitle = st.checkbox("Show Sub Title", value=True)
    subtitle = st.text_input("Sub Title", "Marathon Training")
    subtitle_fs = st.number_input("Sub Title Font Size", value=28, step=1, min_value=8, max_value=200)

    st.markdown("---")

    # Elevation graph
    show_elev_graph = st.checkbox("Show Elevation Graph", value=True)
    # Labels reflect current unit selection
    show_graph_label_distance = st.checkbox(f"Show Graph Label: Distance ({'mi' if unit_system=='imperial' else 'km'})", value=True)
    show_graph_label_elevation = st.checkbox(f"Show Graph Label: Elevation ({'ft' if unit_system=='imperial' else 'm'})", value=True)
    axes_fs = st.number_input("Graph Axes Font Size", value=14, step=1, min_value=6, max_value=72)
    grid = st.checkbox("Show elevation grid", value=False)

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

        # graph
        show_elev_graph=show_elev_graph,
        show_graph_label_distance=show_graph_label_distance,
        show_graph_label_elevation=show_graph_label_elevation,
        grid=grid,

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
        st.image(png_bytes, caption="Overlay Preview", use_column_width=True)
        st.download_button("⬇️ Download PNG", data=png_bytes, file_name="gpx_overlay.png", mime="image/png")

st.markdown("---")
st.caption("Tip: Use a transparent background and place this PNG on top of your video/photo in your editor.")