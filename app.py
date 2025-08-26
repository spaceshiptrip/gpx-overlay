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

    # Peak options (elevation graph)
    st.subheader("Peak options (elevation graph)")
    show_peak = st.checkbox("Show peak elevation (elev graph)", value=True)
    show_peak_marker = st.checkbox("Show peak marker (elev graph)", value=True)
    show_peak_text = st.checkbox("Show peak text (elev graph)", value=True)
    peak_marker_size_elev = st.number_input("Peak marker size (elev)", value=6.0, min_value=1.0, max_value=40.0, step=0.5)
    peak_marker_color_elev = st.color_picker("Peak marker color (elev)", value="#000000")

    st.markdown("---")

    # Track peak options (GPX map)
    st.subheader("Peak options (GPX track)")
    show_track_peak = st.checkbox("Show peak elevation (track)", value=True)
    show_track_peak_marker = st.checkbox("Show peak marker (track)", value=True)
    show_track_peak_text = st.checkbox("Show peak text (track)", value=True)
    peak_marker_size_track = st.number_input("Peak marker size (track)", value=6.0, min_value=1.0, max_value=40.0, step=0.5)
    peak_marker_color_track = st.color_picker("Peak marker color (track)", value="#000000")

    st.markdown("---")

    # Start/Finish arrows
    st.subheader("Start / Finish Arrows")
    show_start_finish_track = st.checkbox("Show on GPX track", value=True)
    show_start_finish_elev = st.checkbox("Show on elevation", value=True)
    arrow_color_track = st.color_picker("Arrow color (track)", value="#000000")
    arrow_color_elev = st.color_picker("Arrow color (elev)", value="#000000")

    st.markdown("---")

    # End mileage
    st.subheader("End Mileage Marker/Label")
    show_end_mileage_track = st.checkbox("Show end mileage on track", value=True)
    show_end_mileage_elev = st.checkbox("Show end mileage on elev", value=True)
    end_marker_style = st.selectbox("End marker style", options=["o","s","^","v","D","X"], index=0)
    end_marker_size = st.number_input("End marker size", value=7.0, min_value=1.0, max_value=40.0, step=0.5)
    end_marker_color = st.color_picker("End marker color", value="#000000")
    end_label_color = st.color_picker("End label color", value="#000000")

    st.markdown("---")

    # Mile markers
    st.subheader("Mile/KM Markers")
    mile_markers_track = st.selectbox("On GPX track", options=["none","ticks","ticks+labels"], index=0)
    mile_marker_size_track = st.number_input("Track marker size", value=4.0, min_value=1.0, max_value=20.0, step=0.5)
    mile_marker_color_track = st.color_picker("Track marker color", value="#000000")

    mile_markers_elev = st.selectbox("On elevation", options=["none","ticks","ticks+labels"], index=0)
    mile_tick_label_rotation = st.number_input("Elev label rotation (deg)", value=0.0, step=5.0)

    st.markdown("---")

    # Line styles
    st.subheader("Line styles")
    line_width_track = st.number_input("Track line width", value=3.0, min_value=0.1, max_value=20.0, step=0.1)
    color_track = st.color_picker("Track line color", value="#000000")
    style_track = st.selectbox("Track line style", options=["solid", "dashed", "dotted"], index=0)
    capstyle_track = st.selectbox("Track cap style", options=["butt", "round", "projecting"], index=1)
    joinstyle_track = st.selectbox("Track join style", options=["miter", "round", "bevel"], index=1)

    line_width_elev = st.number_input("Elevation line width", value=2.0, min_value=0.1, max_value=20.0, step=0.1)
    color_elev = st.color_picker("Elevation line color", value="#1f77b4")
    style_elev = st.selectbox("Elevation line style", options=["solid", "dashed", "dotted"], index=0)
    capstyle_elev = st.selectbox("Elevation cap style", options=["butt", "round", "projecting"], index=1)
    joinstyle_elev = st.selectbox("Elevation join style", options=["miter", "round", "bevel"], index=1)

    # Glow styles
    st.subheader("Glow (stroke)")
    show_glow_track = st.checkbox("Glow on track line", value=False)
    glow_color_track = st.color_picker("Glow color (track)", value="#FFFFFF")
    glow_width_track = st.number_input("Glow width (track)", value=6.0, min_value=0.0, max_value=40.0, step=0.5)

    show_glow_elev = st.checkbox("Glow on elevation line", value=False)
    glow_color_elev = st.color_picker("Glow color (elevation)", value="#FFFFFF")
    glow_width_elev = st.number_input("Glow width (elevation)", value=6.0, min_value=0.0, max_value=40.0, step=0.5)

    # Shadow styles
    st.subheader("Drop Shadow")
    show_shadow_track = st.checkbox("Shadow on track line", value=False)
    shadow_color_track = st.color_picker("Shadow color (track)", value="#000000")
    shadow_alpha_track = st.slider("Shadow alpha (track)", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
    shadow_dx_track = st.number_input("Shadow offset X (px, track)", value=2.0, step=0.5)
    shadow_dy_track = st.number_input("Shadow offset Y (px, track)", value=-2.0, step=0.5)

    show_shadow_elev = st.checkbox("Shadow on elevation line", value=False)
    shadow_color_elev = st.color_picker("Shadow color (elevation)", value="#000000")
    shadow_alpha_elev = st.slider("Shadow alpha (elevation)", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
    shadow_dx_elev = st.number_input("Shadow offset X (px, elev)", value=2.0, step=0.5)
    shadow_dy_elev = st.number_input("Shadow offset Y (px, elev)", value=-2.0, step=0.5)

    st.markdown("---")

    # Color gradients
    st.subheader("Color Gradient")
    gradient_track = st.selectbox("Track color by", options=["none","speed","elevation"], index=0)
    cmap_track = st.selectbox("Track colormap", options=["viridis","plasma","magma","inferno","turbo","cool","hot","jet"], index=0)
    gradient_elev = st.selectbox("Elevation color by", options=["none","speed","elevation"], index=0)
    cmap_elev = st.selectbox("Elevation colormap", options=["viridis","plasma","magma","inferno","turbo","cool","hot","jet"], index=0)

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

        # peak (elevation graph)
        show_peak=show_peak,
        show_peak_marker=show_peak_marker,
        show_peak_text=show_peak_text,
        peak_marker_size_elev=float(peak_marker_size_elev),
        peak_marker_color_elev=peak_marker_color_elev,

        # track peak (gpx map)
        show_track_peak=show_track_peak,
        show_track_peak_marker=show_track_peak_marker,
        show_track_peak_text=show_track_peak_text,
        peak_marker_size_track=float(peak_marker_size_track),
        peak_marker_color_track=peak_marker_color_track,

        # start/finish arrows
        show_start_finish_track=show_start_finish_track,
        show_start_finish_elev=show_start_finish_elev,
        arrow_color_track=arrow_color_track,
        arrow_color_elev=arrow_color_elev,

        # end mileage
        show_end_mileage_track=show_end_mileage_track,
        show_end_mileage_elev=show_end_mileage_elev,
        end_marker_style=end_marker_style,
        end_marker_size=float(end_marker_size),
        end_marker_color=end_marker_color,
        end_label_color=end_label_color,

        # mile markers
        mile_markers_track=mile_markers_track,
        mile_marker_size_track=float(mile_marker_size_track),
        mile_marker_color_track=mile_marker_color_track,
        mile_markers_elev=mile_markers_elev,
        mile_tick_label_rotation=float(mile_tick_label_rotation),

        # line styles
        line_width_track=float(line_width_track),
        color_track=color_track,
        style_track=style_track,
        capstyle_track=capstyle_track,
        joinstyle_track=joinstyle_track,

        line_width_elev=float(line_width_elev),
        color_elev=color_elev,
        style_elev=style_elev,
        capstyle_elev=capstyle_elev,
        joinstyle_elev=joinstyle_elev,

        # glows
        show_glow_track=show_glow_track,
        glow_color_track=glow_color_track,
        glow_width_track=float(glow_width_track),
        show_glow_elev=show_glow_elev,
        glow_color_elev=glow_color_elev,
        glow_width_elev=float(glow_width_elev),

        # shadows
        show_shadow_track=show_shadow_track,
        shadow_color_track=shadow_color_track,
        shadow_alpha_track=float(shadow_alpha_track),
        shadow_dx_track=float(shadow_dx_track),
        shadow_dy_track=float(shadow_dy_track),

        show_shadow_elev=show_shadow_elev,
        shadow_color_elev=shadow_color_elev,
        shadow_alpha_elev=float(shadow_alpha_elev),
        shadow_dx_elev=float(shadow_dx_elev),
        shadow_dy_elev=float(shadow_dy_elev),

        # gradients
        gradient_track=gradient_track,
        gradient_elev=gradient_elev,
        cmap_track=cmap_track,
        cmap_elev=cmap_elev,

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