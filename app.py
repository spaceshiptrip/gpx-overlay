import streamlit as st
from overlay import generate_overlay_image, OverlayOptions

st.set_page_config(page_title="GPX Overlay Maker", layout="wide")
st.title("GPX → Social Overlay Image (Track + Elevation + Stats)")

with st.sidebar:
    st.header("Overlay Options")
    title = st.text_input("Title", "My Run")
    subtitle = st.text_input("Subtitle", "Marathon Training")
    show_loc = st.checkbox("Show Location", value=True)
    custom_loc = st.text_input("Custom Location (optional, overrides lat/lon center)", "")
    show_temp = st.checkbox("Show Temperature", value=True)
    temp_f = st.number_input("Temperature (°F)", value=68, step=1)
    width = st.number_input("Image width (px)", value=1920, step=10)
    height = st.number_input("Image height (px)", value=1080, step=10)
    dpi = st.number_input("DPI", value=150, step=10)
    transparent = st.checkbox("Transparent background (for overlay)", value=True)
    grid = st.checkbox("Show elevation grid", value=False)

st.write("Upload a `.gpx` file. We'll draw a 2D overhead track, plot elevation, and stamp stats.")
uploaded = st.file_uploader("GPX File", type=["gpx"])

if uploaded is not None:
    opts = OverlayOptions(
        title=title,
        subtitle=subtitle,
        temperature_f=float(temp_f) if show_temp else None,
        show_temperature=show_temp,
        show_location=show_loc,
        custom_location=custom_loc if custom_loc.strip() else None,
        width_px=int(width),
        height_px=int(height),
        transparent_bg=transparent,
        dpi=int(dpi),
        grid=grid,
    )

    if st.button("Generate Overlay Image"):
        png_bytes = generate_overlay_image(uploaded.read(), opts)
        st.image(png_bytes, caption="Overlay Preview", use_column_width=True)
        st.download_button("⬇️ Download PNG", data=png_bytes, file_name="gpx_overlay.png", mime="image/png")

st.markdown("---")
st.caption("Tip: Use a transparent background and place this PNG on top of your video/photo in your editor.")