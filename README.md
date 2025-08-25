# GPX Overlay Maker (Track + Elevation + Stats)

## v5 – Track sizing controls
- **Fit width**: track expands to available width; height follows aspect
- **Fixed height (anchored with width)**: set exact track height in pixels; width fills container (aspect preserved)
- **Fixed width (anchored with height)**: set exact track width in pixels; height computed from aspect (capped to available area)

Other features:
- Units toggle (imperial ↔ metric): graph & info switch mi/ft vs km/m
- Run info font family/style/weight controls
- Elevation graph show/hide + label toggles
- Per-field info toggles (location, distance, elevation, time, temp)
- Transparent PNG export; size/DPI controls; titles with font-size

## Quick Start
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```