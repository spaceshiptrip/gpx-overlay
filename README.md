# GPX Overlay Maker (Track + Elevation + Stats)

Create a social-media friendly image from a GPX file.

## v4 – New
- **Units toggle:** Imperial ↔ Metric
  - Imperial: **miles** & **feet**
  - Metric: **kilometers** & **meters**
  - Graph axes and run info reflect the chosen unit system
- **Run info font controls:**
  - Choose **font family** (sans-serif/serif/monospace or any installed name)
  - Choose **font style** (normal/italic)
  - Choose **font weight** (normal/bold/300–900)

## Features
- 2D overhead track (Web Mercator projection)
- Optional elevation graph with label toggles
- Run info block with per-field show/label toggles (Location, Distance, Elevation, Time, Temp)
- Transparent PNG export for overlaying on videos/photos
- Title & Sub Title with show/hide and font-size controls
- Canvas size/DPI controls

## Quick Start
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```