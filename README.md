# GPX Overlay Maker (Track + Elevation + Stats)

Create a social-media friendly image from a GPX file:
- 2D overhead track (projected to Web Mercator)
- Optional Elevation profile (with toggles for axis labels)
- Run information block with individually toggleable fields and per-field label prefixes
- Transparent PNG option for easy overlay on video/photos

## New in v3
- Show/Hide Elevation Graph
  - Show/Hide Graph Labels: Distance (km), Elevation (m)
- Show/Hide Run Information
  - Location (labels on/off)
  - Distance in miles (labels on/off)
  - Elevation gain in feet (labels on/off)
  - Time (labels on/off)
  - Temperature in °F (labels on/off)
- Show/Hide Title and Sub Title
- Font size controls for Title, Sub Title, Graph Axes, and Run Info

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Upload a `.gpx`, tweak options, and Download PNG.