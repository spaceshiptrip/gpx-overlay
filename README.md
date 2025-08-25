# GPX Overlay Maker (Track + Elevation + Stats)

Create a social-media friendly image from a GPX file:
- 2D overhead track (projected to Web Mercator)
- Elevation profile
- Stats text: Distance (mi), Elevation Gain (ft), Duration, Location, Temperature (optional)
- Transparent PNG option for easy overlay on video/photos

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- Distance computed geodesically; elevation gain sums positive deltas.
- Duration uses first/last GPX timestamps when available.
- Location defaults to the midpoint of the GPX bounding box; override with your own label.
- Temperature is manual for now; can be automated later via weather API.