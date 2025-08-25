# GPX Overlay Maker (Track + Elevation + Stats)

## v6 – Elevation plot controls
- **Sizing modes:**
  - *Fit track width* (same width as GPX plot)
  - *Fixed height* (px; width = track width)
  - *Fixed width* (px; height auto)
- **Labels:**
  - Show/Hide **X** label
  - Show/Hide **Y** label
  - Show/Hide **axes lines** (spines)
- **Peak features:**
  - Show/Hide peak elevation
  - Show/Hide peak marker
  - Show/Hide peak text

Everything else from v5:
- Units toggle (imperial ↔ metric)
- Track sizing modes
- Run info font family/style/weight + sizes
- Per-field info toggles
- Titles/subtitle controls, transparent PNG, DPI, canvas size

## Quick Start
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```