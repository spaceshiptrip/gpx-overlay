import io
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Tuple

import gpxpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from pyproj import Geod, Transformer


@dataclass
class OverlayOptions:
    # Units: "imperial" or "metric"
    unit_system: str = "imperial"

    # Titles
    title: str = "My Run"
    subtitle: str = "Marathon Training"
    show_title: bool = True
    show_subtitle: bool = True

    # Graph visibility / labels
    show_elev_graph: bool = True
    show_graph_label_distance: bool = True   # X-axis label
    show_graph_label_elevation: bool = True  # Y-axis label
    grid: bool = False

    # Track sizing modes
    # 'fit_width' -> fills available width; height from aspect
    # 'fixed_height' -> user sets track_height_px; width fills container
    # 'fixed_width' -> user sets track_width_px; height from aspect
    track_sizing_mode: str = "fit_width"
    track_height_px: int = 400
    track_width_px: int = 1200
    track_margin_px: int = 20  # inner padding inside the track axes (visual)

    # Run info block visibility
    show_run_info: bool = True

    # Individual run info fields + label toggles
    show_location: bool = True
    label_location: bool = True

    show_distance: bool = True
    label_distance: bool = True  # "Dist " prefix

    show_elev_gain: bool = True
    label_elev_gain: bool = True  # "Gain " prefix

    show_time: bool = True
    label_time: bool = True  # "Time " prefix

    show_temperature: bool = True
    label_temperature: bool = True  # "Temp " prefix
    temperature_f: Optional[float] = None

    # Canvas / style
    width_px: int = 1920
    height_px: int = 1080
    transparent_bg: bool = True
    dpi: int = 150

    # Fonts
    title_fontsize: int = 48
    subtitle_fontsize: int = 28
    axes_fontsize: int = 14
    info_fontsize: int = 24  # run info text (footer)

    # Run info font customization
    info_fontfamily: str = "sans-serif"     # 'sans-serif' | 'serif' | 'monospace' | custom-installed name
    info_fontstyle: str = "normal"          # 'normal' | 'italic'
    info_fontweight: str = "bold"           # 'normal' | 'bold' | numeric string like '600'

    # Lines
    line_width_track: float = 3.0
    line_width_elev: float = 2.0


@dataclass
class GPXStats:
    distance_km: float
    elev_gain_m: float
    duration: dt.timedelta
    start_time: Optional[dt.datetime]
    end_time: Optional[dt.datetime]
    center_lat: float
    center_lon: float
    bbox: Tuple[float, float, float, float]


def _parse_gpx(gpx_bytes: bytes):
    text = gpx_bytes.decode("utf-8", errors="ignore")
    gpx = gpxpy.parse(io.StringIO(text))
    lats, lons, elevs, times = [], [], [], []
    for track in gpx.tracks:
        for seg in track.segments:
            for p in seg.points:
                lats.append(p.latitude)
                lons.append(p.longitude)
                elevs.append(p.elevation if p.elevation is not None else np.nan)
                times.append(p.time if p.time else None)
    if not lats:
        for rte in gpx.routes:
            for p in rte.points:
                lats.append(p.latitude)
                lons.append(p.longitude)
                elevs.append(p.elevation if p.elevation is not None else np.nan)
                times.append(None)
    return np.array(lats), np.array(lons), np.array(elevs), times


def _compute_stats(lats, lons, elevs, times) -> GPXStats:
    geod = Geod(ellps='WGS84')
    distance_m = 0.0
    elev_gain_m = 0.0

    start_time = next((t for t in times if t is not None), None)
    end_time = next((t for t in reversed(times) if t is not None), None)
    duration = (end_time - start_time) if (start_time and end_time) else dt.timedelta(0)

    for i in range(1, len(lats)):
        if np.isnan(lats[i-1]) or np.isnan(lats[i]):
            continue
        _, _, d = geod.inv(lons[i-1], lats[i-1], lons[i], lats[i])
        distance_m += d
        if not (np.isnan(elevs[i-1]) or np.isnan(elevs[i])):
            diff = elevs[i] - elevs[i-1]
            if diff > 0:
                elev_gain_m += diff

    min_lat, max_lat = np.nanmin(lats), np.nanmax(lats)
    min_lon, max_lon = np.nanmin(lons), np.nanmax(lons)
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0

    return GPXStats(
        distance_km=distance_m / 1000.0,
        elev_gain_m=float(elev_gain_m),
        duration=duration,
        start_time=start_time,
        end_time=end_time,
        center_lat=center_lat,
        center_lon=center_lon,
        bbox=(min_lat, min_lon, max_lat, max_lon)
    )


def _project_to_mercator(lats, lons):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = transformer.transform(lons, lats)
    return np.array(x), np.array(y)


def _format_duration(td: dt.timedelta) -> str:
    total_seconds = int(td.total_seconds())
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    else:
        return f"{m}:{s:02d}"


def _feet(m): return m * 3.28084
def _miles(km): return km * 0.621371


def generate_overlay_image(gpx_bytes: bytes, options: OverlayOptions) -> bytes:
    lats, lons, elevs, times = _parse_gpx(gpx_bytes)
    if len(lats) < 2:
        raise ValueError("GPX contains insufficient points.")

    stats = _compute_stats(lats, lons, elevs, times)
    x, y = _project_to_mercator(lats, lons)

    # Distance array for elevation profile
    geod = Geod(ellps='WGS84')
    dists_m = [0.0]
    for i in range(1, len(lats)):
        _, _, d = geod.inv(lons[i-1], lats[i-1], lons[i], lats[i])
        dists_m.append(dists_m[-1] + d)
    dists_km = np.array(dists_m) / 1000.0

    # Units
    imperial = options.unit_system.lower().startswith("imp")
    # For plotting
    dist_series = dists_km * (0.621371 if imperial else 1.0)
    elev_series = elevs * (3.28084 if imperial else 1.0)
    dist_label = "Distance (mi)" if imperial else "Distance (km)"
    elev_label = "Elevation (ft)" if imperial else "Elevation (m)"

    # Figure
    dpi = options.dpi
    fig_w = options.width_px / dpi
    fig_h = options.height_px / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_alpha(0.0 if options.transparent_bg else 1.0)

    # --- TITLE AXES ---
    title_h_frac = 0.10 if (options.show_title or (options.show_subtitle and options.subtitle)) else 0.05
    title_ax = fig.add_axes([0.04, 1 - title_h_frac, 0.92, title_h_frac])
    title_ax.axis("off")
    y_cursor = 0.62
    if options.show_title:
        t = title_ax.text(0.01, y_cursor, options.title or "Activity",
                          fontsize=options.title_fontsize, weight="bold", va="center", ha="left")
        t.set_path_effects([pe.withStroke(linewidth=2, foreground='black')])
        y_cursor -= 0.45
    if options.show_subtitle and options.subtitle:
        s = title_ax.text(0.01, y_cursor, options.subtitle,
                          fontsize=options.subtitle_fontsize, va="center", ha="left")
        s.set_path_effects([pe.withStroke(linewidth=2, foreground='black')])

    # --- ELEV AXES (optional) ---
    elev_h_frac = 0.22 if options.show_elev_graph else 0.0
    elev_ax = None
    if options.show_elev_graph:
        elev_ax = fig.add_axes([0.06, 0.06, 0.88, elev_h_frac - 0.02])

    # --- TRACK AXES position logic ---
    # Available vertical space for the track between title bottom and elevation top
    top_y = 1 - title_h_frac - 0.02
    bottom_y = (0.06 + elev_h_frac) if options.show_elev_graph else 0.06
    avail_h_frac = max(0.12, top_y - bottom_y)
    avail_w_frac = 0.92

    # Data aspect = height_units / width_units
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    y_min, y_max = np.nanmin(y), np.nanmax(y)
    if x_max - x_min < 1e-6: x_max += 1.0
    if y_max - y_min < 1e-6: y_max += 1.0
    data_aspect = (y_max - y_min) / (x_max - x_min)

    # Default: fit width
    track_w_frac = avail_w_frac
    track_h_frac = min(avail_h_frac, data_aspect * track_w_frac * (fig_w / fig_h))

    if options.track_sizing_mode == "fixed_height":
        track_h_frac = min(avail_h_frac, options.track_height_px / options.height_px)
        needed_w = track_h_frac / (data_aspect * (fig_w / fig_h))
        track_w_frac = min(avail_w_frac, needed_w)
    elif options.track_sizing_mode == "fixed_width":
        track_w_frac = min(avail_w_frac, options.track_width_px / options.width_px)
        needed_h = data_aspect * track_w_frac * (fig_w / fig_h)
        track_h_frac = min(avail_h_frac, needed_h)

    # Center track axes in available area
    x0 = (1 - track_w_frac) / 2
    y0 = bottom_y + (avail_h_frac - track_h_frac) / 2
    track_ax = fig.add_axes([x0, y0, track_w_frac, track_h_frac])
    track_ax.set_axis_off()

    # pad by 5% of data span
    xpad = (x_max - x_min) * 0.05
    ypad = (y_max - y_min) * 0.05
    track_ax.set_xlim(x_min - xpad, x_max + xpad)
    track_ax.set_ylim(y_min - ypad, y_max + ypad)
    track_ax.set_aspect('equal', adjustable='box')
    track_ax.plot(x, y, linewidth=options.line_width_track)

    # Draw elevation plot
    if options.show_elev_graph and elev_ax is not None:
        elev_mask = ~np.isnan(elevs)
        elev_ax.plot(dist_series[elev_mask], elev_series[elev_mask], linewidth=options.line_width_elev)
        if options.grid:
            elev_ax.grid(True, alpha=0.3)
        if options.show_graph_label_distance:
            elev_ax.set_xlabel(dist_label, fontsize=options.axes_fontsize)
        else:
            elev_ax.set_xlabel("")
        if options.show_graph_label_elevation:
            elev_ax.set_ylabel(elev_label, fontsize=options.axes_fontsize)
        else:
            elev_ax.set_ylabel("")
        elev_ax.tick_params(axis='both', labelsize=options.axes_fontsize)

    # Footer / run info (optional)
    if options.show_run_info:
        left = ""
        if options.show_location:
            loc_val = f"{stats.center_lat:.4f}, {stats.center_lon:.4f}"
            left = f"Location: {loc_val}" if options.label_location else loc_val

        parts = []
        if options.show_distance:
            miles_val = _miles(stats.distance_km)
            km_val = stats.distance_km
            parts.append(f"Dist {miles_val:.2f} mi" if imperial and options.label_distance else
                         (f"{miles_val:.2f} mi" if imperial else
                          (f"Dist {km_val:.2f} km" if options.label_distance else f"{km_val:.2f} km")))
        if options.show_elev_gain:
            ft_val = _feet(stats.elev_gain_m)
            m_val = stats.elev_gain_m
            parts.append(f"Gain {ft_val:.0f} ft" if imperial and options.label_elev_gain else
                         (f"{ft_val:.0f} ft" if imperial else
                          (f"Gain {m_val:.0f} m" if options.label_elev_gain else f"{m_val:.0f} m")))
        if options.show_time:
            dur = _format_duration(stats.duration)
            parts.append(f"Time {dur}" if options.label_time else f"{dur}")
        if options.show_temperature and options.temperature_f is not None:
            parts.append(f"Temp {options.temperature_f:.0f}°F" if options.label_temperature else f"{options.temperature_f:.0f}°F")

        right = " • ".join(parts)

        anchor_ax = elev_ax if (options.show_elev_graph and elev_ax is not None) else track_ax
        text_kwargs = dict(
            transform=anchor_ax.transAxes,
            va="bottom",
            fontsize=options.info_fontsize,
            fontfamily=options.info_fontfamily,
            fontstyle=options.info_fontstyle,
            fontweight=options.info_fontweight,
        )
        if left:
            tl = anchor_ax.text(0.01, 1.02, left, ha="left", **text_kwargs)
            tl.set_path_effects([pe.withStroke(linewidth=2, foreground='black')])
        if right:
            tr = anchor_ax.text(0.99, 1.02, right, ha="right", **text_kwargs)
            tr.set_path_effects([pe.withStroke(linewidth=2, foreground='black')])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=options.transparent_bg)
    plt.close(fig)
    buf.seek(0)
    return buf.read()