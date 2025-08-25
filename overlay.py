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

    # Track sizing modes
    track_sizing_mode: str = "fit_width"
    track_height_px: int = 400
    track_width_px: int = 1200

    # Elevation plot sizing modes
    elev_sizing_mode: str = "fit_track_width"
    elev_height_px: int = 240
    elev_width_px: int = 1200

    # Graph toggles
    show_elev_graph: bool = True
    show_graph_label_x: bool = True
    show_graph_label_y: bool = True
    show_graph_axes_lines: bool = True
    grid: bool = False

    # Tick visibility controls
    show_x_ticks: bool = True
    show_y_ticks: bool = True
    show_x_ticklabels: bool = True
    show_y_ticklabels: bool = True

    # Peak controls (elevation chart)
    show_peak: bool = True
    show_peak_marker: bool = True
    show_peak_text: bool = True

    # Peak controls for GPX track plot
    show_track_peak: bool = True
    show_track_peak_marker: bool = True
    show_track_peak_text: bool = True

    # Run info block visibility
    show_run_info: bool = True

    # Individual run info fields + label toggles
    show_location: bool = True
    label_location: bool = True

    show_distance: bool = True
    label_distance: bool = True

    show_elev_gain: bool = True
    label_elev_gain: bool = True

    show_time: bool = True
    label_time: bool = True

    show_temperature: bool = True
    label_temperature: bool = True
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
    info_fontsize: int = 24

    # Run info font customization
    info_fontfamily: str = "sans-serif"
    info_fontstyle: str = "normal"
    info_fontweight: str = "bold"

    # Line styles
    line_width_track: float = 3.0
    line_width_elev: float = 2.0
    color_track: str = "#000000"   # black
    color_elev: str = "#1f77b4"    # matplotlib default blue

    # Glow styles (per plot)
    show_glow_track: bool = False
    glow_color_track: str = "#FFFFFF"
    glow_width_track: float = 6.0

    show_glow_elev: bool = False
    glow_color_elev: str = "#FFFFFF"
    glow_width_elev: float = 6.0


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


def _apply_glow(line, glow_color: str, glow_width: float):
    """Apply a glow/stroke effect to a Matplotlib line."""
    try:
        line.set_path_effects([pe.Stroke(linewidth=glow_width, foreground=glow_color), pe.Normal()])
    except Exception:
        pass


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

    # --- Compute data extents for track ---
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    y_min, y_max = np.nanmin(y), np.nanmax(y)
    if x_max - x_min < 1e-6: x_max += 1.0
    if y_max - y_min < 1e-6: y_max += 1.0
    data_aspect = (y_max - y_min) / (x_max - x_min)

    # --- LAYOUT BOUNDS ---
    top_y = 1 - title_h_frac - 0.02
    base_margin = 0.06
    avail_w_frac = 0.92

    # Track sizing
    if options.track_sizing_mode == "fixed_height":
        track_h_frac = min(top_y - base_margin, options.track_height_px / options.height_px)
        track_w_frac = min(avail_w_frac, track_h_frac / (data_aspect * (fig_w / fig_h)))
    elif options.track_sizing_mode == "fixed_width":
        track_w_frac = min(avail_w_frac, options.track_width_px / options.width_px)
        track_h_frac = min(top_y - base_margin, data_aspect * track_w_frac * (fig_w / fig_h))
    else:  # fit_width
        track_w_frac = avail_w_frac
        track_h_frac = min(top_y - base_margin, data_aspect * track_w_frac * (fig_w / fig_h))

    # Elevation desired sizes
    elev_h_frac_desired = 0.22 if options.show_elev_graph else 0.0
    elev_w_frac_desired = track_w_frac
    if options.show_elev_graph:
        if options.elev_sizing_mode == "fixed_height":
            elev_h_frac_desired = options.elev_height_px / options.height_px
        elif options.elev_sizing_mode == "fixed_width":
            elev_w_frac_desired = min(avail_w_frac, options.elev_width_px / options.width_px)
        else:
            elev_w_frac_desired = track_w_frac

    # Vertical allocation
    total_needed_h = track_h_frac + (elev_h_frac_desired if options.show_elev_graph else 0)
    max_h = top_y - base_margin
    if total_needed_h > max_h and options.show_elev_graph:
        overflow = total_needed_h - max_h
        track_h_frac = max(0.12, track_h_frac - overflow)

    # Track axis
    track_x0 = (1 - track_w_frac) / 2
    elev_h_actual = (elev_h_frac_desired if options.show_elev_graph else 0.0)
    track_y0 = base_margin + elev_h_actual
    track_ax = fig.add_axes([track_x0, track_y0, track_w_frac, track_h_frac])
    track_ax.set_axis_off()
    xpad = (x_max - x_min) * 0.05
    ypad = (y_max - y_min) * 0.05
    track_ax.set_xlim(x_min - xpad, x_max + xpad)
    track_ax.set_ylim(y_min - ypad, y_max + ypad)
    track_ax.set_aspect('equal', adjustable='box')
    line_track, = track_ax.plot(x, y, linewidth=options.line_width_track, color=options.color_track)
    if options.show_glow_track:
        _apply_glow(line_track, options.glow_color_track, options.glow_width_track)

    # --- Track peak marker/text ---
    if options.show_track_peak:
        elev_mask = ~np.isnan(elevs)
        if np.any(elev_mask):
            idx = int(np.nanargmax(elevs[elev_mask]))
            valid_indices = np.flatnonzero(elev_mask)
            real_idx = int(valid_indices[idx])
            peak_x = x[real_idx]
            peak_y = y[real_idx]
            peak_elev_val = elevs[real_idx] * (3.28084 if imperial else 1.0)
            if options.show_track_peak_marker:
                track_ax.plot([peak_x], [peak_y], marker='o')
            if options.show_track_peak_text:
                txt = f"{int(round(peak_elev_val))} {'ft' if imperial else 'm'}"
                dx = (x_max - x_min) * 0.01
                dy = (y_max - y_min) * 0.01
                tt = track_ax.text(peak_x + dx, peak_y + dy, txt,
                                   fontsize=options.axes_fontsize,
                                   ha='left', va='bottom')
                tt.set_path_effects([pe.withStroke(linewidth=2, foreground='white')])

    # Elevation axis
    elev_ax = None
    if options.show_elev_graph:
        elev_ax_h = min(elev_h_frac_desired, top_y - base_margin - track_h_frac)
        elev_w_frac = elev_w_frac_desired
        elev_x0 = (1 - elev_w_frac) / 2
        elev_y0 = base_margin
        elev_ax = fig.add_axes([elev_x0, elev_y0, elev_w_frac, elev_ax_h])

        elev_mask = ~np.isnan(elevs)
        line_elev, = elev_ax.plot(dist_series[elev_mask], elev_series[elev_mask],
                                  linewidth=options.line_width_elev, color=options.color_elev)
        if options.show_glow_elev:
            _apply_glow(line_elev, options.glow_color_elev, options.glow_width_elev)

        if options.grid:
            elev_ax.grid(True, alpha=0.3)
        for spine in elev_ax.spines.values():
            spine.set_visible(options.show_graph_axes_lines)

        # Labels
        if options.show_graph_label_x:
            elev_ax.set_xlabel(dist_label, fontsize=options.axes_fontsize)
        else:
            elev_ax.set_xlabel("")
        if options.show_graph_label_y:
            elev_ax.set_ylabel(elev_label, fontsize=options.axes_fontsize)
        else:
            elev_ax.set_ylabel("")

        # Ticks & tick labels
        elev_ax.tick_params(axis='x',
                            bottom=options.show_x_ticks,
                            labelbottom=options.show_x_ticklabels,
                            labelsize=options.axes_fontsize)
        elev_ax.tick_params(axis='y',
                            left=options.show_y_ticks,
                            labelleft=options.show_y_ticklabels,
                            labelsize=options.axes_fontsize)

        # Peak features (elevation chart)
        if options.show_peak and np.any(elev_mask):
            elev_vals = elev_series[elev_mask]
            dist_vals = dist_series[elev_mask]
            idx2 = int(np.nanargmax(elev_vals))
            peak_elev = elev_vals[idx2]
            peak_dist = dist_vals[idx2]
            if options.show_peak_marker:
                elev_ax.plot([peak_dist], [peak_elev], marker='o')
            if options.show_peak_text:
                txt = f"Peak: {peak_elev:.0f} {'ft' if imperial else 'm'}"
                tt = elev_ax.text(peak_dist, peak_elev, "  " + txt, va="bottom", ha="left", fontsize=options.axes_fontsize)
                tt.set_path_effects([pe.withStroke(linewidth=2, foreground='white')])

    # Footer / run info
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
        anchor_ax = elev_ax if (options.show_elev_graph) else track_ax
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