import io
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Tuple, List

import gpxpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from pyproj import Geod, Transformer


# -------------------- Data classes --------------------

@dataclass
class OverlayOptions:
    # Units
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
    peak_marker_size_elev: float = 6.0
    peak_marker_color_elev: str = "#000000"

    # Peak controls for GPX track plot
    show_track_peak: bool = True
    show_track_peak_marker: bool = True
    show_track_peak_text: bool = True
    peak_marker_size_track: float = 6.0
    peak_marker_color_track: str = "#000000"

    # Start / Finish arrows
    show_start_finish_track: bool = True
    show_start_finish_elev: bool = True
    arrow_color_track: str = "#000000"
    arrow_color_elev: str = "#000000"
    arrow_size_track: float = 10.0  # points
    arrow_size_elev: float = 10.0   # points

    # End mileage label/marker
    show_end_mileage_track: bool = True
    show_end_mileage_elev: bool = True
    end_marker_style: str = "o"  # 'o','s','^','v','D','X'
    end_marker_size: float = 7.0
    end_marker_color: str = "#000000"
    end_label_color: str = "#000000"

    # Mile markers (track map)
    mile_markers_track: str = "none"  # none / ticks / ticks+labels
    mile_marker_size_track: float = 4.0
    mile_marker_color_track: str = "#000000"

    # Mile markers (elevation chart)
    mile_markers_elev: str = "none"   # none / ticks / ticks+labels
    mile_tick_label_rotation: float = 0.0

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
    color_track: str = "#000000"
    color_elev: str = "#1f77b4"
    style_track: str = "solid"     # solid/dashed/dotted
    style_elev: str = "solid"
    capstyle_track: str = "round"  # butt/round/projecting
    capstyle_elev: str = "round"
    joinstyle_track: str = "round" # miter/round/bevel
    joinstyle_elev: str = "round"

    # Glow styles (per plot)
    show_glow_track: bool = False
    glow_color_track: str = "#FFFFFF"
    glow_width_track: float = 6.0

    show_glow_elev: bool = False
    glow_color_elev: str = "#FFFFFF"
    glow_width_elev: float = 6.0

    # Shadow styles (per plot)
    show_shadow_track: bool = False
    shadow_color_track: str = "#000000"
    shadow_alpha_track: float = 0.4
    shadow_dx_track: float = 2.0
    shadow_dy_track: float = -2.0

    show_shadow_elev: bool = False
    shadow_color_elev: str = "#000000"
    shadow_alpha_elev: float = 0.4
    shadow_dx_elev: float = 2.0
    shadow_dy_elev: float = -2.0

    # Color gradient options
    gradient_track: str = "none"  # none / speed / elevation
    gradient_elev: str = "none"   # none / speed / elevation
    cmap_track: str = "viridis"
    cmap_elev: str = "viridis"


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


# -------------------- Helpers --------------------

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


def _linestyle(style: str):
    style = (style or "solid").lower()
    if style == "dashed":
        return "--"
    if style == "dotted":
        return ":"
    return "-"


def _apply_caps_joins(line, capstyle: str, joinstyle: str):
    cap = (capstyle or "round").lower()
    join = (joinstyle or "round").lower()
    line.set_solid_capstyle(cap)
    line.set_solid_joinstyle(join)
    line.set_dash_capstyle(cap)
    line.set_dash_joinstyle(join)


def _apply_glow(line, glow_color: str, glow_width: float):
    try:
        line.set_path_effects([pe.Stroke(linewidth=glow_width, foreground=glow_color), pe.Normal()])
    except Exception:
        pass


def _apply_shadow(line, dx_px: float, dy_px: float, color: str, alpha: float):
    try:
        # SimpleLineShadow offset is in points, not pixels; convert approx using 72 dpi
        # We'll scale with the figure dpi to keep visual consistency
        ax = line.axes
        dpi = ax.figure.dpi if ax and ax.figure else 72.0
        dx_pt = dx_px * 72.0 / dpi
        dy_pt = dy_px * 72.0 / dpi
        effects = list(line.get_path_effects()) if line.get_path_effects() else []
        effects.insert(0, pe.SimpleLineShadow(offset=(dx_pt, dy_pt), shadow_color=color, alpha=alpha))
        effects.append(pe.Normal())
        line.set_path_effects(effects)
    except Exception:
        pass


def _segments(x: np.ndarray, y: np.ndarray):
    pts = np.array([x, y]).T
    return np.stack([pts[:-1], pts[1:]], axis=1)


def _speeds(distances_m: np.ndarray, times: List[Optional[dt.datetime]]):
    # meter per second between points; last speed repeated for equal length
    v = np.zeros(len(distances_m))
    last_t = None
    last_d = None
    for i, (d, t) in enumerate(zip(distances_m, times)):
        if i == 0:
            v[i] = 0.0
        else:
            if t is not None and last_t is not None:
                dt_sec = (t - last_t).total_seconds()
                dd = d - last_d
                v[i] = dd / dt_sec if dt_sec > 0 else 0.0
            else:
                v[i] = v[i-1]
        last_t = t
        last_d = d
    # per-segment speeds = between i-1 and i
    seg_v = np.maximum(0.0, v[1:])
    if len(seg_v)==0:
        seg_v = np.array([0.0])
    return seg_v


def _grades(dists_m: np.ndarray, elevs: np.ndarray) -> np.ndarray:
    """
    Compute terrain grade (%) per segment.
    grade = 100 * delta_elevation / delta_horizontal_distance
    """
    dd = np.diff(dists_m)         # horizontal distance per segment (meters)
    de = np.diff(elevs)           # elevation change per segment (meters)
    dd = np.where(dd <= 0, np.nan, dd)  # avoid divide-by-zero
    grade = 100.0 * de / dd
    grade = np.where(np.isfinite(grade), grade, 0.0)  # replace NaN/inf
    return grade if len(grade) > 0 else np.array([0.0])



# -------------------- Main renderer --------------------

def generate_overlay_image(gpx_bytes: bytes, options: OverlayOptions) -> bytes:
    lats, lons, elevs, times = _parse_gpx(gpx_bytes)
    if len(lats) < 2:
        raise ValueError("GPX contains insufficient points.")

    stats = _compute_stats(lats, lons, elevs, times)
    x, y = _project_to_mercator(lats, lons)

    # Distance array along path
    geod = Geod(ellps='WGS84')
    dists_m = [0.0]
    for i in range(1, len(lats)):
        _, _, d = geod.inv(lons[i-1], lats[i-1], lons[i], lats[i])
        dists_m.append(dists_m[-1] + d)
    dists_m = np.array(dists_m)
    dists_km = dists_m / 1000.0

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

    # --- TITLE ---
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

    # --- Bounds/aspect ---
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    y_min, y_max = np.nanmin(y), np.nanmax(y)
    if x_max - x_min < 1e-6: x_max += 1.0
    if y_max - y_min < 1e-6: y_max += 1.0
    data_aspect = (y_max - y_min) / (x_max - x_min)

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
    else:
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

    total_needed_h = track_h_frac + (elev_h_frac_desired if options.show_elev_graph else 0)
    max_h = top_y - base_margin
    if total_needed_h > max_h and options.show_elev_graph:
        overflow = total_needed_h - max_h
        track_h_frac = max(0.12, track_h_frac - overflow)

    # --- TRACK AXIS ---
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
    line_track, = track_ax.plot(x, y, linewidth=options.line_width_track, color=options.color_track, linestyle=_linestyle(options.style_track))
    _apply_caps_joins(line_track, options.capstyle_track, options.joinstyle_track)
    if options.show_glow_track:
        _apply_glow(line_track, options.glow_color_track, options.glow_width_track)
    if options.show_shadow_track:
        _apply_shadow(line_track, options.shadow_dx_track, options.shadow_dy_track, options.shadow_color_track, options.shadow_alpha_track)

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
                track_ax.plot([peak_x], [peak_y], marker='o', markersize=options.peak_marker_size_track, color=options.peak_marker_color_track)
            if options.show_track_peak_text:
                txt = f"{int(round(peak_elev_val))} {'ft' if imperial else 'm'}"
                dx = (x_max - x_min) * 0.01
                dy = (y_max - y_min) * 0.01
                tt = track_ax.text(peak_x + dx, peak_y + dy, txt,
                                   fontsize=options.axes_fontsize,
                                   ha='left', va='bottom', color=options.peak_marker_color_track)



    # Track rendering (solid color or gradient)
    if options.gradient_track == "none" or len(x) < 2:
        line_track, = track_ax.plot(x, y, linewidth=options.line_width_track,
                                    color=options.color_track, linestyle=_linestyle(options.style_track))
        _apply_caps_joins(line_track, options.capstyle_track, options.joinstyle_track)
        if options.show_glow_track:
            _apply_glow(line_track, options.glow_color_track, options.glow_width_track)
        if options.show_shadow_track:
            _apply_shadow(line_track, options.shadow_dx_track, options.shadow_dy_track,
                          options.shadow_color_track, options.shadow_alpha_track)
    else:
        segs = _segments(x, y)
        if options.gradient_track == "speed":
            speeds = _speeds(dists_m, times)
            values = speeds
        elif options.gradient_track == 'elevation':
            values = elevs[1:]  # segment value at end point
        elif options.gradient_track == 'grade':
            values = _grades(dists_m, elevs)
            # optional: clamp to ±20% so outliers don't blow up colors
            values = np.clip(values, -30, 30)

        if options.gradient_track == "grade":
            norm = TwoSlopeNorm(vcenter=0.0, vmin=np.nanmin(values), vmax=np.nanmax(values))
        else:
            norm = Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))

        # abs on grade%
        # norm = Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))

        lc = LineCollection(segs, cmap=plt.get_cmap(options.cmap_track), norm=norm,
                            linewidths=options.line_width_track)
        lc.set_array(values)
        track_ax.add_collection(lc)
        # path effects on collection
        if options.show_glow_track or options.show_shadow_track:
            effects = []
            if options.show_shadow_track:
                # approximate shadow as stroke with dark color + alpha
                effects.append(pe.Stroke(linewidth=options.line_width_track + max(2.0, options.glow_width_track/2),
                                         foreground=options.shadow_color_track, alpha=options.shadow_alpha_track))
            if options.show_glow_track:
                effects.append(pe.Stroke(linewidth=options.glow_width_track, foreground=options.glow_color_track))
            effects.append(pe.Normal())
            try:
                lc.set_path_effects(effects)
            except Exception:
                pass

    # --- Start/Finish arrows on track ---
    if options.show_start_finish_track and len(x) >= 2:
        # Start arrow
        track_ax.annotate("", xy=(x[1], y[1]), xytext=(x[0], y[0]),
                          arrowprops=dict(arrowstyle="->", color=options.arrow_color_track, lw=1.5),
                          annotation_clip=False)
        # Finish arrow
        track_ax.annotate("", xy=(x[-2], y[-2]), xytext=(x[-1], y[-1]),
                          arrowprops=dict(arrowstyle="->", color=options.arrow_color_track, lw=1.5),
                          annotation_clip=False)

    # --- Mile markers on track ---
    if options.mile_markers_track != "none":
        unit = 1.0 if not imperial else 1.0  # dist_series already in mi/km
        spacing = 1.0  # 1 mi/km
        max_d = float(dist_series[-1])
        targets = np.arange(spacing, max_d + 1e-6, spacing)
        # interpolate along path for marker positions
        for t in targets:
            idx = np.searchsorted(dist_series, t)
            if idx <= 0 or idx >= len(dist_series): continue
            # linear interp between idx-1 and idx
            f = (t - dist_series[idx-1]) / (dist_series[idx] - dist_series[idx-1])
            mx = x[idx-1] + f*(x[idx]-x[idx-1])
            my = y[idx-1] + f*(y[idx]-y[idx-1])
            track_ax.plot(mx, my, marker='o', markersize=options.mile_marker_size_track,
                          color=options.mile_marker_color_track)
            if options.mile_markers_track == "ticks+labels":
                lbl = f"{int(round(t))} {'mi' if imperial else 'km'}"
                tt = track_ax.text(mx, my, f" {lbl}", va="center", ha="left",
                                   fontsize=max(8, options.axes_fontsize-2),
                                   color=options.mile_marker_color_track)
                tt.set_path_effects([pe.withStroke(linewidth=2, foreground='white')])

    # --- Track end mileage marker/label ---
    if options.show_end_mileage_track:
        mx, my = x[-1], y[-1]
        track_ax.plot(mx, my, marker=options.end_marker_style, markersize=options.end_marker_size,
                      color=options.end_marker_color)
        end_dist = dist_series[-1]
        label = f"{end_dist:.2f} {'mi' if imperial else 'km'}"
        tt = track_ax.text(mx, my, f"  {label}", va="center", ha="left",
                           fontsize=options.axes_fontsize, color=options.end_label_color)
        tt.set_path_effects([pe.withStroke(linewidth=2, foreground='white')])

    # --- ELEVATION AXIS ---
    elev_ax = None
    if options.show_elev_graph:
        elev_ax_h = min(elev_h_frac_desired, top_y - base_margin - track_h_frac)
        elev_w_frac = elev_w_frac_desired
        elev_x0 = (1 - elev_w_frac) / 2
        elev_y0 = base_margin
        elev_ax = fig.add_axes([elev_x0, elev_y0, elev_w_frac, elev_ax_h])

        elev_mask = ~np.isnan(elevs)
        line_elev, = elev_ax.plot(dist_series[elev_mask], elev_series[elev_mask], linewidth=options.line_width_elev, color=options.color_elev, linestyle=_linestyle(options.style_elev))
        _apply_caps_joins(line_elev, options.capstyle_elev, options.joinstyle_elev)
        if options.show_glow_elev:
            _apply_glow(line_elev, options.glow_color_elev, options.glow_width_elev)
        if options.show_shadow_elev:
            _apply_shadow(line_elev, options.shadow_dx_elev, options.shadow_dy_elev, options.shadow_color_elev, options.shadow_alpha_elev)

        if options.grid:
            elev_ax.grid(True, alpha=0.3)
        xs = dist_series[elev_mask]; ys = elev_series[elev_mask]

        if options.gradient_elev == "none" or len(xs) < 2:
            line_elev, = elev_ax.plot(xs, ys, linewidth=options.line_width_elev,
                                      color=options.color_elev, linestyle=_linestyle(options.style_elev))
            _apply_caps_joins(line_elev, options.capstyle_elev, options.joinstyle_elev)
            if options.show_glow_elev:
                _apply_glow(line_elev, options.glow_color_elev, options.glow_width_elev)
            if options.show_shadow_elev:
                _apply_shadow(line_elev, options.shadow_dx_elev, options.shadow_dy_elev,
                              options.shadow_color_elev, options.shadow_alpha_elev)
        else:
            segs = _segments(xs, ys)
            if options.gradient_elev == "speed":
                speeds = _speeds(dists_m, times)
                values = speeds[:len(segs)] if len(speeds) >= len(segs) else np.pad(speeds, (0, len(segs)-len(speeds)), 'edge')
            elif options.gradient_elev == "elevation":
                values = elevs[1:]
            elif options.gradient_elev == "grade":
                values = _grades(dists_m, elevs)
                # optional: clamp to ±20% so outliers don't blow up colors
                values = np.clip(values, -20, 20)


            # abs value
            #norm = Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))

            if options.gradient_elev == "grade":
                norm = TwoSlopeNorm(vcenter=0.0, vmin=np.nanmin(values), vmax=np.nanmax(values))
            else:
                norm = Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))




            lc = LineCollection(segs, cmap=plt.get_cmap(options.cmap_elev), norm=norm,
                                linewidths=options.line_width_elev)
            lc.set_array(values)
            elev_ax.add_collection(lc)
            # optional effects
            if options.show_glow_elev or options.show_shadow_elev:
                effects = []
                if options.show_shadow_elev:
                    effects.append(pe.Stroke(linewidth=options.line_width_elev + max(2.0, options.glow_width_elev/2),
                                             foreground=options.shadow_color_elev, alpha=options.shadow_alpha_elev))
                if options.show_glow_elev:
                    effects.append(pe.Stroke(linewidth=options.glow_width_elev, foreground=options.glow_color_elev))
                effects.append(pe.Normal())
                try:
                    lc.set_path_effects(effects)
                except Exception:
                    pass

        # Grid/axes
        if options.grid: elev_ax.grid(True, alpha=0.3)
        for spine in elev_ax.spines.values():
            spine.set_visible(options.show_graph_axes_lines)

        # Mile markers on elevation (override ticks)
        if options.mile_markers_elev != "none":
            max_d = float(dist_series[-1])
            spacing = 1.0
            ticks = np.arange(0, max_d + 1e-6, spacing)
            elev_ax.set_xticks(ticks)
            elev_ax.tick_params(axis='x', bottom=True, labelbottom=(options.mile_markers_elev=="ticks+labels"))
            if options.mile_markers_elev == "ticks+labels":
                labels = [f"{int(t)}" for t in ticks]
                elev_ax.set_xticklabels(labels, rotation=options.mile_tick_label_rotation)
            # Y ticks respect existing toggles
        else:
            elev_ax.tick_params(axis='x',
                                bottom=options.show_x_ticks,
                                labelbottom=options.show_x_ticklabels)

        # Labels after custom ticks (so we don't wipe them)
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


        # Peak on elevation
        if options.show_peak and np.any(elev_mask):
            elev_vals = elev_series[elev_mask]
            dist_vals = dist_series[elev_mask]
            idx2 = int(np.nanargmax(elev_vals))
            peak_elev = elev_vals[idx2]
            peak_dist = dist_vals[idx2]
            if options.show_peak_marker:
                elev_ax.plot([peak_dist], [peak_elev], marker='o',
                             markersize=options.peak_marker_size_elev, color=options.peak_marker_color_elev)
            if options.show_peak_text:
                txt = f"Peak: {peak_elev:.0f} {'ft' if imperial else 'm'}"
                tt = elev_ax.text(peak_dist, peak_elev, "  " + txt, va="bottom", ha="left",
                                  fontsize=options.axes_fontsize, color=options.peak_marker_color_elev)
                tt.set_path_effects([pe.withStroke(linewidth=2, foreground='white')])

        # Start/Finish arrows on elevation (0 and end)
        if options.show_start_finish_elev and len(xs) >= 2:
            elev_ax.annotate("", xy=(xs[1], ys[1]), xytext=(xs[0], ys[0]),
                             arrowprops=dict(arrowstyle="->", color=options.arrow_color_elev, lw=1.5))
            elev_ax.annotate("", xy=(xs[-2], ys[-2]), xytext=(xs[-1], ys[-1]),
                             arrowprops=dict(arrowstyle="->", color=options.arrow_color_elev, lw=1.5))

        # End mileage on elevation
        if options.show_end_mileage_elev:
            xe, ye = xs[-1], ys[-1]
            elev_ax.plot(xe, ye, marker=options.end_marker_style, markersize=options.end_marker_size,
                         color=options.end_marker_color)
            end_dist = dist_series[-1]
            label = f"{end_dist:.2f} {'mi' if imperial else 'km'}"
            tt = elev_ax.text(xe, ye, "  " + label, va="center", ha="left",
                              fontsize=options.axes_fontsize, color=options.end_label_color)
            tt.set_path_effects([pe.withStroke(linewidth=2, foreground='white')])

    # --- Footer / run info ---
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
