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
    title: str = "My Run"
    subtitle: str = ""
    temperature_f: Optional[float] = None
    show_temperature: bool = True
    show_location: bool = True
    custom_location: Optional[str] = None
    width_px: int = 1920
    height_px: int = 1080
    margin_px: int = 32
    line_width_track: float = 3.0
    line_width_elev: float = 2.0
    transparent_bg: bool = True
    dpi: int = 150
    label_fontsize: int = 28
    title_fontsize: int = 48
    subtitle_fontsize: int = 28
    footer_fontsize: int = 24
    grid: bool = False


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
    # Robust parsing: handle bytes -> text for gpxpy
    # Try utf-8 first, ignore errors if needed
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

    track_margin = 0.05
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    y_min, y_max = np.nanmin(y), np.nanmax(y)
    if x_max - x_min < 1e-6: x_max += 1.0
    if y_max - y_min < 1e-6: y_max += 1.0

    geod = Geod(ellps='WGS84')
    dists_m = [0.0]
    for i in range(1, len(lats)):
        _, _, d = geod.inv(lons[i-1], lats[i-1], lons[i], lats[i])
        dists_m.append(dists_m[-1] + d)
    dists_km = np.array(dists_m) / 1000.0

    dpi = options.dpi
    fig_w = options.width_px / dpi
    fig_h = options.height_px / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_alpha(0.0 if options.transparent_bg else 1.0)

    gs = fig.add_gridspec(10, 20)
    ax_title = fig.add_subplot(gs[0, :])
    ax_track = fig.add_subplot(gs[1:7, :])
    ax_elev = fig.add_subplot(gs[7:10, :])

    ax_title.axis("off")
    t = ax_title.text(0.01, 0.65, options.title or "Activity",
                      fontsize=options.title_fontsize, weight="bold", va="center", ha="left")
    t.set_path_effects([pe.withStroke(linewidth=2, foreground='black')])
    if options.subtitle:
        s = ax_title.text(0.01, 0.2, options.subtitle,
                          fontsize=options.subtitle_fontsize, va="center", ha="left")
        s.set_path_effects([pe.withStroke(linewidth=2, foreground='black')])

    # FIXED: removed stray ')' that caused SyntaxError
    ax_track.set_axis_off()
    xpad = (x_max - x_min) * track_margin
    ypad = (y_max - y_min) * track_margin
    ax_track.set_xlim(x_min - xpad, x_max + xpad)
    ax_track.set_ylim(y_min - ypad, y_max + ypad)
    ax_track.set_aspect('equal', adjustable='box')
    ax_track.plot(x, y, linewidth=options.line_width_track)

    elev_mask = ~np.isnan(elevs)
    ax_elev.plot(dists_km[elev_mask], elevs[elev_mask], linewidth=options.line_width_elev)
    if options.grid:
        ax_elev.grid(True, alpha=0.3)
    ax_elev.set_xlabel("Distance (km)")
    ax_elev.set_ylabel("Elevation (m)")

    distance_mi = _miles(stats.distance_km)
    gain_ft = _feet(stats.elev_gain_m)
    dur = _format_duration(stats.duration)
    location = options.custom_location if options.custom_location else f"{stats.center_lat:.4f}, {stats.center_lon:.4f}"
    temp_str = f" • Temp {options.temperature_f:.0f}°F" if (options.show_temperature and options.temperature_f is not None) else ""

    footer_left = f"{location}" if options.show_location else ""
    footer_right = f"Dist {distance_mi:.2f} mi • Gain {gain_ft:.0f} ft • Time {dur}{temp_str}"

    for ax, text, anchor in [(ax_elev, footer_left, 'left'), (ax_elev, footer_right, 'right')]:
        if text:
            tx = 0.01 if anchor == 'left' else 0.99
            tt = ax.text(tx, 1.02, text, transform=ax.transAxes, ha=anchor, va="bottom",
                        fontsize=options.footer_fontsize, weight="bold")
            tt.set_path_effects([pe.withStroke(linewidth=2, foreground='black')])

    plt.subplots_adjust(left=0.06, right=0.97, top=0.92, bottom=0.08)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=options.transparent_bg)
    plt.close(fig)
    buf.seek(0)
    return buf.read()