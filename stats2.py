"""
stats2.py
---------
Analyses the centre-of-mass trajectory produced by generate_step_1.py.

For every detected frame this computes:
  - smoothed COM position
  - velocity  (vx, vy, speed)   in pixels/second
  - acceleration (ax, ay, accel) in pixels/second²

Sign convention (matches visual intuition for climbing):
  - vy_px_s  > 0  →  moving UP   (sign is flipped relative to image y-axis)
  - ay_px_s2 > 0  →  accelerating UP

Prints a motion summary to the console and saves full per-frame data to
com_motion.json, which can be loaded by downstream scripts for further analysis.

Parameters
----------
All configurable values are at the top of the file — no argparse.
"""

import json
import numpy as np

# ── Parameters ────────────────────────────────────────────────────────────────
COM_PATH    = "com.json"
OUTPUT_PATH = "com_motion.json"

# Savitzky-Golay smoothing applied to the raw COM trajectory before
# differentiation.  Larger window = smoother but less temporal detail.
# Window must be odd; polynomial order must be < window.
SMOOTH_WINDOW = 51
SMOOTH_POLY   = 3

# Extra smoothing pass applied to acceleration only (after differentiation).
# Acceleration is the second derivative so it amplifies noise more than velocity.
# Must be odd; set to 1 to disable.
ACCEL_SMOOTH_WINDOW = 51
ACCEL_SMOOTH_POLY   = 2

# Speed below which the climber is considered "stationary" (pixels/second).
# Tune this based on your video resolution; ~1% of frame height per second
# is a reasonable starting point.
STATIC_THRESHOLD = 15.0

# Acceleration above which a frame is flagged as a "dynamic move" (px/s²).
# Set to a high percentile of the observed acceleration distribution.
DYNAMIC_THRESHOLD_PERCENTILE = 90


# ── Smoothing ─────────────────────────────────────────────────────────────────

def smooth(arr: np.ndarray, window: int, poly: int) -> np.ndarray:
    """
    Savitzky-Golay smoothing.  Falls back to Gaussian-weighted convolution
    if scipy is not installed.
    """
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(arr, window, poly).astype(float)
    except ImportError:
        # Gaussian kernel fallback
        sigma = window / 4.0
        k = np.exp(-0.5 * ((np.arange(window) - window // 2) / sigma) ** 2)
        k /= k.sum()
        # 'same' mode means edges are computed with zero-padding; replace
        # edge values with the nearest valid smoothed value
        out = np.convolve(arr, k, mode='same')
        hw = window // 2
        out[:hw]  = out[hw]
        out[-hw:] = out[-hw - 1]
        return out


# ── Main ──────────────────────────────────────────────────────────────────────

def analyse_com(com_path: str, output_path: str) -> None:
    # ── Load data ─────────────────────────────────────────────────────────
    with open(com_path) as f:
        com_data = json.load(f)

    video  = com_data["video"]
    fps    = float(video["fps"])
    width  = int(video["width"])
    height = int(video["height"])

    all_entries = com_data["com_per_frame"]
    detected    = [e for e in all_entries if e["px"] is not None]

    if len(detected) < SMOOTH_WINDOW:
        print(f"Only {len(detected)} detected frames — need at least {SMOOTH_WINDOW}. Aborting.")
        return

    # ── Extract arrays ────────────────────────────────────────────────────
    frames  = np.array([e["frame"]       for e in detected], dtype=float)
    times   = np.array([e["timestamp_s"] for e in detected], dtype=float)
    px_raw  = np.array([float(e["px"])   for e in detected])
    py_raw  = np.array([float(e["py"])   for e in detected])

    # Ensure window is odd and not larger than the data
    w = min(SMOOTH_WINDOW, len(px_raw))
    if w % 2 == 0:
        w -= 1
    w = max(w, 3)

    # ── Smooth positions ──────────────────────────────────────────────────
    px_sm = smooth(px_raw, w, SMOOTH_POLY)
    py_sm = smooth(py_raw, w, SMOOTH_POLY)

    # ── Velocity  (central differences via numpy.gradient) ───────────────
    # numpy.gradient uses central differences for interior points and
    # one-sided differences at the edges — appropriate for non-uniform spacing.
    vx    =  np.gradient(px_sm, times)       # px/s, rightward positive
    vy_up = -np.gradient(py_sm, times)       # px/s, UPWARD positive (flip image y)
    speed = np.sqrt(vx**2 + vy_up**2)

    # ── Acceleration ──────────────────────────────────────────────────────
    ax    =  np.gradient(vx,    times)       # px/s², rightward positive
    ay_up =  np.gradient(vy_up, times)       # px/s², UPWARD positive
    accel = np.sqrt(ax**2 + ay_up**2)

    # Extra smoothing pass on acceleration only
    if ACCEL_SMOOTH_WINDOW > 1:
        aw = min(ACCEL_SMOOTH_WINDOW, len(accel))
        if aw % 2 == 0: aw -= 1
        aw = max(aw, 3)
        ax    = smooth(ax,    aw, ACCEL_SMOOTH_POLY)
        ay_up = smooth(ay_up, aw, ACCEL_SMOOTH_POLY)
        accel = smooth(accel, aw, ACCEL_SMOOTH_POLY)

    # ── Identify dynamic moves ────────────────────────────────────────────
    dynamic_threshold = float(np.percentile(accel, DYNAMIC_THRESHOLD_PERCENTILE))
    is_dynamic = accel >= dynamic_threshold
    is_static  = speed < STATIC_THRESHOLD

    # ── Net vertical displacement ─────────────────────────────────────────
    # py decreases as the climber goes up; flip for "ascent positive"
    total_ascent  = float(max(0.0, py_raw[0] - np.min(py_sm)))    # downward in image = upward climb
    total_descent = float(max(0.0, np.max(py_sm) - py_raw[0]))
    net_ascent    = float(py_raw[0] - py_raw[-1])                  # positive = climbed up

    climb_duration = float(times[-1] - times[0])

    # ── Print summary ─────────────────────────────────────────────────────
    SEP = "─" * 56
    print(f"\n{SEP}")
    print(f"  COM Motion Analysis")
    print(SEP)
    print(f"  Frames analysed    : {len(detected)}")
    print(f"  Duration           : {climb_duration:.1f} s")
    print(f"  Video resolution   : {width}×{height} px\n")

    print(f"  ── Displacement ────────────────────────────────────")
    print(f"  Net vertical ascent: {net_ascent:+.0f} px  "
          f"({'up' if net_ascent > 0 else 'down'})")
    print(f"  Total ascent       : {total_ascent:.0f} px")
    print(f"  Total descent      : {total_descent:.0f} px\n")

    print(f"  ── Speed (pixels/second) ───────────────────────────")
    print(f"  Mean speed         : {np.mean(speed):.1f} px/s")
    print(f"  Peak speed         : {np.max(speed):.1f} px/s  "
          f"(at {times[np.argmax(speed)]:.2f}s)")
    print(f"  Mean upward vel.   : {np.mean(vy_up[vy_up > 0]):.1f} px/s  "
          f"(when ascending)")
    print(f"  Mean downward vel. : {abs(np.mean(vy_up[vy_up < 0])):.1f} px/s  "
          f"(when descending)")
    pct_static  = 100.0 * np.sum(is_static)  / len(is_static)
    pct_moving  = 100.0 - pct_static
    print(f"  Time moving        : {pct_moving:.0f}%")
    print(f"  Time stationary    : {pct_static:.0f}%  "
          f"(speed < {STATIC_THRESHOLD:.0f} px/s)\n")

    print(f"  ── Acceleration (pixels/second²) ───────────────────")
    print(f"  Mean acceleration  : {np.mean(accel):.1f} px/s²")
    print(f"  Peak acceleration  : {np.max(accel):.1f} px/s²  "
          f"(at {times[np.argmax(accel)]:.2f}s)")
    n_dynamic = int(np.sum(is_dynamic))
    pct_dyn   = 100.0 * n_dynamic / len(is_dynamic)
    print(f"  Dynamic moves      : {n_dynamic} frames ({pct_dyn:.0f}%)  "
          f"[accel ≥ {dynamic_threshold:.1f} px/s², top {100-DYNAMIC_THRESHOLD_PERCENTILE:.0f}%]")
    print(SEP + "\n")

    # ── Build per-frame output ────────────────────────────────────────────
    motion_lookup = {}
    for i, e in enumerate(detected):
        motion_lookup[e["frame"]] = {
            "frame":       e["frame"],
            "timestamp_s": e["timestamp_s"],
            "px":          round(float(px_sm[i]), 2),
            "py":          round(float(py_sm[i]), 2),
            "x":           round(float(e["x"]), 6),
            "y":           round(float(e["y"]), 6),
            "vx_px_s":     round(float(vx[i]),    2),
            "vy_px_s":     round(float(vy_up[i]), 2),   # upward positive
            "speed_px_s":  round(float(speed[i]), 2),
            "ax_px_s2":    round(float(ax[i]),    2),
            "ay_px_s2":    round(float(ay_up[i]), 2),   # upward positive
            "accel_px_s2": round(float(accel[i]), 2),
            "is_dynamic":  bool(is_dynamic[i]),
            "is_static":   bool(is_static[i]),
        }

    per_frame_out = []
    for e in all_entries:
        fi = e["frame"]
        if fi in motion_lookup:
            per_frame_out.append(motion_lookup[fi])
        else:
            per_frame_out.append({
                "frame":       fi,
                "timestamp_s": e["timestamp_s"],
                "px": None, "py": None,
                "x":  None, "y":  None,
                "vx_px_s":  None, "vy_px_s":  None, "speed_px_s":  None,
                "ax_px_s2": None, "ay_px_s2": None, "accel_px_s2": None,
                "is_dynamic": None, "is_static": None,
            })

    # ── Save ──────────────────────────────────────────────────────────────
    summary = {
        "duration_s":            round(climb_duration, 3),
        "detected_frames":       len(detected),
        "net_ascent_px":         round(net_ascent, 1),
        "total_ascent_px":       round(total_ascent, 1),
        "total_descent_px":      round(total_descent, 1),
        "mean_speed_px_s":       round(float(np.mean(speed)), 2),
        "peak_speed_px_s":       round(float(np.max(speed)),  2),
        "peak_speed_at_s":       round(float(times[np.argmax(speed)]), 3),
        "mean_accel_px_s2":      round(float(np.mean(accel)), 2),
        "peak_accel_px_s2":      round(float(np.max(accel)),  2),
        "peak_accel_at_s":       round(float(times[np.argmax(accel)]), 3),
        "dynamic_threshold_px_s2": round(dynamic_threshold, 2),
        "pct_time_moving":       round(pct_moving,  1),
        "pct_time_stationary":   round(pct_static,  1),
    }

    output = {
        "video":     video,
        "smoothing": {"window": w, "poly": SMOOTH_POLY},
        "summary":   summary,
        "per_frame": per_frame_out,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    import os
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Saved → {output_path}  ({size_mb:.1f} MB)\n")


if __name__ == "__main__":
    analyse_com(COM_PATH, OUTPUT_PATH)
