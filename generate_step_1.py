"""
generate_step_1.py
------------------
Reads pose_data.json and identifies handholds and footholds —
positions where the climber's hands or feet were stationary.

A hold is detected when a wrist/ankle landmark moves below a velocity
threshold for a minimum number of consecutive frames.

Requirements:
    pip install numpy tqdm

Output:
    holds.json  — structured hold events:
    {
        "video": { ... },
        "parameters": { ... },
        "handholds": [
            {
                "id": 0,
                "side": "left",
                "start_frame": 45,
                "end_frame":   120,
                "start_time_s": 0.75,
                "end_time_s":   2.01,
                "duration_s":   1.26,
                "x": 0.42,      # normalised 0–1 (fraction of frame width)
                "y": 0.35,      # normalised 0–1 (fraction of frame height)
                "px": 507,      # pixel x (using video native resolution)
                "py": 917       # pixel y
            },
            ...
        ],
        "footholds": [ ... ]   # same structure
    }
"""

import json
import sys

# ── Parameters ────────────────────────────────────────────────────────────────

POSE_PATH   = "pose_data.json"   # input: pose data from extract_pose.py
OUTPUT_PATH = "holds.json"       # output: detected holds
COM_PATH    = "com.json"         # output: centre of mass per frame

# Max normalised speed (per frame) to consider a limb stationary.
# Lower = stricter (fewer, longer holds). Raise if holds are being missed.
VELOCITY_THRESHOLD = 0.005

# Minimum consecutive stationary frames to count as a hold.
# At 60 fps: 10 frames ≈ 0.17 s. Raise to filter out brief pauses.
MIN_HOLD_FRAMES = 85

# Rolling window size (frames) for smoothing position and velocity signals.
# Larger = smoother but slower to respond to real movement.
SMOOTH_WINDOW = 4

# Bridge stationary segments separated by this many frames or fewer.
# Prevents a single noisy frame from splitting one hold into two.
MAX_GAP_FRAMES = 5

# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from tqdm import tqdm

# ── Landmark indices we care about ────────────────────────────────────────────
LANDMARK_NAMES = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR",
    "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX",
    "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]
IDX = {name: i for i, name in enumerate(LANDMARK_NAMES)}

# Points to track for handholds and footholds
HAND_POINTS = {
    "left":  IDX["LEFT_WRIST"],
    "right": IDX["RIGHT_WRIST"],
}
FOOT_POINTS = {
    "left":  IDX["LEFT_ANKLE"],
    "right": IDX["RIGHT_ANKLE"],
}


# ── Utility functions ─────────────────────────────────────────────────────────

def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple rolling mean using convolution; pads edges with edge values."""
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def extract_positions(frames: list, landmark_idx: int) -> np.ndarray:
    """
    Returns an (N, 2) array of [x, y] positions for every frame.
    Frames where the landmark was not detected are NaN.
    """
    N = len(frames)
    positions = np.full((N, 2), np.nan)
    for frame in frames:
        fi = frame["frame"]
        if fi >= N:
            continue
        if frame["detected"] and frame["landmarks"]:
            lm = frame["landmarks"][landmark_idx]
            positions[fi] = [lm["x"], lm["y"]]
    return positions


def compute_velocity(positions: np.ndarray, smooth_window: int) -> np.ndarray:
    """
    Compute per-frame speed (magnitude of displacement from previous frame).
    Returns a 1-D array of length N; first frame is always 0.
    NaN positions produce NaN velocity.
    """
    # Smooth positions first to reduce detection noise
    x_smooth = rolling_mean(np.where(np.isnan(positions[:, 0]), 0, positions[:, 0]), smooth_window)
    y_smooth = rolling_mean(np.where(np.isnan(positions[:, 1]), 0, positions[:, 1]), smooth_window)

    dx = np.diff(x_smooth, prepend=x_smooth[0])
    dy = np.diff(y_smooth, prepend=y_smooth[0])
    speed = np.sqrt(dx**2 + dy**2)

    # Mask frames where the landmark wasn't detected
    nan_mask = np.isnan(positions[:, 0])
    speed[nan_mask] = np.nan

    # Smooth the speed signal itself
    speed_clean = np.where(np.isnan(speed), 0, speed)
    speed_smooth = rolling_mean(speed_clean, smooth_window)
    speed_smooth[nan_mask] = np.nan

    return speed_smooth


def find_stationary_segments(
    speed: np.ndarray,
    fps: float,
    velocity_threshold: float,
    min_hold_frames: int,
    max_gap_frames: int,
) -> list[tuple[int, int]]:
    """
    Find contiguous runs where speed < velocity_threshold.
    Small gaps (≤ max_gap_frames) between stationary runs are bridged.
    Segments shorter than min_hold_frames are discarded.
    Returns list of (start_frame, end_frame) tuples (inclusive).
    """
    N = len(speed)
    stationary = np.zeros(N, dtype=bool)
    for i in range(N):
        if not np.isnan(speed[i]):
            stationary[i] = speed[i] < velocity_threshold

    # Bridge small gaps
    if max_gap_frames > 0:
        i = 0
        while i < N:
            if not stationary[i]:
                # Find end of gap
                gap_start = i
                while i < N and not stationary[i]:
                    i += 1
                gap_end = i
                gap_len = gap_end - gap_start
                # Bridge if gap is short and bounded by stationary on both sides
                if gap_len <= max_gap_frames and gap_start > 0 and gap_end < N:
                    stationary[gap_start:gap_end] = True
            else:
                i += 1

    # Extract contiguous segments
    segments = []
    in_seg = False
    seg_start = 0
    for i in range(N):
        if stationary[i] and not in_seg:
            seg_start = i
            in_seg = True
        elif not stationary[i] and in_seg:
            seg_end = i - 1
            if (seg_end - seg_start + 1) >= min_hold_frames:
                segments.append((seg_start, seg_end))
            in_seg = False
    if in_seg and (N - 1 - seg_start + 1) >= min_hold_frames:
        segments.append((seg_start, N - 1))

    return segments


def segments_to_holds(
    segments: list[tuple[int, int]],
    positions: np.ndarray,
    side: str,
    fps: float,
    width: int,
    height: int,
) -> list[dict]:
    """Convert (start, end) segments into hold records with position data."""
    holds = []
    for start, end in segments:
        # Average position over the stationary segment (ignore NaNs)
        seg_pos = positions[start : end + 1]
        valid = seg_pos[~np.isnan(seg_pos[:, 0])]
        if len(valid) == 0:
            continue
        avg_x = float(np.mean(valid[:, 0]))
        avg_y = float(np.mean(valid[:, 1]))

        holds.append({
            "side":         side,
            "start_frame":  int(start),
            "end_frame":    int(end),
            "start_time_s": round(start / fps, 3),
            "end_time_s":   round(end / fps, 3),
            "duration_s":   round((end - start) / fps, 3),
            "x":            round(avg_x, 5),
            "y":            round(avg_y, 5),
            "px":           int(round(avg_x * width)),
            "py":           int(round(avg_y * height)),
        })
    return holds


# ── Centre of Mass estimation ─────────────────────────────────────────────────
#
# Uses Winter's (2009) body-segment parameters: each segment has a known fraction
# of total body mass and its CoM sits at a known fraction along the segment from
# proximal to distal end.
#
# Segments and their definitions:
#   (mass_fraction, proximal_landmark, distal_landmark, com_fraction_from_proximal)
# For the head there is no proximal/distal pair, so we use NOSE directly.

SEGMENTS = [
    # Trunk
    (0.497, "MID_SHOULDER", "MID_HIP",        0.43),
    # Left arm
    (0.028, "LEFT_SHOULDER",  "LEFT_ELBOW",   0.436),
    (0.016, "LEFT_ELBOW",     "LEFT_WRIST",   0.430),
    (0.006, "LEFT_WRIST",     "LEFT_INDEX",   0.506),
    # Right arm
    (0.028, "RIGHT_SHOULDER", "RIGHT_ELBOW",  0.436),
    (0.016, "RIGHT_ELBOW",    "RIGHT_WRIST",  0.430),
    (0.006, "RIGHT_WRIST",    "RIGHT_INDEX",  0.506),
    # Left leg
    (0.100, "LEFT_HIP",       "LEFT_KNEE",    0.433),
    (0.047, "LEFT_KNEE",      "LEFT_ANKLE",   0.433),
    (0.015, "LEFT_ANKLE",     "LEFT_FOOT_INDEX", 0.500),
    # Right leg
    (0.100, "RIGHT_HIP",      "RIGHT_KNEE",   0.433),
    (0.047, "RIGHT_KNEE",     "RIGHT_ANKLE",  0.433),
    (0.015, "RIGHT_ANKLE",    "RIGHT_FOOT_INDEX", 0.500),
]
HEAD_MASS = 0.081   # head treated as a point mass at NOSE


def com_from_frame(landmarks: list) -> tuple[float, float] | None:
    """
    Estimate CoM (x, y) in normalised coords from a single frame's landmark list.
    Returns None if too many key landmarks are missing.
    """
    # Build name → (x, y) lookup, skipping low-visibility landmarks
    lm_map = {}
    for lm in landmarks:
        if lm["visibility"] >= 0.3:   # relaxed threshold for CoM (we average many points)
            lm_map[lm["name"]] = (lm["x"], lm["y"])

    # Virtual midpoints
    if "LEFT_SHOULDER" in lm_map and "RIGHT_SHOULDER" in lm_map:
        ls, rs = lm_map["LEFT_SHOULDER"], lm_map["RIGHT_SHOULDER"]
        lm_map["MID_SHOULDER"] = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    if "LEFT_HIP" in lm_map and "RIGHT_HIP" in lm_map:
        lh, rh = lm_map["LEFT_HIP"], lm_map["RIGHT_HIP"]
        lm_map["MID_HIP"] = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)

    total_weight = 0.0
    wx, wy = 0.0, 0.0

    # Head
    if "NOSE" in lm_map:
        nx, ny = lm_map["NOSE"]
        wx += HEAD_MASS * nx
        wy += HEAD_MASS * ny
        total_weight += HEAD_MASS

    # Segments
    for mass, prox_name, dist_name, frac in SEGMENTS:
        if prox_name in lm_map and dist_name in lm_map:
            px, py = lm_map[prox_name]
            dx, dy = lm_map[dist_name]
            cx = px + frac * (dx - px)
            cy = py + frac * (dy - py)
            wx += mass * cx
            wy += mass * cy
            total_weight += mass

    if total_weight < 0.3:   # less than 30% of body mass visible — unreliable
        return None

    return (wx / total_weight, wy / total_weight)


def compute_com_series(frames: list, fps: float, width: int, height: int) -> list[dict]:
    """Compute CoM for every frame; returns list of per-frame dicts."""
    results = []
    for frame in frames:
        fi  = frame["frame"]
        ts  = round(fi / fps, 4)
        com = None
        if frame["detected"] and frame["landmarks"]:
            com = com_from_frame(frame["landmarks"])

        if com is not None:
            results.append({
                "frame":       fi,
                "timestamp_s": ts,
                "x":           round(com[0], 5),
                "y":           round(com[1], 5),
                "px":          int(round(com[0] * width)),
                "py":          int(round(com[1] * height)),
            })
        else:
            results.append({
                "frame":       fi,
                "timestamp_s": ts,
                "x":           None,
                "y":           None,
                "px":          None,
                "py":          None,
            })
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def generate(
    pose_path: str,
    output_path: str,
    velocity_threshold: float,
    min_hold_frames: int,
    smooth_window: int,
    max_gap_frames: int,
) -> None:

    print(f"Loading pose data from '{pose_path}' …")
    with open(pose_path) as f:
        data = json.load(f)

    fps    = data["video"]["fps"]
    width  = data["video"]["width"]
    height = data["video"]["height"]
    frames = data["landmarks"]

    print(f"  {len(frames)} frames  |  {fps:.2f} fps  |  {width}×{height}\n")

    params = {
        "velocity_threshold": velocity_threshold,
        "min_hold_frames":    min_hold_frames,
        "min_hold_duration_s": round(min_hold_frames / fps, 3),
        "smooth_window":      smooth_window,
        "max_gap_frames":     max_gap_frames,
    }

    all_handholds = []
    all_footholds = []

    # Process hands and feet
    tasks = [
        ("left hand",  HAND_POINTS["left"],  "left",  "hand"),
        ("right hand", HAND_POINTS["right"], "right", "hand"),
        ("left foot",  FOOT_POINTS["left"],  "left",  "foot"),
        ("right foot", FOOT_POINTS["right"], "right", "foot"),
    ]

    for label, lm_idx, side, kind in tasks:
        positions = extract_positions(frames, lm_idx)
        speed     = compute_velocity(positions, smooth_window)
        segments  = find_stationary_segments(
            speed, fps, velocity_threshold, min_hold_frames, max_gap_frames
        )
        holds = segments_to_holds(segments, positions, side, fps, width, height)

        print(f"  {label:12s} → {len(holds):3d} {'handholds' if kind == 'hand' else 'footholds'} detected")

        if kind == "hand":
            all_handholds.extend(holds)
        else:
            all_footholds.extend(holds)

    # Sort by start time and assign IDs
    all_handholds.sort(key=lambda h: h["start_frame"])
    all_footholds.sort(key=lambda h: h["start_frame"])
    for i, h in enumerate(all_handholds):
        h["id"] = i
    for i, h in enumerate(all_footholds):
        h["id"] = i

    # ── Centre of mass ────────────────────────────────────────────────────────
    print("\n  Computing centre of mass …")
    com_series  = compute_com_series(frames, fps, width, height)
    com_valid   = [c for c in com_series if c["x"] is not None]
    com_missing = len(com_series) - len(com_valid)
    print(f"  CoM estimated for {len(com_valid)}/{len(com_series)} frames "
          f"({com_missing} frames missing too many landmarks)")

    output = {
        "video":      data["video"],
        "parameters": params,
        "handholds":  all_handholds,
        "footholds":  all_footholds,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    com_output = {
        "video":         data["video"],
        "com_per_frame": com_series,
    }

    with open(COM_PATH, "w") as f:
        json.dump(com_output, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n── Summary ──────────────────────────────────────────")
    print(f"  Handholds : {len(all_handholds)}  (left: {sum(1 for h in all_handholds if h['side']=='left')}, right: {sum(1 for h in all_handholds if h['side']=='right')})")
    print(f"  Footholds : {len(all_footholds)}  (left: {sum(1 for h in all_footholds if h['side']=='left')}, right: {sum(1 for h in all_footholds if h['side']=='right')})")

    if all_handholds:
        durations = [h["duration_s"] for h in all_handholds]
        print(f"\n  Handhold duration  avg={np.mean(durations):.2f}s  min={np.min(durations):.2f}s  max={np.max(durations):.2f}s")
    if all_footholds:
        durations = [h["duration_s"] for h in all_footholds]
        print(f"  Foothold duration  avg={np.mean(durations):.2f}s  min={np.min(durations):.2f}s  max={np.max(durations):.2f}s")

    if com_valid:
        ys = [c["y"] for c in com_valid]
        print(f"\n  CoM vertical range  top={min(ys):.3f}  bottom={max(ys):.3f}  "
              f"(normalised; 0=top of frame, 1=bottom)")

    print(f"\nSaved → {output_path}")
    print(f"Saved → {COM_PATH}\n")


if __name__ == "__main__":
    generate(
        pose_path=POSE_PATH,
        output_path=OUTPUT_PATH,
        velocity_threshold=VELOCITY_THRESHOLD,
        min_hold_frames=MIN_HOLD_FRAMES,
        smooth_window=SMOOTH_WINDOW,
        max_gap_frames=MAX_GAP_FRAMES,
    )
