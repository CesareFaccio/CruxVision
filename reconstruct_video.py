"""
reconstruct_video.py
--------------------
Reads the original climbing video, pose_data.json, and holds.json, then renders
a new video with the pose skeleton and hold markers overlaid on every frame.

Hold markers:
  - Past holds    : small faded circle (shows the route already climbed)
  - Active hold   : large bright circle + "LH / RH / LF / RF" label
  - Colour coding : left hand=orange, right hand=red, left foot=cyan, right foot=blue

Centre of mass:
  - Yellow diamond drawn at the estimated CoM position each frame

Requirements:
    pip install opencv-python tqdm numpy

Usage:
    python reconstruct_video.py
    python reconstruct_video.py --video climbing_video.mov --pose pose_data.json --holds holds.json --com com.json --output pose_overlay.mp4
"""

import argparse
import json
import sys

import cv2
import numpy as np
from tqdm import tqdm

# ── Skeleton connections (pairs of landmark indices) ──────────────────────────
# Based on the standard MediaPipe Pose topology
CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),       # nose → left eye chain → left ear
    (0, 4), (4, 5), (5, 6), (6, 8),       # nose → right eye chain → right ear
    (9, 10),                               # mouth left ↔ right
    # Torso
    (11, 12),                              # shoulders
    (11, 23), (12, 24),                   # shoulder → hip
    (23, 24),                              # hips
    # Left arm
    (11, 13), (13, 15),                   # shoulder → elbow → wrist
    (15, 17), (15, 19), (15, 21),         # wrist → pinky / index / thumb
    (17, 19),                              # pinky ↔ index
    # Right arm
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22),
    (18, 20),
    # Left leg
    (23, 25), (25, 27),                   # hip → knee → ankle
    (27, 29), (27, 31), (29, 31),         # ankle → heel / foot index
    # Right leg
    (24, 26), (26, 28),
    (28, 30), (28, 32), (30, 32),
]

# Colour scheme (BGR)
COLOUR_BODY     = (0,   220, 100)   # green — torso & limbs
COLOUR_FACE     = (200, 200, 200)   # grey  — face landmarks
COLOUR_HANDS    = (0,   160, 255)   # orange — wrists/hands
COLOUR_FEET     = (255, 80,  80)    # blue  — ankles/feet
COLOUR_SKELETON = (0,   200, 80)    # line colour

HAND_INDICES = {15, 16, 17, 18, 19, 20, 21, 22}   # wrists + fingers
FOOT_INDICES = {27, 28, 29, 30, 31, 32}            # ankles + feet
FACE_INDICES = set(range(11))                       # 0–10

VISIBILITY_THRESHOLD = 0.5   # landmarks below this are not drawn

# ── Hold colours (BGR) ────────────────────────────────────────────────────────
HOLD_COLOURS = {
    ("hand", "left"):  (0,   165, 255),   # orange
    ("hand", "right"): (0,   0,   220),   # red
    ("foot", "left"):  (255, 220, 0  ),   # cyan
    ("foot", "right"): (255, 80,  0  ),   # blue
}
HOLD_LABELS = {
    ("hand", "left"):  "LH",
    ("hand", "right"): "RH",
    ("foot", "left"):  "LF",
    ("foot", "right"): "RF",
}


def landmark_colour(idx: int) -> tuple:
    if idx in FACE_INDICES:
        return COLOUR_FACE
    if idx in HAND_INDICES:
        return COLOUR_HANDS
    if idx in FOOT_INDICES:
        return COLOUR_FEET
    return COLOUR_BODY


def draw_pose(frame: np.ndarray, landmarks: list) -> np.ndarray:
    """Draw skeleton and landmark dots onto frame (in-place)."""
    h, w = frame.shape[:2]

    # Build pixel coords; skip low-visibility landmarks
    coords = {}
    for lm in landmarks:
        if lm["visibility"] >= VISIBILITY_THRESHOLD:
            coords[LANDMARK_NAMES.index(lm["name"])] = (
                int(lm["x"] * w),
                int(lm["y"] * h),
            )

    # Draw connections first (underneath dots)
    for a, b in CONNECTIONS:
        if a in coords and b in coords:
            cv2.line(frame, coords[a], coords[b], COLOUR_SKELETON, thickness=3, lineType=cv2.LINE_AA)

    # Draw landmark dots on top
    for idx, pt in coords.items():
        colour = landmark_colour(idx)
        cv2.circle(frame, pt, radius=6, color=colour, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(frame, pt, radius=6, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)  # outline

    return frame


def draw_holds(frame: np.ndarray, frame_idx: int, holds_by_frame: dict) -> None:
    """
    Draw hold markers for the current frame:
      - Past holds   : small faded filled circle
      - Active holds : large bright circle + label, with a pulsing ring
    holds_by_frame maps frame_idx → list of (hold_dict, kind, side) tuples
    that are active on that frame.  We also need the full hold list to draw
    past markers, so we accept a pre-built structure instead.
    """
    past   = holds_by_frame.get("past",   {}).get(frame_idx, [])
    active = holds_by_frame.get("active", {}).get(frame_idx, [])

    # Past holds — small, faded
    for hold, kind, side in past:
        colour = HOLD_COLOURS[(kind, side)]
        faded  = tuple(int(c * 0.35) for c in colour)
        pt = (hold["px"], hold["py"])
        cv2.circle(frame, pt, 14, faded, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 14, (0, 0, 0), 1, cv2.LINE_AA)

    # Active holds — prominent circle + label
    for hold, kind, side in active:
        colour = HOLD_COLOURS[(kind, side)]
        label  = HOLD_LABELS[(kind, side)]
        pt = (hold["px"], hold["py"])

        # Outer ring
        cv2.circle(frame, pt, 30, colour, 3, cv2.LINE_AA)
        # Filled inner circle
        cv2.circle(frame, pt, 20, colour, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 20, (0, 0, 0), 2, cv2.LINE_AA)

        # Label just above the circle
        text_pt = (pt[0] - 20, pt[1] - 35)
        cv2.putText(frame, label, text_pt, cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, label, text_pt, cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, colour, 2, cv2.LINE_AA)


def draw_com(frame: np.ndarray, px: int, py: int) -> None:
    """Draw a yellow diamond at the centre-of-mass position."""
    size = 18
    pts = np.array([
        [px,        py - size],   # top
        [px + size, py        ],   # right
        [px,        py + size],   # bottom
        [px - size, py        ],   # left
    ], dtype=np.int32)
    cv2.fillPoly(frame, [pts], (0, 230, 255), cv2.LINE_AA)       # yellow fill
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 0),  # black outline
                  thickness=2, lineType=cv2.LINE_AA)
    # "CoM" label
    cv2.putText(frame, "CoM", (px + 22, py + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, "CoM", (px + 22, py + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255), 1, cv2.LINE_AA)


def build_holds_by_frame(holds_data: dict, total_frames: int) -> dict:
    """
    Pre-compute, for every frame, which holds are past and which are active.
    Returns:
        {
            "past":   { frame_idx: [(hold, kind, side), ...] },
            "active": { frame_idx: [(hold, kind, side), ...] },
        }
    """
    from collections import defaultdict
    past_map   = defaultdict(list)
    active_map = defaultdict(list)

    for kind, key in [("hand", "handholds"), ("foot", "footholds")]:
        for hold in holds_data.get(key, []):
            side  = hold["side"]
            start = hold["start_frame"]
            end   = hold["end_frame"]

            for fi in range(start, min(end + 1, total_frames)):
                active_map[fi].append((hold, kind, side))

            # Mark this hold as "past" for every frame after it ends
            for fi in range(end + 1, total_frames):
                past_map[fi].append((hold, kind, side))

    return {"past": dict(past_map), "active": dict(active_map)}


# ── Landmark name → index lookup ──────────────────────────────────────────────
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


def reconstruct(video_path: str, pose_path: str, holds_path: str, com_path: str, output_path: str) -> None:
    # ── Load pose data ────────────────────────────────────────────────────────
    print(f"Loading pose data from {pose_path} …")
    with open(pose_path) as f:
        pose_data = json.load(f)

    frames_pose = {entry["frame"]: entry for entry in pose_data["landmarks"]}

    # ── Load holds data ───────────────────────────────────────────────────────
    import os
    holds_data      = {}
    holds_by_frame  = {"past": {}, "active": {}}
    if holds_path:
        if os.path.exists(holds_path):
            print(f"Loading holds from {holds_path} …")
            with open(holds_path) as f:
                holds_data = json.load(f)
        else:
            print(f"Warning: holds file '{holds_path}' not found — skipping hold markers.")

    # ── Load CoM data ─────────────────────────────────────────────────────────
    com_by_frame = {}
    if com_path:
        if os.path.exists(com_path):
            print(f"Loading CoM from {com_path} …")
            with open(com_path) as f:
                com_data = json.load(f)
            com_by_frame = {
                entry["frame"]: entry
                for entry in com_data["com_per_frame"]
                if entry["px"] is not None
            }
        else:
            print(f"Warning: CoM file '{com_path}' not found — skipping CoM overlay.")

    # ── Open source video ─────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Error: could not open video at '{video_path}'")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Source: {width}×{height} @ {fps:.2f} fps  ({total} frames)")

    if holds_data:
        holds_by_frame = build_holds_by_frame(holds_data, total)

    # ── Set up output writer ──────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        sys.exit(f"Error: could not create output file at '{output_path}'")

    # ── Render frames ─────────────────────────────────────────────────────────
    with tqdm(total=total, unit="frame", desc="Rendering overlay") as pbar:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Draw holds first (underneath skeleton)
            if holds_by_frame["active"] or holds_by_frame["past"]:
                draw_holds(frame, frame_idx, holds_by_frame)

            entry = frames_pose.get(frame_idx)
            if entry and entry["detected"] and entry["landmarks"]:
                draw_pose(frame, entry["landmarks"])

            # Draw CoM on top of skeleton
            com = com_by_frame.get(frame_idx)
            if com:
                draw_com(frame, com["px"], com["py"])

            # Stamp frame number + timestamp in corner
            ts = frame_idx / fps
            label = f"frame {frame_idx}  {ts:.2f}s"
            cv2.putText(frame, label, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, label, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

            out.write(frame)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    out.release()
    print(f"\nSaved → {output_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay pose skeleton, holds, and CoM on climbing video.")
    parser.add_argument("--video",  default="climbing_video.mov", help="Source video")
    parser.add_argument("--pose",   default="pose_data.json",     help="Pose JSON from extract_pose.py")
    parser.add_argument("--holds",  default="holds.json",         help="Holds JSON from generate_step_1.py")
    parser.add_argument("--com",    default="com.json",           help="CoM JSON from generate_step_1.py")
    parser.add_argument("--output", default="pose_overlay.mp4",   help="Output video path")
    args = parser.parse_args()

    reconstruct(args.video, args.pose, args.holds, args.com, args.output)
