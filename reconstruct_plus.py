"""
reconstruct_plus.py
-------------------
Side-by-side video:

  LEFT  panel — skeleton + hold markers with weight circles (vertical load, colour-coded)
  RIGHT panel — skeleton + limb segments colour-coded by axial force + joint angles at
                knees and elbows

Colour scales:
  Weight circles  : green (low) → red (≥ MAX_WEIGHT_PCT % BW)
  Limb segments   : green (low axial) → red (≥ MAX_SEGMENT_PCT % BW)
  Joint angle dot : green (straight, 180°) → red (acutely bent, ≤ 90°)

Requirements:
    pip install opencv-python numpy tqdm

Usage:
    python reconstruct_plus.py
    python reconstruct_plus.py --video climbing_video.mov --output climbing_plus.mp4
"""

import argparse
import json
import sys
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

# ── Colour scale parameters ───────────────────────────────────────────────────
MAX_WEIGHT_PCT  = 50.0   # % BW → full red on left-panel weight circles
MAX_SEGMENT_PCT = 80.0   # % BW → full red on right-panel segment lines

# ── Motion indicator scales ───────────────────────────────────────────────────
MAX_SPEED_DISPLAY = 300.0    # px/s → fills velocity bar completely
MAX_ACCEL_DISPLAY = 1000.0   # px/s² → fills acceleration bar completely
MAX_ARROW_PX      = 110      # max arrow length on schematic (pixels)

# ── Temporal smoothing ────────────────────────────────────────────────────────
# Exponential moving average applied to all numeric display values before rendering.
# alpha=1.0 → no smoothing (raw values); alpha=0.1 → heavy smoothing.
SMOOTHING_ALPHA = 0.5


# ── Skeleton connections ──────────────────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,23),(12,24),(23,24),
    (11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (23,25),(25,27),(27,29),(27,31),(29,31),
    (24,26),(26,28),(28,30),(28,32),(30,32),
]

LANDMARK_NAMES = [
    "NOSE","LEFT_EYE_INNER","LEFT_EYE","LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER","RIGHT_EYE","RIGHT_EYE_OUTER",
    "LEFT_EAR","RIGHT_EAR","MOUTH_LEFT","MOUTH_RIGHT",
    "LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW",
    "LEFT_WRIST","RIGHT_WRIST","LEFT_PINKY","RIGHT_PINKY",
    "LEFT_INDEX","RIGHT_INDEX","LEFT_THUMB","RIGHT_THUMB",
    "LEFT_HIP","RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE",
    "LEFT_ANKLE","RIGHT_ANKLE","LEFT_HEEL","RIGHT_HEEL",
    "LEFT_FOOT_INDEX","RIGHT_FOOT_INDEX",
]
LM_IDX = {name: i for i, name in enumerate(LANDMARK_NAMES)}

COLOUR_SKELETON = (0, 200, 80)
COLOUR_BODY     = (0, 220, 100)
COLOUR_FACE     = (200, 200, 200)
COLOUR_HANDS    = (0, 160, 255)
COLOUR_FEET     = (255, 80,  80)
HAND_INDICES    = {15,16,17,18,19,20,21,22}
FOOT_INDICES    = {27,28,29,30,31,32}
FACE_INDICES    = set(range(11))
VISIBILITY_THRESHOLD = 0.5

# ── Hold / extremity mappings ─────────────────────────────────────────────────
HOLD_LABELS = {
    ("hand","left"):  "LH", ("hand","right"): "RH",
    ("foot","left"):  "LF", ("foot","right"): "RF",
}
HOLD_TO_EXT = {
    ("hand","left"):  "left_hand",  ("hand","right"): "right_hand",
    ("foot","left"):  "left_foot",  ("foot","right"): "right_foot",
}

# ── Limb segments to colour on right panel ────────────────────────────────────
# (segment_key_in_json, distal_landmark, proximal_landmark)
DRAW_SEGMENTS = [
    ("left_lower_leg",  "LEFT_ANKLE",   "LEFT_KNEE"),
    ("left_upper_leg",  "LEFT_KNEE",    "LEFT_HIP"),
    ("right_lower_leg", "RIGHT_ANKLE",  "RIGHT_KNEE"),
    ("right_upper_leg", "RIGHT_KNEE",   "RIGHT_HIP"),
    ("left_forearm",    "LEFT_WRIST",   "LEFT_ELBOW"),
    ("left_upper_arm",  "LEFT_ELBOW",   "LEFT_SHOULDER"),
    ("right_forearm",   "RIGHT_WRIST",  "RIGHT_ELBOW"),
    ("right_upper_arm", "RIGHT_ELBOW",  "RIGHT_SHOULDER"),
]

# Joint angle annotations on right panel
DRAW_JOINTS = [
    ("left_knee",  "LEFT_KNEE"),
    ("right_knee", "RIGHT_KNEE"),
    ("left_elbow", "LEFT_ELBOW"),
    ("right_elbow","RIGHT_ELBOW"),
]


# ── Colour helpers ────────────────────────────────────────────────────────────

def _hsv_colour(ratio: float) -> tuple:
    """ratio 0→1 maps green→yellow→red in BGR."""
    ratio = float(np.clip(ratio, 0.0, 1.0))
    hue = int(60 * (1.0 - ratio))          # 60=green, 0=red (OpenCV 0-180 hue)
    hsv = np.uint8([[[hue, 240, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def weight_colour(pct: float) -> tuple:
    return _hsv_colour(pct / MAX_WEIGHT_PCT)


def segment_colour(pct: float) -> tuple:
    return _hsv_colour(pct / MAX_SEGMENT_PCT)


def joint_angle_colour(angle_deg: float) -> tuple:
    """180°=straight=green, 90°=right-angle=red, <90°=red."""
    ratio = (180.0 - float(np.clip(angle_deg, 0.0, 180.0))) / 90.0
    return _hsv_colour(ratio)


# ── Skeleton / CoM drawing (shared) ──────────────────────────────────────────

def landmark_colour(idx):
    if idx in FACE_INDICES:  return COLOUR_FACE
    if idx in HAND_INDICES:  return COLOUR_HANDS
    if idx in FOOT_INDICES:  return COLOUR_FEET
    return COLOUR_BODY


def get_coords(landmarks, w, h):
    """Return {landmark_idx: (px, py)} for visible landmarks."""
    coords = {}
    for lm in landmarks:
        if lm["visibility"] >= VISIBILITY_THRESHOLD:
            idx = LM_IDX.get(lm["name"])
            if idx is not None:
                coords[idx] = (int(lm["x"] * w), int(lm["y"] * h))
    return coords


def draw_pose(frame, landmarks):
    h, w = frame.shape[:2]
    coords = get_coords(landmarks, w, h)
    for a, b in CONNECTIONS:
        if a in coords and b in coords:
            cv2.line(frame, coords[a], coords[b], COLOUR_SKELETON, 3, cv2.LINE_AA)
    for idx, pt in coords.items():
        c = landmark_colour(idx)
        cv2.circle(frame, pt, 6, c, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 6, (0, 0, 0), 1, cv2.LINE_AA)


def draw_com(frame, px, py):
    size = 18
    pts = np.array([[px,py-size],[px+size,py],[px,py+size],[px-size,py]], np.int32)
    cv2.fillPoly(frame, [pts], (0, 230, 255), cv2.LINE_AA)
    cv2.polylines(frame, [pts], True, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "CoM", (px+22, py+6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, "CoM", (px+22, py+6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,230,255), 1, cv2.LINE_AA)


# ── Left-panel drawing ────────────────────────────────────────────────────────

def draw_past_hold(frame, pt):
    cv2.circle(frame, pt, 14, (56, 56, 56), -1, cv2.LINE_AA)
    cv2.circle(frame, pt, 14, (0, 0, 0), 1, cv2.LINE_AA)


def draw_weight_schematic(panel, extremities, limb_segs, width, height, motion=None):
    """
    Stationary body schematic on a dark background.
    Limb segments are coloured green→red by axial force.
    Large circles at each extremity show vertical weight (kg + %).
    """
    panel[:] = (25, 25, 30)   # near-black background

    cx = width // 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    def px(frac): return int(cx + frac * width)
    def py(frac): return int(frac * height)

    # ── Joint positions (fractions of panel width/height) ─────────────────
    j = {
        'head':  (px(0),      py(0.06)),
        'l_sho': (px(-0.16),  py(0.17)),
        'r_sho': (px(+0.16),  py(0.17)),
        'l_elb': (px(-0.26),  py(0.29)),
        'r_elb': (px(+0.26),  py(0.29)),
        'l_wri': (px(-0.31),  py(0.41)),
        'r_wri': (px(+0.31),  py(0.41)),
        'l_hip': (px(-0.09),  py(0.49)),
        'r_hip': (px(+0.09),  py(0.49)),
        'l_kne': (px(-0.11),  py(0.65)),
        'r_kne': (px(+0.11),  py(0.65)),
        'l_ank': (px(-0.12),  py(0.83)),
        'r_ank': (px(+0.12),  py(0.83)),
    }

    # ── Torso / spine (neutral grey) ──────────────────────────────────────
    mid_sho = ((j['l_sho'][0]+j['r_sho'][0])//2, (j['l_sho'][1]+j['r_sho'][1])//2)
    mid_hip = ((j['l_hip'][0]+j['r_hip'][0])//2, (j['l_hip'][1]+j['r_hip'][1])//2)
    GREY = (80, 80, 85)
    cv2.line(panel, j['head'],  mid_sho, GREY,  8, cv2.LINE_AA)
    cv2.line(panel, mid_sho,    mid_hip, GREY, 14, cv2.LINE_AA)
    cv2.line(panel, j['l_sho'], j['r_sho'], GREY, 10, cv2.LINE_AA)
    cv2.line(panel, j['l_hip'], j['r_hip'], GREY, 10, cv2.LINE_AA)
    cv2.circle(panel, j['head'], 32, GREY, -1, cv2.LINE_AA)   # head

    # ── Coloured limb segments ────────────────────────────────────────────
    SCHEMATIC_SEGS = [
        ('left_upper_arm',  'l_sho', 'l_elb'),
        ('left_forearm',    'l_elb', 'l_wri'),
        ('right_upper_arm', 'r_sho', 'r_elb'),
        ('right_forearm',   'r_elb', 'r_wri'),
        ('left_upper_leg',  'l_hip', 'l_kne'),
        ('left_lower_leg',  'l_kne', 'l_ank'),
        ('right_upper_leg', 'r_hip', 'r_kne'),
        ('right_lower_leg', 'r_kne', 'r_ank'),
    ]
    for seg_key, j_from, j_to in SCHEMATIC_SEGS:
        seg    = limb_segs.get(seg_key)
        colour = segment_colour(seg['axial_pct']) if seg else (55, 55, 60)
        cv2.line(panel, j[j_from], j[j_to], colour, 22, cv2.LINE_AA)
        cv2.line(panel, j[j_from], j[j_to], (255, 255, 255), 2, cv2.LINE_AA)

    # Joint dots at elbows and knees
    for jname in ('l_elb', 'r_elb', 'l_kne', 'r_kne'):
        cv2.circle(panel, j[jname], 12, (160, 160, 165), -1, cv2.LINE_AA)

    # ── Extremity circles ─────────────────────────────────────────────────
    EXT_MAP = {
        'left_hand':  ('l_wri', 'LH'),
        'right_hand': ('r_wri', 'RH'),
        'left_foot':  ('l_ank', 'LF'),
        'right_foot': ('r_ank', 'RF'),
    }
    R = 75
    for ext_key, (jname, label) in EXT_MAP.items():
        pt   = j[jname]
        info = extremities.get(ext_key)
        if info:
            colour = weight_colour(info['vertical_pct'])
            cv2.circle(panel, pt, R, colour, -1, cv2.LINE_AA)
            cv2.circle(panel, pt, R, (0, 0, 0), 3, cv2.LINE_AA)
            for text, scale, y_off in [
                (f"{info['vertical_kg']:.1f}kg", 1.05, -14),
                (f"{info['vertical_pct']:.0f}%",  0.90,  24),
            ]:
                (tw, _), _ = cv2.getTextSize(text, font, scale, 2)
                tx, ty = pt[0] - tw // 2, pt[1] + y_off
                cv2.putText(panel, text, (tx, ty), font, scale, (0,0,0),     5, cv2.LINE_AA)
                cv2.putText(panel, text, (tx, ty), font, scale, (255,255,255), 2, cv2.LINE_AA)
        else:
            cv2.circle(panel, pt, R, (45, 45, 50), -1, cv2.LINE_AA)
            cv2.circle(panel, pt, R, (90, 90, 95),  2, cv2.LINE_AA)
            (lw, _), _ = cv2.getTextSize(label, font, 1.1, 2)
            cv2.putText(panel, label, (pt[0]-lw//2, pt[1]+10), font, 1.1, (90,90,95), 2, cv2.LINE_AA)

    # ── Motion indicators ──────────────────────────────────────────────────────
    if motion is not None:
        vx    = float(motion.get("vx_px_s",    0.0) or 0.0)
        vy_up = float(motion.get("vy_px_s",    0.0) or 0.0)   # upward positive
        speed = float(motion.get("speed_px_s", 0.0) or 0.0)
        accel = float(motion.get("accel_px_s2",0.0) or 0.0)

        BAR_W      = 38
        MARGIN     = 14
        BAR_TOP    = int(height * 0.04)
        BAR_BOTTOM = int(height * 0.96)
        bar_h      = BAR_BOTTOM - BAR_TOP
        center_y   = (BAR_TOP + BAR_BOTTOM) // 2
        DARK       = (38, 38, 42)
        BORDER     = (95, 95, 100)
        TEXT_COL   = (155, 155, 160)

        # ── Left bar: vertical velocity (up = green, down = blue-red) ─────────
        bx0, bx1 = MARGIN, MARGIN + BAR_W
        cv2.rectangle(panel, (bx0, BAR_TOP), (bx1, BAR_BOTTOM), DARK, -1)

        ratio_v = float(np.clip(abs(vy_up) / MAX_SPEED_DISPLAY, 0.0, 1.0))
        fill_v  = int(ratio_v * bar_h / 2)
        if vy_up >= 0:                            # moving up
            v_colour = (60, 200, 60)
            cv2.rectangle(panel, (bx0, center_y - fill_v), (bx1, center_y), v_colour, -1)
        else:                                     # moving down
            v_colour = (60, 60, 220)
            cv2.rectangle(panel, (bx0, center_y), (bx1, center_y + fill_v), v_colour, -1)

        cv2.line(panel, (bx0, center_y), (bx1, center_y), (200, 200, 200), 2)
        cv2.rectangle(panel, (bx0, BAR_TOP), (bx1, BAR_BOTTOM), BORDER, 1)

        # Tick marks at 25 / 50 / 75 %
        for frac in (0.25, 0.5, 0.75):
            ty = BAR_TOP + int(frac * bar_h)
            cv2.line(panel, (bx1, ty), (bx1 + 5, ty), BORDER, 1)

        cv2.putText(panel, "VEL",  (bx0, BAR_TOP - 10), font, 0.6, TEXT_COL, 1, cv2.LINE_AA)
        cv2.putText(panel, "UP",   (bx0, BAR_TOP + 22), font, 0.45, (80,80,85), 1, cv2.LINE_AA)
        cv2.putText(panel, "DN",   (bx0, BAR_BOTTOM - 10), font, 0.45, (80,80,85), 1, cv2.LINE_AA)
        val_v = f"{vy_up:+.0f}"
        (tw,_),_ = cv2.getTextSize(val_v, font, 0.55, 1)
        cv2.putText(panel, val_v, (bx0 + (BAR_W-tw)//2, BAR_BOTTOM + 24),
                    font, 0.55, TEXT_COL, 1, cv2.LINE_AA)
        cv2.putText(panel, "px/s", (bx0, BAR_BOTTOM + 44), font, 0.42, (100,100,100), 1, cv2.LINE_AA)

        # ── Right bar: acceleration magnitude ─────────────────────────────────
        bx0r = width - MARGIN - BAR_W
        bx1r = width - MARGIN
        cv2.rectangle(panel, (bx0r, BAR_TOP), (bx1r, BAR_BOTTOM), DARK, -1)

        ratio_a = float(np.clip(accel / MAX_ACCEL_DISPLAY, 0.0, 1.0))
        fill_a  = int(ratio_a * bar_h)
        a_colour = _hsv_colour(ratio_a)
        cv2.rectangle(panel, (bx0r, BAR_BOTTOM - fill_a), (bx1r, BAR_BOTTOM), a_colour, -1)
        cv2.rectangle(panel, (bx0r, BAR_TOP), (bx1r, BAR_BOTTOM), BORDER, 1)

        for frac in (0.25, 0.5, 0.75):
            ty = BAR_TOP + int(frac * bar_h)
            cv2.line(panel, (bx0r - 5, ty), (bx0r, ty), BORDER, 1)

        cv2.putText(panel, "ACC",  (bx0r, BAR_TOP - 10), font, 0.6, TEXT_COL, 1, cv2.LINE_AA)
        val_a = f"{accel:.0f}"
        (tw,_),_ = cv2.getTextSize(val_a, font, 0.55, 1)
        cv2.putText(panel, val_a, (bx0r + (BAR_W-tw)//2, BAR_BOTTOM + 24),
                    font, 0.55, TEXT_COL, 1, cv2.LINE_AA)
        cv2.putText(panel, "px/s2", (bx0r - 4, BAR_BOTTOM + 44), font, 0.42, (100,100,100), 1, cv2.LINE_AA)

        # ── Velocity arrow at torso centre ────────────────────────────────────
        if speed > 2.0:
            torso_x = int(width * 0.5)
            torso_y = int(height * 0.33)   # midpoint between shoulders and hips

            mag   = max(speed, 1e-6)
            dx_n  =  vx    / mag
            dy_n  = -vy_up / mag            # image y is inverted

            arrow_len = int(np.clip(speed / MAX_SPEED_DISPLAY, 0.0, 1.0) * MAX_ARROW_PX)
            if arrow_len > 6:
                ex = int(torso_x + dx_n * arrow_len)
                ey = int(torso_y + dy_n * arrow_len)
                a_col = _hsv_colour(speed / MAX_SPEED_DISPLAY)
                cv2.arrowedLine(panel, (torso_x, torso_y), (ex, ey),
                                (0, 0, 0), 9, cv2.LINE_AA, tipLength=0.38)
                cv2.arrowedLine(panel, (torso_x, torso_y), (ex, ey),
                                a_col, 4, cv2.LINE_AA, tipLength=0.38)


# ── Right-panel drawing ───────────────────────────────────────────────────────

def draw_segment_forces(frame, landmarks, seg_data: dict):
    """
    Draw each limb segment as a thick coloured line whose colour reflects
    the axial force through that bone.  Drawn on top of the skeleton.
    """
    h, w = frame.shape[:2]
    coords = get_coords(landmarks, w, h)

    for seg_name, distal_lm, proximal_lm in DRAW_SEGMENTS:
        d_idx = LM_IDX.get(distal_lm)
        p_idx = LM_IDX.get(proximal_lm)
        if d_idx not in coords or p_idx not in coords:
            continue

        seg = seg_data.get(seg_name)
        if seg is None:
            colour = (120, 120, 120)   # grey — no data
        else:
            colour = segment_colour(seg["axial_pct"])

        cv2.line(frame, coords[d_idx], coords[p_idx], colour, 12, cv2.LINE_AA)
        # White centre stripe so lines remain readable on any background
        cv2.line(frame, coords[d_idx], coords[p_idx], (255, 255, 255), 2, cv2.LINE_AA)


def draw_joint_angles(frame, landmarks, joint_data: dict):
    """
    Draw a large coloured dot at each knee/elbow with the joint angle
    printed inside. Colour reflects how bent the joint is.
    """
    h, w = frame.shape[:2]
    coords = get_coords(landmarks, w, h)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for joint_name, lm_name in DRAW_JOINTS:
        idx = LM_IDX.get(lm_name)
        if idx not in coords:
            continue

        pt    = coords[idx]
        angle = joint_data.get(joint_name)

        if angle is None:
            colour = (120, 120, 120)
            text   = "—"
        else:
            colour = joint_angle_colour(angle)
            text   = f"{angle:.0f}deg"

        R = 19
        cv2.circle(frame, pt, R, colour, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, R, (0, 0, 0), 2, cv2.LINE_AA)


def draw_active_hold_small(frame, pt, label, vertical_pct):
    """Compact weight circle for the right panel."""
    colour = weight_colour(vertical_pct)
    R = 20
    cv2.circle(frame, pt, R, colour, -1, cv2.LINE_AA)
    cv2.circle(frame, pt, R, (0, 0, 0), 2, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    (lw, _), _ = cv2.getTextSize(label, font, 0.7, 2)
    cv2.putText(frame, label, (pt[0]-lw//2, pt[1]-R-8), font, 0.7, (0,0,0),  3, cv2.LINE_AA)
    cv2.putText(frame, label, (pt[0]-lw//2, pt[1]-R-8), font, 0.7, colour,   1, cv2.LINE_AA)


def draw_panel_label(frame, text, x=20, y=55):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,0,0),     5, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,255,255), 2, cv2.LINE_AA)


# ── Temporal smoothing ───────────────────────────────────────────────────────

def smooth_weight_data(weight_by_frame: dict, alpha: float) -> dict:
    """
    Apply exponential moving average across frames to all numeric display values:
      - extremity vertical_kg / vertical_pct  (left-panel circles)
      - limb segment axial_pct               (right-panel segment colours)
      - joint angles                         (right-panel angle labels)

    For frames with no data the EMA state is preserved but nothing is written,
    so the next detected frame inherits the smoothed history rather than jumping.
    """
    smoothed = {}

    # EMA state
    ema_ext   = {}   # {ext_name: {"vertical_pct": float, "vertical_kg": float}}
    ema_seg   = {}   # {seg_name:  {"axial_pct": float}}
    ema_joint = {}   # {joint_name: float}

    def ema(prev, new):
        return alpha * new + (1.0 - alpha) * prev if prev is not None else new

    for fi in sorted(weight_by_frame.keys()):
        rec = weight_by_frame[fi]

        # ── Extremity weights ─────────────────────────────────────────────────
        new_ext = {}
        for ext_name, info in rec.get("extremities", {}).items():
            prev = ema_ext.get(ext_name, {})
            s_pct = ema(prev.get("vertical_pct"), info["vertical_pct"])
            s_kg  = ema(prev.get("vertical_kg"),  info["vertical_kg"])
            ema_ext[ext_name] = {"vertical_pct": s_pct, "vertical_kg": s_kg}
            new_ext[ext_name] = {**info, "vertical_pct": s_pct, "vertical_kg": s_kg}

        # ── Limb segments ─────────────────────────────────────────────────────
        new_seg = {}
        for seg_name, seg in rec.get("limb_segments", {}).items():
            if seg is None:
                new_seg[seg_name] = None
                continue
            if seg.get("axial_pct") is None:
                new_seg[seg_name] = seg
                continue
            prev_pct = ema_seg.get(seg_name, {}).get("axial_pct")
            s_pct = ema(prev_pct, seg["axial_pct"])
            # Recompute kg from smoothed pct (body_weight = axial_kg / (axial_pct/100))
            bw = seg["axial_kg"] / (seg["axial_pct"] / 100.0) if seg["axial_pct"] != 0 else 70.0
            ema_seg[seg_name] = {"axial_pct": s_pct}
            new_seg[seg_name] = {**seg, "axial_pct": s_pct, "axial_kg": s_pct / 100.0 * bw}

        # ── Joint angles ──────────────────────────────────────────────────────
        new_joint = {}
        for jname, angle in rec.get("joint_angles", {}).items():
            if angle is None:
                new_joint[jname] = None
                continue
            s_angle = ema(ema_joint.get(jname), angle)
            ema_joint[jname] = s_angle
            new_joint[jname] = s_angle

        smoothed[fi] = {
            "extremities":   new_ext,
            "limb_segments": new_seg,
            "joint_angles":  new_joint,
        }

    return smoothed


# ── Pre-build per-frame lookups ───────────────────────────────────────────────

def build_holds_by_frame(holds_data, total_frames):
    past_map   = defaultdict(list)
    active_map = defaultdict(list)
    for kind, key in [("hand","handholds"),("foot","footholds")]:
        for hold in holds_data.get(key, []):
            side, start, end = hold["side"], hold["start_frame"], hold["end_frame"]
            for fi in range(start, min(end+1, total_frames)):
                active_map[fi].append((hold, kind, side))
            for fi in range(end+1, total_frames):
                past_map[fi].append((hold, kind, side))
    return {"past": dict(past_map), "active": dict(active_map)}


def build_weight_by_frame(weight_data):
    out = {}
    for rec in weight_data.get("per_frame", []):
        fi = rec["frame"]
        out[fi] = {
            "extremities": {
                ext: rec[ext]
                for ext in ("left_hand","right_hand","left_foot","right_foot")
                if rec.get(ext) and rec[ext].get("active")
            },
            "limb_segments": rec.get("limb_segments", {}),
            "joint_angles":  rec.get("joint_angles",  {}),
        }
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def reconstruct_plus(video_path, pose_path, holds_path, com_path, weight_path, motion_path, output_path):
    import os
    print("Loading data …")
    with open(pose_path)   as f: pose_data   = json.load(f)
    with open(holds_path)  as f: holds_data  = json.load(f)
    with open(com_path)    as f: com_data    = json.load(f)
    with open(weight_path) as f: weight_data = json.load(f)

    # com_motion.json is optional — generated by stats2.py
    motion_by_frame = {}
    if motion_path and os.path.exists(motion_path):
        with open(motion_path) as f:
            motion_data = json.load(f)
        motion_by_frame = {
            e["frame"]: e
            for e in motion_data.get("per_frame", [])
            if e.get("speed_px_s") is not None
        }
        print(f"Loaded motion data: {len(motion_by_frame)} frames")
    else:
        print("No com_motion.json found — velocity/acceleration indicators disabled")

    frames_pose  = {e["frame"]: e for e in pose_data["landmarks"]}
    com_by_frame = {e["frame"]: e for e in com_data["com_per_frame"] if e["px"] is not None}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Source: {width}×{height} @ {fps:.2f} fps  ({total} frames)")
    print(f"Output: {width*2}×{height} (side-by-side)")

    holds_by_frame  = build_holds_by_frame(holds_data, total)
    weight_by_frame = smooth_weight_data(build_weight_by_frame(weight_data), SMOOTHING_ALPHA)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    if not out_writer.isOpened():
        sys.exit(f"Cannot create output: {output_path}")

    with tqdm(total=total, unit="frame", desc="Rendering") as pbar:
        fi = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Copy raw frame for right panel BEFORE left-panel drawing
            frame_right = frame.copy()

            w_rec        = weight_by_frame.get(fi, {})
            extremities  = w_rec.get("extremities",  {})
            limb_segs    = w_rec.get("limb_segments", {})
            joint_angles = w_rec.get("joint_angles",  {})
            pose_entry   = frames_pose.get(fi)
            com          = com_by_frame.get(fi)

            # ── LEFT PANEL — weight distribution schematic ────────────────────
            motion = motion_by_frame.get(fi)
            draw_weight_schematic(frame, extremities, limb_segs, width, height, motion)

            # ── RIGHT PANEL ───────────────────────────────────────────────────

            # Past holds (same faded dots for context)
            for hold, kind, side in holds_by_frame.get("past", {}).get(fi, []):
                draw_past_hold(frame_right, (hold["px"], hold["py"]))

            # Skeleton base layer
            if pose_entry and pose_entry["detected"] and pose_entry["landmarks"]:
                draw_pose(frame_right, pose_entry["landmarks"])

                # Coloured limb segments on top of skeleton
                draw_segment_forces(frame_right, pose_entry["landmarks"], limb_segs)

                # Joint angle circles on top of segments
                draw_joint_angles(frame_right, pose_entry["landmarks"], joint_angles)

            # CoM
            if com:
                draw_com(frame_right, com["px"], com["py"])

            # Small weight circles at hold positions (on top of everything)
            for hold, kind, side in holds_by_frame.get("active", {}).get(fi, []):
                ext    = HOLD_TO_EXT[(kind, side)]
                label  = HOLD_LABELS[(kind, side)]
                pt     = (hold["px"], hold["py"])
                w_info = extremities.get(ext)
                if w_info:
                    draw_active_hold_small(frame_right, pt, label, w_info["vertical_pct"])
                else:
                    cv2.circle(frame_right, pt, 40, (180, 180, 180), 2, cv2.LINE_AA)

            draw_panel_label(frame_right, f"{fi/fps:.2f}s")

            # ── Combine and write ─────────────────────────────────────────────
            combined = np.hstack([frame, frame_right])
            out_writer.write(combined)

            fi += 1
            pbar.update(1)

    cap.release()
    out_writer.release()
    print(f"\nSaved → {output_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Side-by-side climbing analysis video.")
    parser.add_argument("--video",  default="climbing_video.mov",    help="Source video")
    parser.add_argument("--pose",   default="pose_data.json",        help="Pose data")
    parser.add_argument("--holds",  default="holds.json",            help="Holds data")
    parser.add_argument("--com",    default="com.json",              help="CoM data")
    parser.add_argument("--weight", default="weight_per_frame.json", help="Per-frame weight data")
    parser.add_argument("--motion", default="com_motion.json",       help="CoM motion data from stats2.py")
    parser.add_argument("--output", default="climbing_plus.mp4",     help="Output video")
    args = parser.parse_args()

    reconstruct_plus(args.video, args.pose, args.holds, args.com, args.weight, args.motion, args.output)
