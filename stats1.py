"""
stats1.py
---------
Estimates the fraction of body weight carried by each extremity (left hand,
right hand, left foot, right foot) at every frame.

Physical model
--------------
The COM is connected to each active contact point by a rigid, non-extensible
rod that can only carry axial force (along the rod).  Static equilibrium
requires the vector sum of all contact forces to equal the body weight acting
downward.  This gives a 2×N linear system:

    D · f = g           (D: 2×N direction matrix, f: N force fractions, g: unit gravity)

For N=2 the system is exactly determined; for N>2 it is underdetermined and
the minimum-norm (pseudo-inverse) solution is used, which minimises the sum of
squared forces — equivalent to the principle of minimum muscular effort.

A contact point is considered active only when the corresponding hold is
active at that frame (from holds.json).

Requirements:
    pip install numpy
"""

import json
import os
import numpy as np

# ── Parameters ────────────────────────────────────────────────────────────────

HOLDS_PATH   = "holds.json"              # from generate_step_1.py
COM_PATH     = "com.json"               # from generate_step_1.py
POSE_PATH    = "pose_data.json"         # from extract_pose.py
OUTPUT_PATH  = "weight_per_frame.json"  # per-frame weight output

BODY_WEIGHT_KG = 70.0             # climber's total body weight in kg

# Upward direction in image coordinates (x right, y down).
# Upward = negative y direction.
UPWARD = np.array([0.0, -1.0])

# ─────────────────────────────────────────────────────────────────────────────

EXTREMITY_LANDMARK = {
    "left_hand":  "LEFT_WRIST",
    "right_hand": "RIGHT_WRIST",
    "left_foot":  "LEFT_ANKLE",
    "right_foot": "RIGHT_ANKLE",
}

EXTREMITY_NAMES = ["left_hand", "right_hand", "left_foot", "right_foot"]

# ── Limb segment definitions ──────────────────────────────────────────────────
# Each entry: (segment_name, distal_landmark, proximal_landmark, force_extremity)
# Force travels from the contact point (distal) up toward the body (proximal).
# Assumption: vertical force at distal == vertical force at proximal
# (i.e. the limb segment carries no additional external load between joints).
LIMB_SEGMENTS = [
    ("left_lower_leg",  "LEFT_ANKLE",    "LEFT_KNEE",      "left_foot"),
    ("left_upper_leg",  "LEFT_KNEE",     "LEFT_HIP",       "left_foot"),
    ("right_lower_leg", "RIGHT_ANKLE",   "RIGHT_KNEE",     "right_foot"),
    ("right_upper_leg", "RIGHT_KNEE",    "RIGHT_HIP",      "right_foot"),
    ("left_forearm",    "LEFT_WRIST",    "LEFT_ELBOW",     "left_hand"),
    ("left_upper_arm",  "LEFT_ELBOW",    "LEFT_SHOULDER",  "left_hand"),
    ("right_forearm",   "RIGHT_WRIST",   "RIGHT_ELBOW",    "right_hand"),
    ("right_upper_arm", "RIGHT_ELBOW",   "RIGHT_SHOULDER", "right_hand"),
]

# Joint angle definitions: (name, distal_landmark, joint_landmark, proximal_landmark)
# Joint angle convention: 180° = straight limb, 90° = right-angle bend, 0° = fully folded
JOINT_DEFS = [
    ("left_knee",  "LEFT_ANKLE",  "LEFT_KNEE",  "LEFT_HIP"),
    ("right_knee", "RIGHT_ANKLE", "RIGHT_KNEE", "RIGHT_HIP"),
    ("left_elbow", "LEFT_WRIST",  "LEFT_ELBOW", "LEFT_SHOULDER"),
    ("right_elbow","RIGHT_WRIST", "RIGHT_ELBOW","RIGHT_SHOULDER"),
]

SEGMENT_NAMES = [s[0] for s in LIMB_SEGMENTS]
JOINT_NAMES   = [j[0] for j in JOINT_DEFS]


# ── Force distribution ────────────────────────────────────────────────────────

def distribute_weight(com_xy: np.ndarray, contacts: dict[str, np.ndarray]) -> dict[str, float]:
    """
    Given COM position and active contact positions (all in normalised coords),
    return the force at each contact as a fraction of total body weight.

    Sign convention  (image space: y increases downward):
      +  contact pushes/pulls COM upward  (away from ground)  ← desired for most holds
      -  contact pushes COM downward      (toward ground)

    Direction model:
      Contacts ABOVE the COM (y_contact < y_COM) act in tension — the rod pulls
      the COM toward the hold, so the force direction is (contact → COM direction
      reversed) = (contact - COM)/norm, which points upward.

      Contacts BELOW the COM (y_contact > y_COM) act in compression — the rod
      pushes the COM away from the hold, so the force direction is (COM - contact)/norm,
      which also points upward.

      Unified: direction_i = sign(y_COM - y_contact) * (contact - COM) / norm
      This always points upward for load-bearing contacts, so positive f_i = upward force.

    Uses numpy lstsq (SVD pseudoinverse) for the minimum-norm solution when
    the system is underdetermined (N > 2 contacts).
    """
    names = list(contacts.keys())
    N = len(names)

    if N == 0:
        return {}

    # Build 2×N direction matrix — each column points upward for load-bearing contacts
    D = np.zeros((2, N))
    for i, name in enumerate(names):
        vec = contacts[name] - com_xy
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            s = np.sign(com_xy[1] - contacts[name][1])  # +1 above, -1 below
            if s == 0:
                s = 1.0
            D[:, i] = s * vec / norm

    # Solve D @ f = UPWARD  →  f_i is the axial force along each rod (fraction of BW)
    f = np.linalg.lstsq(D, UPWARD, rcond=None)[0]

    # Return both axial and vertical components for each contact.
    # Vertical component = f_i × |d_iy|  (projection onto the gravity axis).
    # These vertical components always sum to 1.0 per frame by construction.
    # Sign convention: positive = supports COM upward, negative = pushes COM down.
    result = {}
    for i, name in enumerate(names):
        axial    = float(f[i])
        vertical = float(f[i] * abs(D[1, i]))
        if f[i] < 0:
            vertical = -abs(vertical)
        result[name] = {"axial": axial, "vertical": vertical}

    return result


# ── Limb segment / joint analysis ────────────────────────────────────────────

def get_lm_map(frame_entry: dict, min_vis: float = 0.3) -> dict[str, np.ndarray]:
    """Extract {landmark_name: np.array([x, y])} from a frame entry."""
    if not frame_entry["detected"] or not frame_entry["landmarks"]:
        return {}
    return {
        lm["name"]: np.array([lm["x"], lm["y"]])
        for lm in frame_entry["landmarks"]
        if lm["visibility"] >= min_vis
    }


def compute_limb_forces(lm_map: dict, vert_fracs: dict[str, float]) -> dict:
    """
    For each limb segment, compute:
      - axial_frac : axial bone force as a fraction of body weight
                     = vert_frac / cos(θ)   where θ = segment angle from vertical
      - angle_deg  : segment angle from vertical (0° = vertical, 90° = horizontal)

    A larger angle (more bent joint) → smaller cos(θ) → larger axial force
    for the same vertical load.  A straight limb (θ≈0) carries the minimum force.

    Also computes the anatomical joint angle at each knee/elbow
    (180° = straight, 90° = right-angle bend).

    Returns a flat dict with keys like "left_lower_leg", "left_knee", etc.
    Missing landmarks or inactive extremities produce None for that entry.
    """
    out = {}

    # ── Segment axial forces ──────────────────────────────────────────────────
    for seg_name, distal_lm, proximal_lm, ext_name in LIMB_SEGMENTS:
        if distal_lm not in lm_map or proximal_lm not in lm_map:
            out[seg_name] = None
            continue

        vert_frac = vert_fracs.get(ext_name)       # vertical load fraction at the contact
        if vert_frac is None:
            out[seg_name] = None
            continue

        vec      = lm_map[proximal_lm] - lm_map[distal_lm]   # distal → proximal
        seg_len  = np.linalg.norm(vec)
        if seg_len < 1e-6:
            out[seg_name] = None
            continue

        vert_extent = abs(vec[1])       # |Δy| = vertical span of the segment

        if vert_extent < seg_len * 0.01:    # segment nearly horizontal → clamp
            out[seg_name] = {"axial_frac": None, "angle_deg": round(float(np.degrees(np.arccos(0.01))), 1)}
            continue

        cos_theta = vert_extent / seg_len                          # cos of angle from vertical
        angle_deg = float(np.degrees(np.arccos(np.clip(cos_theta, 0.0, 1.0))))
        axial_frac = vert_frac / cos_theta                         # amplified by 1/cos

        out[seg_name] = {
            "axial_frac": round(float(axial_frac), 6),
            "angle_deg":  round(angle_deg, 2),
        }

    # ── Joint angles ──────────────────────────────────────────────────────────
    for joint_name, distal_lm, joint_lm, proximal_lm in JOINT_DEFS:
        if any(lm not in lm_map for lm in (distal_lm, joint_lm, proximal_lm)):
            out[joint_name] = None
            continue

        v_distal   = lm_map[distal_lm]   - lm_map[joint_lm]  # toward foot/hand
        v_proximal = lm_map[proximal_lm] - lm_map[joint_lm]  # toward hip/shoulder

        n1 = np.linalg.norm(v_distal)
        n2 = np.linalg.norm(v_proximal)
        if n1 < 1e-6 or n2 < 1e-6:
            out[joint_name] = None
            continue

        cos_a = np.dot(v_distal / n1, v_proximal / n2)
        out[joint_name] = round(float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))), 2)

    return out


# ── Data loading helpers ──────────────────────────────────────────────────────

def load_active_holds_by_frame(holds_data: dict, total_frames: int) -> dict[int, dict[str, tuple]]:
    """
    Returns {frame_idx: {extremity_name: (x, y)}} for every frame.
    (x, y) is the averaged hold position in normalised coords.
    """
    # Map hold kind+side → extremity name
    kind_side_to_name = {
        ("hand", "left"):  "left_hand",
        ("hand", "right"): "right_hand",
        ("foot", "left"):  "left_foot",
        ("foot", "right"): "right_foot",
    }

    active: dict[int, dict[str, tuple]] = {i: {} for i in range(total_frames)}

    for kind, key in [("hand", "handholds"), ("foot", "footholds")]:
        for hold in holds_data.get(key, []):
            ext_name = kind_side_to_name[(kind, hold["side"])]
            for fi in range(hold["start_frame"], min(hold["end_frame"] + 1, total_frames)):
                active[fi][ext_name] = (hold["x"], hold["y"])

    return active


def load_pose_positions(pose_data: dict) -> dict[int, dict[str, tuple]]:
    """
    Returns {frame_idx: {extremity_name: (x, y)}} using instantaneous
    wrist / ankle positions from the pose data.
    """
    lm_name_to_ext = {v: k for k, v in EXTREMITY_LANDMARK.items()}
    result = {}
    for entry in pose_data["landmarks"]:
        fi = entry["frame"]
        positions = {}
        if entry["detected"] and entry["landmarks"]:
            for lm in entry["landmarks"]:
                if lm["name"] in lm_name_to_ext and lm["visibility"] >= 0.3:
                    positions[lm_name_to_ext[lm["name"]]] = (lm["x"], lm["y"])
        result[fi] = positions
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    print("Loading data …")
    with open(HOLDS_PATH)  as f: holds_data = json.load(f)
    with open(COM_PATH)    as f: com_data   = json.load(f)
    with open(POSE_PATH)   as f: pose_data  = json.load(f)

    fps          = pose_data["video"]["fps"]
    total_frames = pose_data["video"]["total_frames"]

    com_by_frame = {
        e["frame"]: np.array([e["x"], e["y"]])
        for e in com_data["com_per_frame"]
        if e["x"] is not None
    }

    active_holds_by_frame = load_active_holds_by_frame(holds_data, total_frames)
    pose_positions        = load_pose_positions(pose_data)

    # Pre-build full landmark map per frame for joint/segment analysis
    frames_lm = {entry["frame"]: get_lm_map(entry) for entry in pose_data["landmarks"]}

    # ── Per-frame force calculation ───────────────────────────────────────────
    axial_sum    = {name: 0.0 for name in EXTREMITY_NAMES}
    vertical_sum = {name: 0.0 for name in EXTREMITY_NAMES}
    frame_count  = {name: 0   for name in EXTREMITY_NAMES}
    frames_with_any_contact = 0
    per_frame_records = []

    # Limb segment / joint accumulators
    seg_axial_sum   = {s: 0.0 for s in SEGMENT_NAMES}
    seg_axial_count = {s: 0   for s in SEGMENT_NAMES}
    seg_angle_sum   = {s: 0.0 for s in SEGMENT_NAMES}
    seg_angle_count = {s: 0   for s in SEGMENT_NAMES}
    joint_angle_sum  = {j: 0.0 for j in JOINT_NAMES}
    joint_angle_count= {j: 0   for j in JOINT_NAMES}

    for fi in range(total_frames):
        com = com_by_frame.get(fi)
        if com is None:
            continue

        active_holds = active_holds_by_frame.get(fi, {})
        if not active_holds:
            continue

        contacts: dict[str, np.ndarray] = {}
        for ext_name, (hx, hy) in active_holds.items():
            contacts[ext_name] = np.array([hx, hy])

        forces = distribute_weight(com, contacts)
        if not forces:
            continue

        frames_with_any_contact += 1

        record = {
            "frame":       fi,
            "timestamp_s": round(fi / fps, 4),
        }
        for name in EXTREMITY_NAMES:
            if name in forces:
                axial_kg    = forces[name]["axial"]    * BODY_WEIGHT_KG
                vertical_kg = forces[name]["vertical"] * BODY_WEIGHT_KG
                record[name] = {
                    "active":      True,
                    "axial_kg":    round(axial_kg,    3),
                    "axial_pct":   round(forces[name]["axial"]    * 100, 3),
                    "vertical_kg": round(vertical_kg, 3),
                    "vertical_pct":round(forces[name]["vertical"] * 100, 3),
                }
                axial_sum[name]    += forces[name]["axial"]
                vertical_sum[name] += forces[name]["vertical"]
                frame_count[name]  += 1
            else:
                record[name] = {
                    "active":       False,
                    "axial_kg":     None,
                    "axial_pct":    None,
                    "vertical_kg":  None,
                    "vertical_pct": None,
                }

        # ── Limb segment / joint forces ───────────────────────────────────────
        vert_fracs = {
            name: forces[name]["vertical"]
            for name in EXTREMITY_NAMES
            if name in forces
        }
        lm_map = frames_lm.get(fi, {})
        limb = compute_limb_forces(lm_map, vert_fracs)

        record["limb_segments"] = {}
        record["joint_angles"]  = {}

        for seg_name in SEGMENT_NAMES:
            seg = limb.get(seg_name)
            if seg and seg["axial_frac"] is not None:
                axial_kg = seg["axial_frac"] * BODY_WEIGHT_KG
                record["limb_segments"][seg_name] = {
                    "axial_kg":  round(axial_kg, 3),
                    "axial_pct": round(seg["axial_frac"] * 100, 3),
                    "angle_deg": seg["angle_deg"],
                }
                seg_axial_sum[seg_name]   += seg["axial_frac"]
                seg_axial_count[seg_name] += 1
                seg_angle_sum[seg_name]   += seg["angle_deg"]
                seg_angle_count[seg_name] += 1
            else:
                record["limb_segments"][seg_name] = None

        for joint_name in JOINT_NAMES:
            angle = limb.get(joint_name)
            record["joint_angles"][joint_name] = angle
            if angle is not None:
                joint_angle_sum[joint_name]   += angle
                joint_angle_count[joint_name] += 1

        per_frame_records.append(record)

    # ── Print results ─────────────────────────────────────────────────────────
    N = frames_with_any_contact
    W = BODY_WEIGHT_KG

    print(f"\n── Weight distribution  ({N} frames with active holds, {N/fps:.1f}s) ──")
    print(f"   Climber weight : {W} kg")
    print(f"   Video          : {total_frames} frames @ {fps:.1f} fps  ({total_frames/fps:.1f}s)\n")

    label_map = {
        "left_hand":  "Left  Hand",
        "right_hand": "Right Hand",
        "left_foot":  "Left  Foot",
        "right_foot": "Right Foot",
    }

    col = 12   # label column width

    # Header
    print(f"  {'':>{col}}  {'── Axial force ──':^34}  {'── Vertical force ──':^34}  active")
    print(f"  {'':>{col}}  {'when active':>16}  {'over climb':>16}  {'when active':>16}  {'over climb':>16}")
    print(f"  {'':->{col}}  {'':->16}  {'':->16}  {'':->16}  {'':->16}  {'':->8}")

    v_total_climb = 0.0
    for name in EXTREMITY_NAMES:
        lbl    = label_map[name]
        count  = frame_count[name]
        active_s = f"{count/fps:.1f}s" if count > 0 else "—"

        if count > 0:
            ax_when  = axial_sum[name]    / count * W
            ax_climb = axial_sum[name]    / N     * W
            vt_when  = vertical_sum[name] / count * W
            vt_climb = vertical_sum[name] / N     * W
            v_total_climb += vertical_sum[name] / N
        else:
            ax_when = ax_climb = vt_when = vt_climb = 0.0

        def fmt(kg):
            return f"{kg:5.1f}kg ({kg/W*100:5.1f}%)"

        print(f"  {lbl:<{col}}  {fmt(ax_when):>16}  {fmt(ax_climb):>16}  {fmt(vt_when):>16}  {fmt(vt_climb):>16}  {active_s:>8}")

    print(f"\n  Vertical force sum (over climb): {v_total_climb*W:.1f} kg  ({v_total_climb*100:.1f}% of BW)")
    print(f"  Gap to 100% = frames with no active holds\n")

    # ── Limb segment forces ───────────────────────────────────────────────────
    print("── Limb segment axial forces (avg when active) ──────────────────────────")
    print(f"  A higher axial force than the vertical load means the segment is angled —")
    print(f"  the more bent the joint, the greater the amplification.\n")

    seg_label = {
        "left_lower_leg":  "L lower leg  (tibia)",
        "left_upper_leg":  "L upper leg  (femur)",
        "right_lower_leg": "R lower leg  (tibia)",
        "right_upper_leg": "R upper leg  (femur)",
        "left_forearm":    "L forearm",
        "right_forearm":   "R forearm",
        "left_upper_arm":  "L upper arm",
        "right_upper_arm": "R upper arm",
    }

    print(f"  {'Segment':<26}  {'Axial force':>14}  {'Seg angle':>10}  {'Frames':>8}")
    print(f"  {'':-<26}  {'':->14}  {'':->10}  {'':->8}")
    for seg_name in SEGMENT_NAMES:
        cnt = seg_axial_count[seg_name]
        if cnt > 0:
            avg_axial_kg  = seg_axial_sum[seg_name]  / cnt * W
            avg_angle_deg = seg_angle_sum[seg_name]  / cnt
            print(f"  {seg_label[seg_name]:<26}  "
                  f"{avg_axial_kg:6.1f}kg ({avg_axial_kg/W*100:5.1f}%)  "
                  f"{avg_angle_deg:8.1f}°  "
                  f"{cnt/fps:6.1f}s")
        else:
            print(f"  {seg_label[seg_name]:<26}  {'—':>14}  {'—':>10}  {'—':>8}")

    print(f"\n── Joint angles (avg when active) ───────────────────────────────────────")
    print(f"  180° = straight limb  |  90° = right-angle bend  |  <90° = acute bend\n")
    joint_label = {
        "left_knee":  "Left  knee",
        "right_knee": "Right knee",
        "left_elbow": "Left  elbow",
        "right_elbow":"Right elbow",
    }
    for joint_name in JOINT_NAMES:
        cnt = joint_angle_count[joint_name]
        if cnt > 0:
            avg_angle = joint_angle_sum[joint_name] / cnt
            print(f"  {joint_label[joint_name]:<14}  avg {avg_angle:6.1f}°  ({cnt/fps:.1f}s valid)")
        else:
            print(f"  {joint_label[joint_name]:<14}  —")
    print()

    # ── Save per-frame data ───────────────────────────────────────────────────
    output = {
        "video":            pose_data["video"],
        "body_weight_kg":   BODY_WEIGHT_KG,
        "frames_with_contact": frames_with_any_contact,
        "per_frame":        per_frame_records,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
    print(f"Per-frame data saved → {OUTPUT_PATH}  ({size_mb:.1f} MB)\n")


if __name__ == "__main__":
    run()
