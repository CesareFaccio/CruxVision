"""
extract_pose_VITPose.py
-----------------------
Drop-in replacement for extract_pose.py using RTMPose via rtmlib.

Top-down pipeline per frame:
  1. RTMDet-nano detects the person bounding box
  2. RTMPose-x estimates 17 COCO keypoints within that box
  3. Keypoints are mapped to the same 33-landmark MediaPipe format so that
     generate_step_1.py, stats1.py, and reconstruct_plus.py work unchanged.

Landmarks not present in COCO-17 (finger details, inner eye corners, toes, heels)
are written with visibility=0.  None of the downstream scripts use those landmarks
for any calculation — they only use shoulders, elbows, wrists, hips, knees, ankles,
and the nose, all of which are covered by COCO-17.

Why RTMPose over MediaPipe:
  - Top-down design handles occlusion (rope, harness, chalk bag) better
  - RTMPose-x scores 75+ AP on COCO vs MediaPipe's lighter single-stage model
  - Runs via ONNX runtime — no mmcv dependency, installs in seconds

Installation:
    pip install rtmlib opencv-python tqdm numpy onnxruntime

Model weights are downloaded automatically on the first run (~50 MB).

Usage:
    python extract_pose_VITPose.py
    python extract_pose_VITPose.py --video my_clip.mov
    python extract_pose_VITPose.py --video my_clip.mov --output pose_data.json
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

# ── Parameters ────────────────────────────────────────────────────────────────
VIDEO_PATH  = "climbing_video.mov"
OUTPUT_PATH = "pose_data.json"

# RTMPose mode. Options:
#   'performance'  — RTMPose-x + YOLOX-x  (most accurate, recommended)
#   'balanced'     — RTMPose-m + YOLOX-m
#   'lightweight'  — RTMPose-s + YOLOX-tiny (fastest)
POSE_MODE = "performance"

# Minimum keypoint score to mark a landmark as visible
VISIBILITY_THRESHOLD = 0.3

# When multiple people appear in frame, pick the one with the largest bounding
# box (the climber typically fills most of the frame).
# Set False to pick by highest mean keypoint score instead.
SELECT_BY_BBOX_AREA = True


# ── Device selection ──────────────────────────────────────────────────────────
def _select_device():
    """
    Returns (backend, device) for rtmlib.
    rtmlib backend must be 'onnxruntime', 'openvino', or 'opencv'.
    device must be 'cpu' or 'cuda'.
    On Apple Silicon, onnxruntime uses CoreML automatically when device='cpu'.
    """
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            return "onnxruntime", "cuda"
    except Exception:
        pass
    return "onnxruntime", "cpu"


# ── MediaPipe 33-landmark schema ──────────────────────────────────────────────
LANDMARK_NAMES = [
    "NOSE",                                                      # 0
    "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",            # 1-3
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",         # 4-6
    "LEFT_EAR", "RIGHT_EAR",                                    # 7-8
    "MOUTH_LEFT", "MOUTH_RIGHT",                                # 9-10
    "LEFT_SHOULDER", "RIGHT_SHOULDER",                          # 11-12
    "LEFT_ELBOW", "RIGHT_ELBOW",                                # 13-14
    "LEFT_WRIST", "RIGHT_WRIST",                                # 15-16
    "LEFT_PINKY", "RIGHT_PINKY",                                # 17-18
    "LEFT_INDEX", "RIGHT_INDEX",                                # 19-20
    "LEFT_THUMB", "RIGHT_THUMB",                                # 21-22
    "LEFT_HIP", "RIGHT_HIP",                                    # 23-24
    "LEFT_KNEE", "RIGHT_KNEE",                                  # 25-26
    "LEFT_ANKLE", "RIGHT_ANKLE",                                # 27-28
    "LEFT_HEEL", "RIGHT_HEEL",                                  # 29-30
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",                      # 31-32
]

# COCO-17 index → MediaPipe index
COCO_TO_MP = {
    0:  0,   # nose           → NOSE
    1:  2,   # left_eye       → LEFT_EYE
    2:  5,   # right_eye      → RIGHT_EYE
    3:  7,   # left_ear       → LEFT_EAR
    4:  8,   # right_ear      → RIGHT_EAR
    5:  11,  # left_shoulder  → LEFT_SHOULDER
    6:  12,  # right_shoulder → RIGHT_SHOULDER
    7:  13,  # left_elbow     → LEFT_ELBOW
    8:  14,  # right_elbow    → RIGHT_ELBOW
    9:  15,  # left_wrist     → LEFT_WRIST
    10: 16,  # right_wrist    → RIGHT_WRIST
    11: 23,  # left_hip       → LEFT_HIP
    12: 24,  # right_hip      → RIGHT_HIP
    13: 25,  # left_knee      → LEFT_KNEE
    14: 26,  # right_knee     → RIGHT_KNEE
    15: 27,  # left_ankle     → LEFT_ANKLE
    16: 28,  # right_ankle    → RIGHT_ANKLE
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_landmarks(keypoints_px, scores, width, height):
    """
    Map RTMPose COCO-17 output to the 33-landmark MediaPipe JSON format.

    keypoints_px : array (17, 2) — [x, y] pixel coordinates
    scores       : array (17,)   — per-keypoint confidence
    Returns a list of 33 dicts: {name, x, y, z, visibility}
    x/y are normalised to [0, 1]; z is always 0.0 (model is 2D).
    """
    lms = [
        {"name": name, "x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0}
        for name in LANDMARK_NAMES
    ]
    for coco_i, mp_i in COCO_TO_MP.items():
        x_px  = float(keypoints_px[coco_i][0])
        y_px  = float(keypoints_px[coco_i][1])
        score = float(scores[coco_i])
        lms[mp_i] = {
            "name":       LANDMARK_NAMES[mp_i],
            "x":          round(x_px / width,  6),
            "y":          round(y_px / height, 6),
            "z":          0.0,
            "visibility": round(score, 4),
        }
    return lms


def select_best_person(keypoints_list, scores_list, bboxes):
    """
    Given parallel lists of keypoints/scores/bboxes for all detected people,
    return the index of the best candidate (climber).
    """
    if len(keypoints_list) == 1:
        return 0

    if SELECT_BY_BBOX_AREA and bboxes is not None and len(bboxes) > 0:
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
        return int(np.argmax(areas))

    mean_scores = [float(np.mean(s)) for s in scores_list]
    return int(np.argmax(mean_scores))


# ── Main ──────────────────────────────────────────────────────────────────────

def extract_pose_rtmpose(video_path: str, output_path: str) -> None:

    try:
        from rtmlib import Body
    except ImportError:
        sys.exit(
            "\nrtmlib is not installed. Run:\n"
            "  pip install rtmlib opencv-python tqdm numpy onnxruntime\n"
        )

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Error: could not open video at '{video_path}'")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo : {video_path}")
    print(f"  Resolution : {width}×{height}")
    print(f"  FPS        : {fps:.2f}")
    print(f"  Frames     : {total_frames}")
    print(f"  Duration   : {total_frames / fps:.1f}s\n")

    backend, device = _select_device()
    print(f"Loading RTMPose ({POSE_MODE} mode, backend: {backend}, device: {device})")
    print("(First run downloads model weights — subsequent runs are instant)\n")

    body = Body(mode=POSE_MODE, backend=backend, device=device)

    # ── Process frames ────────────────────────────────────────────────────────
    frames_data    = []
    detected_count = 0

    with tqdm(total=total_frames, unit="frame", desc="Extracting pose") as pbar:
        frame_idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # rtmlib Body() returns (keypoints, scores)
            # keypoints : (N, 17, 2)  pixel coords
            # scores    : (N, 17)     confidence per keypoint
            keypoints, scores = body(frame_bgr)

            frame_entry = {
                "frame":       frame_idx,
                "timestamp_s": round(frame_idx / fps, 4),
                "detected":    False,
                "landmarks":   [],
            }

            if keypoints is not None and len(keypoints) > 0:
                best_i = select_best_person(keypoints, scores, bboxes=None)  # bboxes internal to rtmlib
                kpts   = keypoints[best_i]   # (17, 2)
                scos   = scores[best_i]      # (17,)

                # Only mark as detected if at least one key joint is confident
                key_joints = [5, 6, 11, 12]   # shoulders + hips
                if any(scos[j] >= VISIBILITY_THRESHOLD for j in key_joints):
                    frame_entry["detected"]  = True
                    frame_entry["landmarks"] = build_landmarks(kpts, scos, width, height)
                    detected_count += 1

            frames_data.append(frame_entry)
            frame_idx += 1
            pbar.update(1)

    cap.release()

    detection_rate = detected_count / total_frames * 100 if total_frames else 0
    print(f"\nPose detected in {detected_count}/{total_frames} frames ({detection_rate:.1f}%)")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "video": {
            "path":         os.path.abspath(video_path),
            "fps":          fps,
            "width":        width,
            "height":       height,
            "total_frames": total_frames,
        },
        "landmarks": frames_data,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Saved → {output_path}  ({size_mb:.1f} MB)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract RTMPose keypoints from a climbing video (drop-in for extract_pose.py)."
    )
    parser.add_argument("--video",  default=VIDEO_PATH,  help="Path to input video")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Path to output JSON")
    args = parser.parse_args()

    extract_pose_rtmpose(args.video, args.output)
