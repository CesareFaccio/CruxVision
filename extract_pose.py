"""
extract_pose.py
---------------
Extracts MediaPipe Pose landmarks from every frame of a climbing video.
Uses the modern MediaPipe Tasks API (mediapipe >= 0.10).

Requirements:
    pip install mediapipe opencv-python tqdm

Output:
    pose_data.json  — landmark positions for every frame, structured as:
    {
        "video": { "path", "fps", "width", "height", "total_frames" },
        "landmarks": [
            {
                "frame": 0,
                "timestamp_s": 0.0,
                "detected": true,
                "landmarks": [          # 33 body landmarks
                    {
                        "name": "NOSE",
                        "x": 0.52,      # normalised 0–1 (fraction of frame width)
                        "y": 0.12,      # normalised 0–1 (fraction of frame height)
                        "z": -0.04,     # relative depth (smaller = closer to camera)
                        "visibility": 0.99
                    },
                    ...
                ]
            },
            ...
        ]
    }

Usage:
    python extract_pose.py
    python extract_pose.py --video my_clip.mov
    python extract_pose.py --video my_clip.mov --output results.json
"""

import argparse
import json
import os
import sys
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

# ── MediaPipe landmark names in index order ───────────────────────────────────
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

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
MODEL_FILE = "pose_landmarker_full.task"


def download_model(path: str) -> None:
    """Download the MediaPipe pose landmarker model if not already present."""
    if os.path.exists(path):
        print(f"Model already present: {path}")
        return
    print(f"Downloading model → {path}  (this is a one-time ~30 MB download)")
    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
        print(f"\r  {pct:.1f}%", end="", flush=True)
    urllib.request.urlretrieve(MODEL_URL, path, reporthook=_progress)
    print(f"\r  Done.          ")


def extract_pose(video_path: str, output_path: str, model_path: str) -> None:
    # ── Download model if needed ──────────────────────────────────────────────
    download_model(model_path)

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Error: could not open video at '{video_path}'")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo: {video_path}")
    print(f"  Resolution : {width}×{height}")
    print(f"  FPS        : {fps:.2f}")
    print(f"  Frames     : {total_frames}")
    print(f"  Duration   : {total_frames / fps:.1f}s\n")

    # ── Initialise MediaPipe Pose (Tasks API) ─────────────────────────────────
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ── Process frames ────────────────────────────────────────────────────────
    frames_data   = []
    detected_count = 0

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        with tqdm(total=total_frames, unit="frame", desc="Extracting pose") as pbar:
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                # MediaPipe Tasks API expects RGB mp.Image
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                # Timestamp in milliseconds (must be monotonically increasing)
                timestamp_ms = int(frame_idx / fps * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                detected = len(result.pose_landmarks) > 0
                if detected:
                    detected_count += 1

                frame_entry = {
                    "frame":       frame_idx,
                    "timestamp_s": round(frame_idx / fps, 4),
                    "detected":    detected,
                    "landmarks":   [],
                }

                if detected:
                    for idx, lm in enumerate(result.pose_landmarks[0]):
                        frame_entry["landmarks"].append({
                            "name":       LANDMARK_NAMES[idx],
                            "x":          round(lm.x, 6),
                            "y":          round(lm.y, 6),
                            "z":          round(lm.z, 6),
                            "visibility": round(lm.visibility, 4),
                        })

                frames_data.append(frame_entry)
                frame_idx += 1
                pbar.update(1)

    cap.release()

    detection_rate = detected_count / total_frames * 100 if total_frames else 0
    print(f"\nPose detected in {detected_count}/{total_frames} frames ({detection_rate:.1f}%)")

    # ── Save output ───────────────────────────────────────────────────────────
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
    parser = argparse.ArgumentParser(description="Extract MediaPipe pose from a climbing video.")
    parser.add_argument("--video",  default="climbing_video.mov",       help="Path to input video")
    parser.add_argument("--output", default="pose_data.json",           help="Path to output JSON")
    parser.add_argument("--model",  default=MODEL_FILE,                 help="Path to .task model file")
    args = parser.parse_args()

    extract_pose(args.video, args.output, args.model)
