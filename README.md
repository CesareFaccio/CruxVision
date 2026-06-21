# CruxVision

**Computer vision pipeline for biomechanical analysis of rock climbing video.**

CruxVision analyses climbing footage to extract insights that are invisible to the naked eye. It tracks body pose frame-by-frame, detects where the climber's hands and feet are on holds, estimates the centre of mass, distributes bodyweight across contact points using a physics model, computes the forces through individual limb segments, and analyses the motion of the centre of mass over time. All results are rendered as an annotated dual-panel video.

---

## Pipeline Overview

```
climbing_video.mov
       │
       ▼
 extract_pose.py  ──────────────────────────►  pose_data.json
       │
       ▼
 generate_step_1.py  ───────────────────────►  holds.json
                     ───────────────────────►  com.json
       │
       ▼
 stats1.py  ────────────────────────────────►  weight_per_frame.json
       │
       ▼
 stats2.py  ────────────────────────────────►  com_motion.json
       │
       ▼
 reconstruct_plus.py  ──────────────────────►  climbing_plus.mp4
```

Run the scripts in this order. `reconstruct_video.py` is an optional lightweight alternative to `reconstruct_plus.py` that produces a simple overlay without the biomechanical analysis panel.

---

## File Reference

### `extract_pose.py` / `extract_pose_RTMPose.py`
**Input:** `climbing_video.mov`  
**Output:** `pose_data.json`

Extracts the 3D pose of the climber from every frame of the video. Two backends are available:

- **`extract_pose.py`** uses Google MediaPipe, which is fast and easy to install but uses a single-stage lightweight model that can struggle with unusual body positions and occlusion from climbing equipment.
- **`extract_pose_RTMPose.py`** uses RTMPose via rtmlib (ONNX runtime). It takes a top-down approach — a person detector (YOLOX) first finds the climber's bounding box, then RTMPose estimates keypoints within that box. This is more robust to unusual positions and partial occlusion because the pose model only ever sees the region containing the climber.

Both produce identical output: a JSON file with 33 body landmarks per frame, each with normalised x/y coordinates (fraction of frame width/height), a relative depth z value, and a visibility confidence score.

**Key output fields:**
```json
{
  "video": { "fps": 60.0, "width": 776, "height": 1586, "total_frames": 1118 },
  "landmarks": [
    {
      "frame": 0,
      "timestamp_s": 0.0,
      "detected": true,
      "landmarks": [
        { "name": "LEFT_WRIST", "x": 0.42, "y": 0.35, "z": -0.04, "visibility": 0.97 },
        ...
      ]
    }
  ]
}
```

---

### `generate_step_1.py`
**Input:** `pose_data.json`  
**Output:** `holds.json`, `com.json`

Does two things: detects holds from the pose trajectory, and estimates the centre of mass for every frame.

#### Hold Detection

A hold is defined as a period during which a wrist or ankle landmark is stationary — i.e. the climber's hand or foot is not moving. The algorithm works as follows:

1. **Extract positions** — the (x, y) position of each wrist and ankle landmark is read for every frame. Frames where the landmark was not detected (low visibility) are marked as NaN.
2. **Smooth** — a rolling mean with window `SMOOTH_WINDOW` is applied to the position signal before differentiation, reducing the effect of per-frame pose jitter.
3. **Compute velocity** — the per-frame speed of each landmark is computed as the Euclidean distance moved between consecutive frames (in normalised coordinates). The speed signal is also smoothed.
4. **Threshold** — frames where the smoothed speed is below `VELOCITY_THRESHOLD` are marked as stationary.
5. **Gap bridging** — short gaps (≤ `MAX_GAP_FRAMES`) between stationary periods are filled in, preventing a single noisy detection from splitting one hold into two.
6. **Duration filter** — stationary segments shorter than `MIN_HOLD_FRAMES` frames are discarded, removing brief pauses that do not represent genuine holds.
7. **Hold position** — the position of the hold is the median (x, y) of the landmark across all stationary frames, which is more robust to outliers than the mean.

The output stores the start frame, end frame, duration, and position of every detected handhold and foothold.

#### Centre of Mass Estimation

The centre of mass (CoM) is estimated from pose landmarks using Winter's (2009) body segment parameters — a widely-used biomechanical reference that specifies, for each body segment, what fraction of total body mass it represents and where along the segment its local CoM lies.

The 13 segments used and their mass fractions are:

| Segment | Mass fraction | CoM position along segment |
|---|---|---|
| Trunk (shoulder→hip) | 49.7% | 43% from shoulder |
| Head | 8.1% | modelled at midpoint of ear landmarks |
| Left / Right upper arm | 2.8% each | 43.6% from shoulder |
| Left / Right forearm | 1.6% each | 43.0% from elbow |
| Left / Right hand | 0.6% each | at wrist |
| Left / Right thigh | 10.0% each | 43.3% from hip |
| Left / Right lower leg | 4.65% each | 43.3% from knee |
| Left / Right foot | 1.45% each | at ankle |

For each segment, the local CoM position is computed as a linear interpolation between the two endpoint landmarks. The whole-body CoM is then the mass-weighted average of all segment CoM positions:

```
CoM = Σ (mass_fraction_i × local_CoM_i)
```

This is computed independently in x and y. The result is a (x, y) position in normalised image coordinates for every frame.

---

### `stats1.py`
**Input:** `holds.json`, `com.json`, `pose_data.json`  
**Output:** `weight_per_frame.json`

Estimates the fraction of the climber's bodyweight carried by each active contact point (left hand, right hand, left foot, right foot) at every frame. Also computes the axial force through each limb segment and the angles at the knees and elbows.

#### Force Distribution Model

The physical model treats the climber as a point mass (the CoM) connected to each active contact point by a rigid, non-extensible rod. Each rod can only carry force along its own axis — it cannot transmit shear. This is a standard static equilibrium model used in biomechanics.

Static equilibrium requires the vector sum of all contact forces to exactly balance the body weight acting downward:

```
Σ f_i × d̂_i = [0, -1]   (unit gravity vector, y downward in image space)
```

where f_i is the scalar force magnitude at contact i (as a fraction of body weight) and d̂_i is the unit direction vector from the contact point toward the CoM, corrected so it always points upward for a load-bearing contact:

```
d̂_i = sign(y_CoM - y_contact) × (contact_i - CoM) / |contact_i - CoM|
```

The sign term handles contacts both above and below the CoM:
- A handhold **above** the CoM acts in **tension** — the climber hangs from it, and the rod pulls the CoM upward.
- A foothold **below** the CoM acts in **compression** — the leg pushes the CoM upward.
- In both cases the sign convention ensures the force direction vector points upward.

This produces a 2×N linear system:

```
D · f = g
```

where D is a 2×N matrix whose columns are the direction vectors, f is the N-vector of force fractions, and g = [0, -1] is the unit gravity vector.

**For N = 1 contact:** the system is overdetermined (2 equations, 1 unknown). The single contact must carry 100% of body weight and the equation gives the required rod direction.

**For N = 2 contacts:** the system is exactly determined and has a unique solution.

**For N > 2 contacts:** the system is underdetermined — there are more unknowns than equations and infinitely many solutions exist. The minimum-norm solution is selected using the Moore-Penrose pseudoinverse (`numpy.linalg.lstsq`). This minimises the sum of squared forces across all contacts, which corresponds to the principle of minimum muscular effort: the body distributes load in the way that minimises the total force required.

#### Axial vs Vertical Force

The force returned by the equilibrium model is the **axial force** along each rod — the force the rod actually carries. The **vertical component** of that force (its projection onto the gravity axis) is:

```
F_vertical_i = F_axial_i × |d̂_iy|
```

where d̂_iy is the y-component of the unit direction vector. By construction, the vertical components always sum to exactly 100% of bodyweight per frame, making them the correct quantity to compare across contacts. The axial forces may individually exceed 100% BW when a rod is nearly horizontal, but their vertical projections are bounded.

#### Limb Segment Forces

The force travelling through each limb segment (lower leg, upper leg, forearm, upper arm) is computed from the contact force using a simple geometric amplification.

Assumption: the vertical force passing through the contact point is the same at every joint along the limb above it (i.e. the limb carries no additional external load between joints). This is the standard assumption for a weightless limb model.

If a limb segment makes an angle θ from the vertical, the axial force through the bone is:

```
F_axial_segment = F_vertical / cos(θ)
```

This comes directly from resolving forces along the segment axis. When the limb is straight and vertical (θ = 0°), cos(θ) = 1 and the axial force equals the vertical force. As the joint bends and the segment tilts away from vertical, cos(θ) decreases and the axial force increases. At θ = 45° the axial force is √2 ≈ 1.41× the vertical load; at θ = 60° it is 2×. This quantifies the mechanical penalty of a bent joint: a climber with deeply bent knees must generate significantly more bone and muscle force to support the same bodyweight than a climber with straight legs.

#### Joint Angles

The internal joint angle at each knee and elbow is computed from the pose landmarks using the dot product of the two limb vectors meeting at the joint:

```
cos(angle) = (v_distal · v_proximal) / (|v_distal| × |v_proximal|)
```

where v_distal points from the joint toward the foot/hand and v_proximal points from the joint toward the hip/shoulder. The result is the interior angle:
- 180° = fully straight limb
- 90° = right-angle bend
- < 90° = acute bend (deeply crouched or arms fully tucked)

---

### `stats2.py`
**Input:** `com.json`  
**Output:** `com_motion.json`

Analyses the trajectory of the centre of mass over time to extract velocity and acceleration. This reveals the climber's movement dynamics — distinguishing static held positions from dynamic moves and quantifying the speed and force of transitions.

#### Smoothing

Raw CoM positions contain noise from per-frame pose estimation errors. Differentiating a noisy signal amplifies that noise — velocity is noisy, acceleration is very noisy. A Savitzky-Golay filter is applied before differentiation to address this. Unlike a simple moving average, Savitzky-Golay fits a polynomial to the data within a sliding window and evaluates it at the centre point. This preserves peak shapes and inflection points better than a boxcar average while still suppressing high-frequency noise.

A second independent smoothing pass with its own window size is applied to the acceleration signal after differentiation, since acceleration (the second derivative) is especially sensitive to noise.

#### Velocity and Acceleration

After smoothing, velocity and acceleration are computed using central differences via `numpy.gradient`:

```
v[t] = (pos[t+1] - pos[t-1]) / (2 × Δt)
a[t] = (v[t+1]  - v[t-1])  / (2 × Δt)
```

Central differences use information from both sides of each point, giving lower truncation error than one-sided (forward or backward) differences. Edge points use one-sided differences automatically.

**Sign convention:** image y-coordinates increase downward, so upward movement corresponds to decreasing y. The output `vy_px_s` and `ay_px_s2` have their signs flipped so that upward velocity and acceleration are positive — matching physical intuition.

**Units:** all velocities and accelerations are in pixels per second and pixels per second squared respectively. Without knowing the camera distance and focal length it is not possible to convert to metres per second, but relative values and temporal patterns are fully meaningful.

**Dynamic move detection:** frames where the acceleration magnitude exceeds the 90th percentile of the observed distribution are flagged as dynamic moves. This percentile threshold adapts automatically to each climb rather than requiring a fixed absolute value.

---

### `reconstruct_video.py`
**Input:** `climbing_video.mov`, `pose_data.json`, `holds.json`, `com.json`  
**Output:** `pose_overlay.mp4`

A lightweight single-panel overlay video. Draws the skeleton, past hold markers (small faded circles showing the route taken), active hold markers (large coloured circles at current contact points), and the CoM diamond on top of the original video. Useful for quickly checking that pose detection and hold detection are working correctly before running the full analysis.

---

### `reconstruct_plus.py`
**Input:** `climbing_video.mov`, `pose_data.json`, `holds.json`, `com.json`, `weight_per_frame.json`, `com_motion.json`  
**Output:** `climbing_plus.mp4`

The main output of the pipeline. Produces a dual-panel side-by-side video at twice the original width.

**Left panel — weight distribution schematic**

A stationary body schematic drawn on a dark background. The figure does not move — it always shows the same anatomical position so that colours can be read consistently across the climb without the distraction of the skeleton moving around.

- The 8 limb segments (upper and lower arm and leg on each side) are coloured green→yellow→red by axial force magnitude, using the same colour scale as the right panel.
- Four circles at the wrist and ankle positions show the current vertical weight on each extremity as both kilograms and a percentage of bodyweight. Circles are green at low load and red at high load. When a limb is not on an active hold its circle is dimmed grey.
- A velocity arrow at the torso shows the current direction and speed of the CoM. Its length and colour scale with speed.
- A vertical bar on the left margin shows the vertical component of CoM velocity — filling upward in green when the climber is ascending, downward in blue when descending.
- A vertical bar on the right margin shows the CoM acceleration magnitude, coloured green→red.

**Right panel — skeleton overlay**

The original video frame with several layers drawn on top:

- Past hold markers: small faded grey circles showing holds already visited.
- The full pose skeleton.
- Limb segments coloured by axial force (same colour scale as the left panel schematic).
- Coloured circles at each knee and elbow showing the joint angle.
- The CoM diamond.
- Small coloured circles at each active hold showing the weight percentage.

**Temporal smoothing**

All displayed numeric values (weight fractions, segment forces, joint angles) are passed through an exponential moving average before rendering:

```
smoothed[t] = α × raw[t] + (1 - α) × smoothed[t-1]
```

`SMOOTHING_ALPHA = 0.5` is set at the top of the file. Lower values give heavier smoothing; set to 1.0 to disable.

---

## Installation

```bash
# Core dependencies
pip install mediapipe opencv-python numpy tqdm

# For RTMPose backend (recommended for accuracy)
pip install rtmlib onnxruntime

# For stats2.py smoothing (optional but recommended)
pip install scipy
```

---

## Running the Pipeline

```bash
# 1. Extract pose (choose one)
python extract_pose.py                  # MediaPipe (fast, easy)
python extract_pose_RTMPose.py          # RTMPose (more accurate)

# 2. Detect holds and compute CoM
python generate_step_1.py

# 3. Compute weight distribution and limb forces
python stats1.py

# 4. Compute CoM velocity and acceleration
python stats2.py

# 5. Render analysis video
python reconstruct_plus.py
```

All parameters (file paths, thresholds, body weight, smoothing settings) are defined at the top of each file — there are no command-line arguments required for the standard workflow.

---

## Key Assumptions and Limitations

**2D model.** All force calculations are performed in the 2D image plane. The z-coordinates provided by pose estimation (relative depth) are not used. This is a reasonable approximation for climbing footage shot approximately perpendicular to the wall, but will introduce error on overhanging routes where the climber's body is pulled significantly away from the wall plane.

**Static equilibrium.** The force distribution model assumes the climber is in static equilibrium at every frame — that is, the net force and net torque are both zero. In reality, climbing involves dynamic movement, and during transitions between holds the climber is accelerating. The model is most accurate during held positions and least accurate during dynamic moves.

**Weightless limbs.** The limb segment force model assumes the limb itself has no weight and carries no load between joints. In reality, limb segments do have mass (accounted for in the CoM calculation) and the forces at each joint differ slightly from those at the contact point. For typical climbing loads this is a minor effect.

**Body weight.** A single body weight value (`BODY_WEIGHT_KG = 70.0` in `stats1.py`) is used for all force calculations. Change this to the climber's actual weight for accurate force values in kilograms.

**Winter's parameters.** The CoM estimation uses population-average body segment parameters from Winter (2009). Individual variation in body composition and proportions will introduce some error in the CoM position estimate.

**Hold position.** Holds are detected purely from the absence of movement at the wrist or ankle landmark. The algorithm cannot distinguish between a hand resting on a hold and a hand pressed against the wall with no grip. Loose or baggy clothing, chalk, and equipment (harness, rope) can cause the pose estimator to misplace landmarks, leading to false or missed holds.

**Pixel units.** Velocities and accelerations are in pixels per second. Converting to physical units (m/s, m/s²) requires knowledge of the camera intrinsics and the distance from the camera to the climber, which are not available.

---

## References

Winter, D.A. (2009). *Biomechanics and Motor Control of Human Movement* (4th ed.). John Wiley & Sons.
