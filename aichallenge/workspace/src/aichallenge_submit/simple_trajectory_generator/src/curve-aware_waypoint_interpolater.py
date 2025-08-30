import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Plan:
# 1. Use the current data points to generate a smoother raceline with interpolated points (via Catmull-Rom)
# 2. Use smoother raceline with interpolated points to calculate curvatures at each throughout the raceline with 


# ------------
# FUNCTIONS
# ------------

# removing points that are near each other (in order to avoid errors caused by dividing by 0)
def remove_duplicate_points(P, eps=1e-6):
    keep = [0]
    for i in range(1, len(P)):
        if np.linalg.norm(P[i] - P[keep[-1]]) > eps:
            keep.append(i)
    return P[keep]

# generates time values along the points such that transitions between time is smooth
def centripetal_params(P, alpha=0.5):
    # t_0 = 0; t_i = t_{i-1} + |P_i - P_{i-1}|^alpha
    t = [0.0]
    for i in range(1, len(P)):
        dist = np.linalg.norm(P[i] - P[i-1])
        t.append(t[-1] + dist**alpha)
    return np.array(t)

def lerp(A, B, u0, u1, u):
        return ((u1 - u)[:, None] * A + (u - u0)[:, None] * B) / (u1 - u0)

def catmull_rom_segment(P0, P1, P2, P3, t0, t1, t2, t3, t):
    # General Catmull–Rom via recursive linear interpolation (centripetal parametrization)
    # t is a vector in [t1, t2]

    A1 = lerp(P0, P1, t0, t1, t)
    A2 = lerp(P1, P2, t1, t2, t)
    A3 = lerp(P2, P3, t2, t3, t)

    B1 = lerp(A1, A2, t0, t2, t)
    B2 = lerp(A2, A3, t1, t3, t)

    C  = lerp(B1, B2, t1, t2, t)  # final point(s)
    return C

def spline_resample_dense(P_raw, base_ds=0.3, alpha=0.5):
    """
    Build a dense, smooth polyline from raw points using centripetal Catmull–Rom.
    base_ds ~ approximate spacing of dense points (meters).
    """
    P = remove_duplicate_points(np.asarray(P_raw, dtype=float))
    n = len(P)
    if n < 4:
        raise ValueError("Need at least 4 points for Catmull–Rom spline.")

    P_ext = np.vstack([P[-1], P, P[0]])     # extended P using mirrored endpoints
    t = centripetal_params(P_ext, alpha=alpha)

    dense = []
    # Build per-segment sampling proportionally to chord length
    for i in range(1, len(P_ext)-2):
        P0, P1, P2, P3 = P_ext[i-1], P_ext[i], P_ext[i+1], P_ext[i+2]
        t0, t1, t2, t3 = t[i-1], t[i], t[i+1], t[i+2]
        # Number of samples for this segment based on P1->P2 length
        seg_len = np.linalg.norm(P2 - P1)   # take segment length
        m = max(3, int(np.ceil(seg_len / base_ds)))     # take equally spaced points within the segment based on user-inputted base distance
        ts = np.linspace(t1, t2, m, endpoint=False)     # avoid duplicating seam points
        C = catmull_rom_segment(P0, P1, P2, P3, t0, t1, t2, t3, ts)
        dense.append(C)
    # Append the very last point
    #dense.append(P[-1][None, :])
    #dense.insert(0, P[0][None, :])  # insert at start
    dense.append(P[-1][None, :])    # append at end

    return np.vstack(dense)

def discrete_curvature(P):
    
    # Discrete curvature κ ≈ Δθ / Δs using turning angle between consecutive segments.
    
    N = len(P)
    kappa = np.zeros(N)
    # Tangents (finite differences)
    V = np.diff(P, axis=0)                                  # vector differences
    seg_len = np.linalg.norm(V, axis=1) + 1e-12             # segment lengths
    T = V / seg_len[:, None]                                # normalized tangent vectors

    # Angle change between tangents
    for i in range(1, N-1):
        t_prev = T[i-1]
        t_next = T[i]  # tangent after point i
        dot = np.clip(np.dot(t_prev, t_next), -1.0, 1.0)        # cosine of angle between them
        dtheta = np.arccos(dot)                                 # actual turning angle in radians
        ds = 0.5 * (np.linalg.norm(P[i] - P[i-1]) + np.linalg.norm(P[i+1] - P[i])) + 1e-12
        kappa[i] = dtheta / ds

    # Edge handling
    kappa[0] = kappa[1]
    kappa[-1] = kappa[-2]
    return kappa

def adaptive_downsample(P_dense, kappa_dense,
                        ds_min=0.5, ds_max=5.0, kappa_scale=10.0):
    """
    Select waypoints with spacing that shrinks in high curvature and grows on straights.
    ds_target = clip(ds_max / (1 + kappa_scale * κ), [ds_min, ds_max])
    """
    ds_target = np.clip(ds_max / (1.0 + kappa_scale * kappa_dense), ds_min, ds_max)

    out_idx = [0]
    last = 0
    accum = 0.0
    for i in range(1, len(P_dense)):
        step = np.linalg.norm(P_dense[i] - P_dense[i-1])
        accum += step
        if accum >= ds_target[i]:
            out_idx.append(i)
            accum = 0.0
            last = i
    if out_idx[-1] != len(P_dense) - 1:
        out_idx.append(len(P_dense) - 1)

    return P_dense[out_idx], kappa_dense[out_idx]

def yaw_from_polyline(P):
    # Central difference for interior, forward/backward for ends
    N = len(P)
    yaw = np.zeros(N)
    for i in range(N):
        if i == 0:
            d = P[1] - P[0]
        elif i == N - 1:
            d = P[-1] - P[-2]
        else:
            d = P[i+1] - P[i-1]
        yaw[i] = np.arctan2(d[1], d[0])
    return yaw

def quat_from_yaw(yaw):
    # Planar quaternion: (0, 0, sin(yaw/2), cos(yaw/2))
    half = 0.5 * yaw
    z = np.sin(half)
    w = np.cos(half)
    x = np.zeros_like(z)
    y = np.zeros_like(z)
    return x, y, z, w

def arc_lengths(P):
    ds = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    return s, ds

# ------------------------------------------------------------
# 2) SPEED PLANNING
# ------------------------------------------------------------
def speed_plan(P, kappa,
               v_min=1.0, v_max=None,
               a_lat_max=2.5,      # m/s^2 lateral comfort/limit
               a_acc_max=1.5,      # m/s^2 forward accel
               a_dec_max=2.5):     # m/s^2 braking/decel
    """
    v_curve = sqrt(a_lat_max / max(kappa, eps))
    Then enforce longitudinal accel/decel limits via forward/backward pass.
    """
    eps = 1e-6
    if v_max is None or v_max <= 0:
        v_max = 15.0  # fallback (≈54 km/h)

    # Curve speed
    v_des = np.sqrt(np.clip(a_lat_max / np.maximum(kappa, eps), 0, (v_max+10)**2))
    v_des = np.clip(v_des, v_min, v_max)

    # Longitudinal constraints along arc length
    s, ds = arc_lengths(P)
    v = v_des.copy()

    # Forward pass (acceleration limit)
    for i in range(1, len(v)):
        v[i] = min(v[i], np.sqrt(v[i-1]**2 + 2.0 * a_acc_max * ds[i-1]))

    # Backward pass (deceleration limit)
    for i in range(len(v)-2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i+1]**2 + 2.0 * a_dec_max * ds[i]))

    return v

# ----------------
# MAIN FUNCTION
# ----------------

df = pd.read_csv("../data/raceline_awsim_30km_from_garage_og.csv")

# Extract path (assumes columns named exactly as you showed)
Px = df["x"].to_numpy(dtype=float)
Py = df["y"].to_numpy(dtype=float)
P_raw = np.column_stack([Px, Py])       # shape: (N, 2)

# 1) Make a smooth dense curve
P_dense = spline_resample_dense(P_raw, base_ds=0.3, alpha=0.5)

# 2) Curvature on dense curve
kappa_dense = discrete_curvature(P_dense)
kappa_raw = discrete_curvature(P_raw)

# 3) Adaptive downsample: finer in tight curves, coarser on straights
P_adapt, kappa_adapt = adaptive_downsample(
    P_dense, kappa_dense,
    ds_min=0.01,      # smallest spacing in tight turns (m)
    ds_max=6.0,      # largest spacing on straights (m)
    kappa_scale=14.0 # higher => more aggressive densification in curves
)
"""
P_adapt, kappa_adapt = adaptive_downsample(
    P_raw, kappa_raw,
    ds_min=0.05,      # smallest spacing in tight turns (m)
    ds_max=8.0,      # largest spacing on straights (m)
    kappa_scale=15.0 # higher => more aggressive densification in curves
)
"""

# 4) Yaw & quaternion for new path
yaw = yaw_from_polyline(P_adapt)
xq, yq, zq, wq = quat_from_yaw(yaw)

# 5) Speed plan: use your data's max speed as the straight-line cap (if present)
# v_max_cap = float(df["speed"].max()) if "speed" in df.columns else 15.0
v_max_cap = 50.0
speed = speed_plan(
    P_adapt, kappa_adapt,
    v_min=1.0, v_max=v_max_cap,
    a_lat_max=25.5,   # tune: lower => slower through curves
    a_acc_max=40.0,   # tune: how quickly to speed up
    a_dec_max=15.5    # tune: how hard you allow braking
)

# 6) Build the new waypoint table with your original schema
out = pd.DataFrame({
    "x": P_adapt[:, 0],
    "y": P_adapt[:, 1],
    "z": 0.0,                 # flat ground
    "x_quat": 0.0,
    "y_quat": 0.0,
    "z_quat": zq,
    "w_quat": wq,
    "speed": speed
})

# 7) (Optional) Save and visualize
out.to_csv("raceline_awsim_30km_from_garage.csv", index=False)
print("Wrote raceline_adaptive.csv with", len(out), "waypoints")

# --- Quick visualization: original vs new (speed-colored) ---
plt.figure(figsize=(9, 8))
#plt.plot(P_raw[:,0], P_r5.aw[:,1], linewidth=1.0, label="Original points")
#plt.plot(P_dense[:,0], P_dense[:,1], linewidth=1.0, label="Densified points")
plt.plot(P_adapt[:,0], P_adapt[:,1], linewidth=1.0, label="Adapted points")
sc = plt.scatter(out["x"], out["y"], c=out["speed"], cmap="RdYlBu_r", s=50)
plt.colorbar(sc, label="Planned speed (m/s)")
plt.axis("equal"); plt.grid(True)
plt.title("Adaptive Raceline: dense on curves, sparse on straights")
plt.legend()
plt.show()
