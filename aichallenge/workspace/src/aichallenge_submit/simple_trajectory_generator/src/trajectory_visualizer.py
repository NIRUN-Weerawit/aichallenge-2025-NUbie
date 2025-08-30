import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # set GUI backend


# Load CSV
df = pd.read_csv("../data/raceline_awsim_30km_from_garage.csv")
# df = pd.read_csv("raceline_adaptive.csv")

# Extract values
x = df["x"].values
y = df["y"].values
z_quat = df["z_quat"].values
w_quat = df["w_quat"].values
speed = df["speed"].values

# Convert quaternion (z, w) to yaw (heading angle)
# Since only z and w are nonzero: yaw = 2 * atan2(z, w)
yaw = 2 * np.arctan2(z_quat, w_quat)

# Plot path colored by speed
plt.figure(figsize=(10, 8))
sc = plt.scatter(x, y, c=speed, cmap="RdYlBu_r", s=30, label="Path")
plt.colorbar(sc, label="Speed")

# Draw orientation arrows every Nth point
N = 5  # interval for drawing arrows
arrow_scale = 5
for i in range(0, len(x), N):
    dx = np.cos(yaw[i]) * arrow_scale
    dy = np.sin(yaw[i]) * arrow_scale
    plt.arrow(x[i], y[i], dx, dy, head_width=5, head_length=8, fc="red", ec="red")

plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("Trajectory Visualization with Speed and Orientation")
plt.axis("equal")
plt.grid(True)
plt.show()