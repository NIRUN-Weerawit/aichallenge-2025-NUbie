import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from scipy.interpolate import CubicSpline
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os


class PathVisualizer:
    def __init__(self, csv_path, n_control_points=20):
        self.csv_path = csv_path
        print(f"Loading data from: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.n_control_points = n_control_points
        self.selected_point = None
        self.drag_active = False

        # Calculate display range (only once during initialization)
        margin = 10
        map_name = "lanelet2_map.osm"
        edge_points = pd.read_csv("points.csv")
        self.map = edge_points
        self.x_min = edge_points["x"].min() - margin
        self.x_max = edge_points["x"].max() + margin
        self.y_min = edge_points["y"].min() - margin
        self.y_max = edge_points["y"].max() + margin

        self.setup_gui()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Path Visualizer")

        # Control Panel
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # Control points input
        tk.Label(control_frame, text="Control Points:").pack(side=tk.LEFT)
        self.control_points_var = tk.StringVar(value=str(self.n_control_points))
        control_entry = tk.Entry(
            control_frame, textvariable=self.control_points_var, width=5
        )
        control_entry.pack(side=tk.LEFT)
        tk.Button(
            control_frame, text="Update", command=self.update_control_points
        ).pack(side=tk.LEFT)

        # Reset Button
        tk.Button(control_frame, text="Reset Path", command=self.reset_path).pack(
            side=tk.LEFT
        )

        # Save Button
        tk.Button(
            control_frame, text="Save Path", command=self.save_modified_path
        ).pack(side=tk.LEFT)

        # Main figure with fixed size
        self.fig = Figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111)

        # Fixed layout
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)

        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Initialize data
        self.initialize_path_data()

        # Event handlers
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)

    def save_modified_path(self):
        # Generate new filename from original path
        dir_path = os.path.join(os.path.dirname(self.csv_path), "modified_trajectory")
        base_name = os.path.basename(self.csv_path)
        name_without_ext = os.path.splitext(base_name)[0]
        new_file_name = f"{name_without_ext}.csv"
        new_path = os.path.join(dir_path, new_file_name)

        # Create modified dataframe
        modified_df = pd.DataFrame(
            {
                "x": self.interpolated_path[:, 0],
                "y": self.interpolated_path[:, 1],
                "z": self.df["z"].values,
                "x_quat": self.df["x_quat"].values,
                "y_quat": self.df["y_quat"].values,
                "z_quat": self.df["z_quat"].values,
                "w_quat": self.df["w_quat"].values,
                "speed": self.df["speed"].values,
            }
        )

        # Ensure column order matches original file
        modified_df = modified_df[self.df.columns]

        # Save to CSV
        modified_df.to_csv(new_path, index=False)
        print(f"Modified path saved to: {new_path}")
        
    def initialize_path_data(self):
        # Original path points
        self.original_path = np.column_stack((self.df["x"],self.df["y"]))
        self.original_velocities = self.df["speed"].values
        
        # Create control points by sampling the original path
        path_indices = np.linspace(
            0, len(self.original_path) - 1, self.n_control_points, dtype=int
        )
        
        self.control_points = self.original_path[path_indices]
        self.control_velocities = self.original_velocities[path_indices]

        # Set end point same as start point for path
        self.control_points[-1] = self.control_points[0]
        self.control_velocities[-1] = self.control_velocities[0]
        
        # Initialize interpolated path
        self.update_interpolated_path()

        # Draw initial path
        self.draw_path()
        
    def update_control_points(self):
        try:
            new_n = int(self.control_points_var.get())
            if new_n > 1:
                self.n_control_points = new_n
                self.initialize_path_data()
        except ValueError:
            pass
        
    def update_interpolated_path(self):
        # Parameter along the path
        t = np.linspace(0, 1, len(self.control_points))
        
        # Create interpolation for path
        cs_x = CubicSpline(t, self.control_points[:, 0], bc_type='periodic')
        cs_y = CubicSpline(t, self.control_points[:, 1], bc_type='periodic')
        
        # Create interpolation for velocities
        cs_v = CubicSpline(t, self.control_velocities, bc_type='periodic')
        
        # Display path (high resolution for smooth visualization)
        t_fine = np.linspace(0, 1, 200)
        self.display_path = np.column_stack((cs_x(t_fine), cs_y(t_fine)))
        self.display_velocities = cs_v(t_fine)
        
        # Original resolution path and velocities (for saving)
        t_orig = np.linspace(0, 1, len(self.df))
        self.interpolated_path = np.column_stack((cs_x(t_orig), cs_y(t_orig)))
        self.interpolated_velocities = cs_v(t_orig)
    
    def draw_path(self):
        self.ax.clear()
        
        # Draw boundaries
        self.ax.scatter(self.map['x'], self.map['y'], color='black', s = 5)
        
        # Create line segments for the interpolated path
        points = self.display_path
        segments = np.array(
            [[points[i], points[i+1]] for i in range(len(points)-1)]
        )
        
        # Create line collection with velocity colors
        norm = mcolors.Normalize(
            vmin=self.display_velocities.min(), vmax=self.display_velocities.max()
        )
        lc = LineCollection(segments, cmap="RdYlBu_r", norm=norm, linewidth=5)
        lc.set_array(self.display_velocities[:-1])
        self.ax.add_collection(lc)
        
        # Draw control points
        self.ax.scatter(
            self.control_points[:, 0],
            self.control_points[:, 1],
            c="blue",
            s=40,
            zorder=5,
            label="Control Points",
        )
        
        # Graph settings
        self.ax.set_aspect("equal")
        self.ax.grid(True)
        self.ax.legend()
        self.ax.set_title(
            f"Path (Red=Fast, Blue=Slow, Speed: {self.interpolated_velocities.min():.1f}-{self.interpolated_velocities.max():.1f} m/s)"
        )
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")

        # Set fixed limits
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)

        self.canvas.draw()
        
    def find_nearest_control_point(self, x, y):
        distances = np.sqrt(np.sum((self.control_points - [x, y]) ** 2, axis=1))
        return np.argmin(distances)
    
    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.selected_point = self.find_nearest_control_point(event.xdata, event.ydata)
        self.drag_active = True
        
    def on_motion(self, event):
        if not self.drag_active or event.inaxes != self.ax:
            return

        # Update control point position
        new_position = [event.xdata, event.ydata]
        self.control_points[self.selected_point] = new_position

        # Synchronize start and end points
        if self.selected_point == 0:
            self.control_points[-1] = new_position
        elif self.selected_point == len(self.control_points) - 1:
            self.control_points[0] = new_position

        # Update interpolated path
        self.update_interpolated_path()

        # Redraw
        self.draw_path()

    def on_release(self, event):
        self.drag_active = False

    def reset_path(self):
        self.initialize_path_data()

    def run(self):
        self.root.mainloop()
        
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "raceline_awsim_30km_from_garage.csv"

    print(f"Starting visualization with file: {csv_path}")
    try:
        visualizer = PathVisualizer(csv_path, n_control_points=113)
        visualizer.run()
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)