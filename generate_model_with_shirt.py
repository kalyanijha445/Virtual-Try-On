import pyvista as pv
import json
import numpy as np


# Load the OBJ file
obj_file = r"static/Generic_Male.obj"
print("Loading Generic Male model...")
mesh = pv.read(obj_file)

# Load landmarks from JSON file
landmarks_file = "landmarks.json"
print("Loading landmarks...")
with open(landmarks_file, "r") as f:
    landmarks_data = json.load(f)

# Convert landmarks to 3D points
points = np.array([[lm["x"], lm["y"], lm["z"]] for lm in landmarks_data])

# Add the landmarks as a new point cloud to the model
point_cloud = pv.PolyData(points)

# Plot the model with the landmarks overlaid
print("Rendering model with landmarks...")
plotter = pv.Plotter()
plotter.add_mesh(mesh, color="tan", show_edges=True)
plotter.add_points(point_cloud, color="red", point_size=5.0, render_points_as_spheres=True)
plotter.add_title("Model with Landmarks Overlay")
plotter.show()
