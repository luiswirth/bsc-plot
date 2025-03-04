import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

path = "/home/luis/thesis/formoniq/out/visual"

# General file loader
def load_file(filename, dtype=float):
    with open(filename, 'r') as f:
        return [list(map(dtype, line.strip().split())) for line in f if line.strip()]

# Load vector field evaluations
def load_evaluations(filename):
    evaluations = []
    with open(filename, 'r') as f:
        cell = []
        for line in f:
            line = line.strip()
            if line == "cell":
                if cell:
                    evaluations.append(np.array(cell))
                    cell = []
            elif line:
                cell.append(list(map(float, line.split())))
        if cell:
            evaluations.append(np.array(cell))
    return evaluations

# Generate a grid over a triangle and mask points outside the triangle
def grid_over_triangle(triangle, nsamples):
    """
    Generates a rectangular grid over the bounding box of a triangle
    and masks out points outside the triangle.
    """
    # Compute the bounding box
    x_min, y_min = np.min(triangle, axis=0)
    x_max, y_max = np.max(triangle, axis=0)

    # Create a regular grid over that bounding box
    x = np.linspace(x_min, x_max, nsamples)
    y = np.linspace(y_min, y_max, nsamples)
    grid_x, grid_y = np.meshgrid(x, y)

    p0, p1, p2 = triangle

    def barycentric_coords(px, py):
        """
        Solve for barycentric coords of (px,py) w.r.t. triangle (p0,p1,p2).
        """
        A = np.array([p1 - p0, p2 - p0]).T
        rhs = np.array([px, py]) - p0
        lambdas = np.linalg.solve(A, rhs)  # [lambda1, lambda2]
        l0 = 1.0 - (lambdas[0] + lambdas[1])
        return l0, lambdas[0], lambdas[1]

    # Compute barycentric coords for each grid point
    flat_coords = np.array([
        barycentric_coords(px, py)
        for (px, py) in zip(grid_x.ravel(), grid_y.ravel())
    ])
    bary_coords = flat_coords.reshape(grid_x.shape + (3,))

    # Mask points that lie outside the reference triangle
    mask = (
        (bary_coords[..., 0] >= 0) &
        (bary_coords[..., 1] >= 0) &
        (bary_coords[..., 2] >= 0)
    )
    return grid_x, grid_y, mask, bary_coords

# -------------------------------------------------
# Load data
coords = np.array(load_file(f'{path}/coords.txt'))
triangles = np.array(load_file(f'{path}/cells.txt', dtype=int))
vector_field = load_evaluations(f'{path}/evaluations.txt')

# We will create a figure and axes
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

nsamples_per_cell = 200

# First pass: find global min/max of the field magnitude,
# so each cell has the same color scale
all_magnitudes = []
for triangle, triangle_eval in zip(triangles, vector_field):
    triangle_coords = coords[triangle]
    grid_x, grid_y, mask, bary_coords = grid_over_triangle(triangle_coords, nsamples_per_cell)

    # Interpolate vector field values
    # triangle_eval is shape (3, 2) presumably: [ (u0,v0), (u1,v1), (u2,v2) ]
    u = (bary_coords[..., 0] * triangle_eval[0, 0] +
         bary_coords[..., 1] * triangle_eval[1, 0] +
         bary_coords[..., 2] * triangle_eval[2, 0])
    v = (bary_coords[..., 0] * triangle_eval[0, 1] +
         bary_coords[..., 1] * triangle_eval[1, 1] +
         bary_coords[..., 2] * triangle_eval[2, 1])

    # Magnitude
    magnitude = np.sqrt(u**2 + v**2)

    # Consider only valid (inside) points
    valid_mag = magnitude[mask]
    if valid_mag.size > 0:
        all_magnitudes.append(valid_mag)

if len(all_magnitudes) == 0:
    # Fallback: no data
    global_min, global_max = 0.0, 1.0
else:
    global_min = min(m.min() for m in all_magnitudes)
    global_max = max(m.max() for m in all_magnitudes)

# A tiny epsilon to avoid degenerate color range
if np.isclose(global_min, global_max):
    global_max += 1e-14

# -------------------------------------------------
# Second pass: actually plot each cell
for triangle, triangle_eval in zip(triangles, vector_field):
    triangle_coords = coords[triangle]
    grid_x, grid_y, mask, bary_coords = grid_over_triangle(triangle_coords, nsamples_per_cell)

    # Interpolate the vector field
    u = (bary_coords[..., 0] * triangle_eval[0, 0] +
         bary_coords[..., 1] * triangle_eval[1, 0] +
         bary_coords[..., 2] * triangle_eval[2, 0])
    v = (bary_coords[..., 0] * triangle_eval[0, 1] +
         bary_coords[..., 1] * triangle_eval[1, 1] +
         bary_coords[..., 2] * triangle_eval[2, 1])

    # Compute magnitude
    magnitude = np.sqrt(u**2 + v**2)

    # Mask outside
    u[~mask] = np.nan
    v[~mask] = np.nan
    magnitude[~mask] = np.nan

    heatmap = plt.scatter(grid_x, grid_y, c=magnitude, cmap='viridis', alpha=0.8, edgecolors='none')
        #cbar = plt.colorbar(heatmap)
    

    # Plot the streamlines (white, on top of the color)
    ax.streamplot(
        grid_x, grid_y, u, v,
        color='white', linewidth=0.7,
        density=0.3,
        broken_streamlines=False,
        zorder=2
    )

# Draw the triangulation mesh on top
tri = Triangulation(coords[:, 0], coords[:, 1], triangles)
ax.triplot(tri, color='black', linewidth=1, zorder=3)

# Make it equal aspect, label, and add a colorbar
ax.set_aspect('equal', 'box')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Streamlines per Cell (Piecewise-Linear Field)")

cbar = plt.colorbar(heatmap, ax=ax)
cbar.set_label("Magnitude of vector field")

# Save and show
plt.savefig(f"{path}/plot.png", dpi=300, bbox_inches='tight')
plt.show()
