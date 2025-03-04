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

# Barycentric interpolation inside a triangle
def barycentric_interpolation(values, nsamples):
  barys = [
    (
      1 - (i + j) / (nsamples - 1),
      j / (nsamples - 1),
      i / (nsamples - 1),
    )
    for i in range(nsamples)
    for j in range(nsamples - i)
  ]

  v0, v1, v2 = values

  interpolated_values = [
    alpha * v0 + beta * v1 + gamma * v2
    for alpha, beta, gamma in barys
  ]

  return np.array(interpolated_values)

# Load data
coords = np.array(load_file(f'{path}/coords.txt'))
triangles = np.array(load_file(f'{path}/cells.txt', dtype=int))
vector_field = load_evaluations(f'{path}/evaluations.txt')

# Plotting the mesh
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
tri = Triangulation(coords[:, 0], coords[:, 1], triangles)
ax.triplot(tri, color='black', linewidth=0.5)

# Sample and plot vector field inside each triangle
nsamples_per_cell = 10  # Number of samples per edge

# Cache quiver data
sample_points = []
sample_vectors = []
sample_magnitudes = []

for triangle, triangle_eval in zip(triangles, vector_field):
  triangle_coords = coords[triangle]
  points = barycentric_interpolation(triangle_coords, nsamples_per_cell)
  vectors = barycentric_interpolation(triangle_eval, nsamples_per_cell)

  # Normalize vectors for uniform length
  magnitudes = np.linalg.norm(vectors, axis=1)
  normalized_vectors = vectors / magnitudes[:, np.newaxis]

  sample_points.append(points)
  sample_vectors.append(normalized_vectors)
  sample_magnitudes.append(magnitudes)

# Combine all quiver data
sample_points = np.vstack(sample_points)
sample_vectors = np.vstack(sample_vectors)
sample_magnitudes = np.hstack(sample_magnitudes)

# Determine appropriate scaling factor
scale_factor = 0.01 * np.max(np.linalg.norm(coords, axis=1))

quiver = ax.quiver(
  sample_points[:, 0], sample_points[:, 1],  # X, Y positions
  sample_vectors[:, 0], sample_vectors[:, 1],  # U, V components
  sample_magnitudes, cmap='viridis', # color = magnitude
  angles='xy', scale_units='xy',
  scale=1 / scale_factor,
  alpha=0.8,
  #width, headwidth, headlength and headaxislength
)
cbar = plt.colorbar(quiver, ax=ax, label='Magnitude')

# Set axis labels and aspect ratio
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_aspect('equal')
ax.set_title('Mesh and Vector Field')

# Add colorbar for magnitude

# Save and show the plot
plt.savefig(f"{path}/plot.png", dpi=300)
plt.show()
