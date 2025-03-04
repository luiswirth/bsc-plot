import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from mpl_toolkits.mplot3d import Axes3D

# Directory containing the Whitney form data files
input_dir = "/home/luis/thesis/formoniq/out/"
output_dir = "/home/luis/thesis/formoniq/out/plots/"
os.makedirs(output_dir, exist_ok=True)

# List all Whitney form files
files = glob.glob(os.path.join(input_dir, "whitney*.txt"))

for file in files:
    data = np.loadtxt(file)
    file_name = os.path.basename(file)

    # Check the number of columns to determine if the data contains vectors or scalars
    if data.shape[1] == 6:
        # Extract x, y, z, vx, vy, vz for vector field
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        vx = data[:, 3]
        vy = data[:, 4]
        vz = data[:, 5]

        # Compute the magnitude of the vectors
        magnitude = np.sqrt(vx**2 + vy**2 + vz**2)

        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-10
        normalized_vx = vx / (magnitude + epsilon)
        normalized_vy = vy / (magnitude + epsilon)
        normalized_vz = vz / (magnitude + epsilon)

        # Scale the vectors to have more visible length
        enhanced_scale = 0.2
        vx = normalized_vx * enhanced_scale
        vy = normalized_vy * enhanced_scale
        vz = normalized_vz * enhanced_scale

        # Map magnitudes to RGBA colors using a colormap
        cmap = plt.cm.viridis
        colors = cmap(magnitude / magnitude.max())

        # Set default properties for white stroke color
        plt.rcParams['text.color'] = 'white'
        plt.rcParams['axes.labelcolor'] = 'white'
        plt.rcParams['xtick.color'] = 'white'
        plt.rcParams['ytick.color'] = 'white'
        plt.rcParams['axes.edgecolor'] = 'white'

        # Create the plot
        fig = plt.figure(figsize=(10, 10), facecolor='black')
        ax = fig.add_subplot(111, projection='3d', facecolor='black')

        # Plot the vector field using quiver with larger arrow size and color for magnitude
        quiver = ax.quiver(x, y, z, vx, vy, vz, length=0.1, linewidth=1.5, arrow_length_ratio=1.0, color=colors)
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, label="Magnitude")
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        # Draw the edges of the tetrahedron
        tetrahedron_edges = [
            ([0, 1], [0, 0], [0, 0]),  # Edge from (0,0,0) to (1,0,0)
            ([0, 0], [0, 1], [0, 0]),  # Edge from (0,0,0) to (0,1,0)
            ([0, 0], [0, 0], [0, 1]),  # Edge from (0,0,0) to (0,0,1)
            ([1, 0], [0, 1], [0, 0]),  # Edge from (1,0,0) to (0,1,0)
            ([1, 0], [0, 0], [0, 1]),  # Edge from (1,0,0) to (0,0,1)
            ([0, 0], [1, 0], [0, 1]),  # Edge from (0,1,0) to (0,0,1)
        ]
        for edge in tetrahedron_edges:
            ax.plot(*edge, color="white", linewidth=1.0)

        # Set up the plot to visualize the unit tetrahedron
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])
        ax.set_zlim([-0.1, 1.1])
        ax.set_box_aspect([1, 1, 1])
        ax.grid(color='white', linestyle=':', linewidth=0.5)

        # Labels and title
        ax.set_title(f"Whitney Form Vector Field: {file_name}", color="white")
        ax.set_xlabel("x", color="white")
        ax.set_ylabel("y", color="white")
        ax.set_zlabel("z", color="white")

        # Save the plot
        plt.savefig(os.path.join(output_dir, f"{file_name.replace('.txt', '.png')}"), facecolor='black')
        plt.close()

    elif data.shape[1] > 3:
        # Extract x, y, z, and coefficients for scalar fields
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        v = data[:, 3:]  # Remaining columns are coefficients

        # Set default properties for white stroke color
        plt.rcParams['text.color'] = 'white'
        plt.rcParams['axes.labelcolor'] = 'white'
        plt.rcParams['xtick.color'] = 'white'
        plt.rcParams['ytick.color'] = 'white'
        plt.rcParams['axes.edgecolor'] = 'white'

        for i, scalar_field in enumerate(v.T):
            # Create the plot
            fig = plt.figure(figsize=(10, 10), facecolor='black')
            ax = fig.add_subplot(111, projection='3d', facecolor='black')

            # Plot the scalar field using scatter with color for magnitude
            scatter = ax.scatter(x, y, z, c=scalar_field, cmap='viridis')
            cbar = fig.colorbar(scatter, ax=ax, label=f"Scalar Value {i}")
            cbar.ax.yaxis.set_tick_params(color='white')
            cbar.outline.set_edgecolor('white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

            # Draw the edges of the tetrahedron
            tetrahedron_edges = [
                ([0, 1], [0, 0], [0, 0]),
                ([0, 0], [0, 1], [0, 0]),
                ([0, 0], [0, 0], [0, 1]),
                ([1, 0], [0, 1], [0, 0]),
                ([1, 0], [0, 0], [0, 1]),
                ([0, 0], [1, 0], [0, 1]),
            ]
            for edge in tetrahedron_edges:
                ax.plot(*edge, color="white", linewidth=1.0)

            # Set up the plot to visualize the unit tetrahedron
            ax.set_xlim([-0.1, 1.1])
            ax.set_ylim([-0.1, 1.1])
            ax.set_zlim([-0.1, 1.1])
            ax.set_box_aspect([1, 1, 1])
            ax.grid(color='white', linestyle=':', linewidth=0.5)

            # Labels and title
            ax.set_title(f"Whitney Form Scalar Field {i}: {file_name}", color="white")
            ax.set_xlabel("x", color="white")
            ax.set_ylabel("y", color="white")
            ax.set_zlabel("z", color="white")

            # Save the plot
            plt.savefig(os.path.join(output_dir, f"{file_name.replace('.txt', f'_scalar_{i}.png')}"), facecolor='black')
            plt.close()

    else:
        raise ValueError(f"Unsupported data format in file {file_name}. Expected 4 or more columns.")
