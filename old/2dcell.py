import matplotlib.pyplot as plt
import numpy as np
import os
import glob

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
    if data.shape[1] == 4:
        # Extract x, y, vx, vy for vector field
        x = data[:, 0]
        y = data[:, 1]
        vx = data[:, 2]
        vy = data[:, 3]

        # Compute the magnitude of the vectors
        magnitude = np.sqrt(vx**2 + vy**2)

        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-10
        normalized_vx = vx / (magnitude + epsilon)
        normalized_vy = vy / (magnitude + epsilon)

        # Scale the vectors to have uniform small length
        uniform_scale = 0.03
        vx = normalized_vx * uniform_scale
        vy = normalized_vy * uniform_scale

        # Set default properties for white stroke color
        plt.rcParams['text.color'] = 'white'
        plt.rcParams['axes.labelcolor'] = 'white'
        plt.rcParams['xtick.color'] = 'white'
        plt.rcParams['ytick.color'] = 'white'
        plt.rcParams['axes.edgecolor'] = 'white'

        # Create the plot
        plt.figure(figsize=(8, 8), facecolor='black')

        # Plot the vector field using quiver with normalized vectors and color for magnitude
        quiver = plt.quiver(x, y, vx, vy, magnitude, angles='xy', scale_units='xy', scale=1, cmap='viridis', alpha=0.8)
        cbar = plt.colorbar(quiver, label="Magnitude")
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        # Set up the plot to visualize the unit triangle
        plt.plot([0, 1, 0, 0], [0, 0, 1, 0], color="white", linewidth=2)  # Triangle edges
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().set_facecolor('black')

        # Labels and title
        plt.title(f"Whitney Form Vector Field: {file_name}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.gca().tick_params(colors='white')

        # Save the plot
        plt.savefig(os.path.join(output_dir, f"{file_name.replace('.txt', '.png')}"), facecolor='black')
        plt.close()

    elif data.shape[1] > 2:
        # Extract x, y, and coefficients for scalar fields
        x = data[:, 0]
        y = data[:, 1]
        v = data[:, 2:]  # Remaining columns are coefficients

        # Set default properties for white stroke color
        plt.rcParams['text.color'] = 'white'
        plt.rcParams['axes.labelcolor'] = 'white'
        plt.rcParams['xtick.color'] = 'white'
        plt.rcParams['ytick.color'] = 'white'
        plt.rcParams['axes.edgecolor'] = 'white'

        for i, scalar_field in enumerate(v.T):
            # Create the plot
            plt.figure(figsize=(8, 8), facecolor='black')

            # Plot the scalar field using scatter with color for magnitude
            scatter = plt.scatter(x, y, c=scalar_field, cmap='viridis', alpha=0.8, edgecolors='none')
            cbar = plt.colorbar(scatter, label=f"Scalar Value {i}")
            cbar.ax.yaxis.set_tick_params(color='white')
            cbar.outline.set_edgecolor('white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

            # Set up the plot to visualize the unit triangle
            plt.plot([0, 1, 0, 0], [0, 0, 1, 0], color="white", linewidth=2)  # Triangle edges
            plt.xlim(-0.1, 1.1)
            plt.ylim(-0.1, 1.1)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.gca().set_facecolor('black')

            # Labels and title
            plt.title(f"Whitney Form Scalar Field {i}: {file_name}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.gca().tick_params(colors='white')

            # Save the plot
            plt.savefig(os.path.join(output_dir, f"{file_name.replace('.txt', f'_scalar_{i}.png')}"), facecolor='black')
            plt.close()

    else:
        raise ValueError(f"Unsupported data format in file {file_name}. Expected 3 or more columns.")
