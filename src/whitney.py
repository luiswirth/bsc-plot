import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

def generate_triangle_grid(n_points=20):
    """Generate a grid of points within the reference triangle."""
    # Create evenly spaced points within triangle
    points = []
    for i in range(n_points+1):
        for j in range(n_points+1-i):
            # Convert to Cartesian coordinates
            x = j / n_points
            y = (n_points - i - j) / n_points
            points.append([x, y])
    
    return np.array(points)

def whitney_lambda_01(x, y):
    """Whitney 1-form lambda_01 = (1-y)dx + xdy"""
    return np.array([(1-y), x])

def whitney_lambda_02(x, y):
    """Whitney 1-form lambda_02 = ydx + (1-x)dy"""
    return np.array([y, (1-x)])

def whitney_lambda_12(x, y):
    """Whitney 1-form lambda_12 = -ydx + xdy"""
    return np.array([-y, x])

def plot_whitney_form(form_func, form_name, filename):
    """Plot a Whitney 1-form as a vector field with magnitude heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Generate triangulation for the reference triangle
    n_grid = 50  # Reduced for performance
    points = generate_triangle_grid(n_grid)
    x, y = points[:, 0], points[:, 1]
    
    # Create triangulation
    triang = Triangulation(x, y)
    
    # Calculate vector field at each point
    vectors = np.array([form_func(xi, yi) for xi, yi in points])
    u, v = vectors[:, 0], vectors[:, 1]
    magnitude = np.sqrt(u**2 + v**2)
    
    # Plot magnitude as a heatmap using tripcolor
    im = ax.tripcolor(triang, magnitude, cmap='viridis', shading='gouraud')
    
    # Generate fewer quiver points
    quiver_points = generate_triangle_grid(15)
    quiver_vectors = np.array([form_func(p[0], p[1]) for p in quiver_points])
    quiver_u, quiver_v = quiver_vectors[:, 0], quiver_vectors[:, 1]
    quiver_mag = np.sqrt(quiver_u**2 + quiver_v**2)
    
    # Normalize for direction
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_u = np.where(quiver_mag > 0, quiver_u / quiver_mag, 0)
        norm_v = np.where(quiver_mag > 0, quiver_v / quiver_mag, 0)
    
    # Plot vector field
    ax.quiver(quiver_points[:, 0], quiver_points[:, 1], 
              norm_u, norm_v,
              angles='xy', scale_units='xy', scale=15, 
              pivot='tail', color='white', alpha=0.8)
    
    # Plot triangle boundary
    triangle_vertices = np.array([[0, 0], [1, 0], [0, 1], [0, 0]])
    ax.plot(triangle_vertices[:, 0], triangle_vertices[:, 1], 'k-', linewidth=1.5)
    
    # Add colorbar
    #cbar = plt.colorbar(im, ax=ax)
    #cbar.set_label('Magnitude')
    
    fontsize = 20
    ax.set_title(f'{form_name}', fontsize=fontsize)
    ax.set_xlabel('x', fontsize=fontsize)
    ax.set_ylabel('y', fontsize=fontsize)
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved as '{filename}'")
    plt.close(fig)

def main():
    """Create separate plots for each Whitney 1-form."""
    plot_whitney_form(whitney_lambda_01, "λ₀₁ = (1-y)dx + xdy", "out/ref_lambda01.png")
    plot_whitney_form(whitney_lambda_02, "λ₀₂ = ydx + (1-x)dy", "out/ref_lambda02.png")
    plot_whitney_form(whitney_lambda_12, "λ₁₂ = -ydx + xdy", "out/ref_lambda12.png")

if __name__ == "__main__":
    main()
