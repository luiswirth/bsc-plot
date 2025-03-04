import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

def generate_triangle_grid(n_points=100):
    """Generate a high-resolution grid of points within the reference triangle."""
    # Create points along each edge
    t = np.linspace(0, 1, n_points)
    
    # Generate points inside the triangle using barycentric coordinates
    points = []
    for i in range(n_points):
        for j in range(n_points - i):
            # Barycentric coordinates
            lambda1 = i / (n_points - 1)
            lambda2 = j / (n_points - 1)
            lambda3 = 1 - lambda1 - lambda2
            
            # Only include points where all barycentric coordinates are non-negative
            if lambda1 >= 0 and lambda2 >= 0 and lambda3 >= 0:
                x = lambda2
                y = lambda3
                points.append([x, y])
    
    points = np.array(points)
    return points

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
    """Plot a Whitney 1-form as a vector field with a high-resolution magnitude heatmap."""
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # Generate uniform grid for plotting
    n_grid = 200
    x = np.linspace(0, 1, n_grid)
    y = np.linspace(0, 1, n_grid)
    X, Y = np.meshgrid(x, y)
    
    # Create mask for points inside the triangle
    mask = (X + Y <= 1.0 + 1e-10)
    
    # Calculate vector field and magnitude on the grid
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    magnitude = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if mask[i, j]:
                vec = form_func(X[i, j], Y[i, j])
                U[i, j] = vec[0]
                V[i, j] = vec[1]
                magnitude[i, j] = np.sqrt(vec[0]**2 + vec[1]**2)
    
    # Create triangulation for precise boundary plotting
    triang = Triangulation(X[mask], Y[mask])
    
    # Plot magnitude as a heatmap using tripcolor for exact boundary
    im = ax.tripcolor(X[mask], Y[mask], magnitude[mask], triangles=triang.triangles, 
                      cmap='viridis', shading='gouraud')
    
    # Generate quiver points that stay inside the triangle
    quiver_n = 20
    quiver_points = generate_triangle_grid(quiver_n)
    
    # Calculate vectors at quiver points
    quiver_u = np.zeros(len(quiver_points))
    quiver_v = np.zeros(len(quiver_points))
    quiver_mag = np.zeros(len(quiver_points))
    
    for i, point in enumerate(quiver_points):
        vec = form_func(point[0], point[1])
        quiver_u[i] = vec[0]
        quiver_v[i] = vec[1]
        quiver_mag[i] = np.sqrt(vec[0]**2 + vec[1]**2)
    
    # Normalize for direction
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_u = np.where(quiver_mag > 0, quiver_u / quiver_mag, 0)
        norm_v = np.where(quiver_mag > 0, quiver_v / quiver_mag, 0)
    
    # Plot normalized vector field
    ax.quiver(quiver_points[:, 0], quiver_points[:, 1], 
              norm_u, norm_v, quiver_mag,
              angles='xy', scale_units='xy', scale=20, 
              pivot='tail', cmap='viridis', alpha=0.8)
    
    # Plot triangle boundary
    triangle_vertices = np.array([[0, 0], [1, 0], [0, 1], [0, 0]])
    ax.plot(triangle_vertices[:, 0], triangle_vertices[:, 1], 'k-', linewidth=2)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Magnitude')
    
    ax.set_title(f'Whitney 1-form: {form_name}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved as '{filename}'")
    plt.close(fig)

def main():
    """Create separate plots for each Whitney 1-form."""
    plot_whitney_form(whitney_lambda_01, "λ₀₁ = (1-y)dx + xdy", "whitney_lambda_01.png")
    plot_whitney_form(whitney_lambda_02, "λ₀₂ = ydx + (1-x)dy", "whitney_lambda_02.png")
    plot_whitney_form(whitney_lambda_12, "λ₁₂ = -ydx + xdy", "whitney_lambda_12.png")

if __name__ == "__main__":
    main()