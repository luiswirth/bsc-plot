import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

def generate_triangle_grid(vertices, n_points=20):
    """Generate a grid of points within an arbitrary triangle.
    
    Args:
        vertices: 3x2 array of triangle vertex coordinates [(x0,y0), (x1,y1), (x2,y2)]
        n_points: number of points along each edge
        
    Returns:
        Array of (x,y) points inside the triangle
    """
    # Create evenly spaced points using barycentric coordinates
    points = []
    for i in range(n_points+1):
        for j in range(n_points+1-i):
            # Barycentric coordinates
            lambda0 = (n_points - i - j) / n_points
            lambda1 = j / n_points
            lambda2 = i / n_points
            
            # Convert to Cartesian coordinates using the formula:
            # x = lambda0*x0 + lambda1*x1 + lambda2*x2
            # y = lambda0*y0 + lambda1*y1 + lambda2*y2
            x = lambda0 * vertices[0, 0] + lambda1 * vertices[1, 0] + lambda2 * vertices[2, 0]
            y = lambda0 * vertices[0, 1] + lambda1 * vertices[1, 1] + lambda2 * vertices[2, 1]
            
            points.append([x, y])
    
    return np.array(points)

def compute_barycentric_gradients(vertices):
    """Compute gradients of barycentric coordinate functions.
    
    Args:
        vertices: 3x2 array of triangle vertex coordinates [(x0,y0), (x1,y1), (x2,y2)]
        
    Returns:
        Array of gradient vectors for each barycentric coordinate function [del_lambda0, del_lambda1, del_lambda2]
        Each gradient is a vector [dx, dy]
    """
    # Extract vertices
    x0, y0 = vertices[0]
    x1, y1 = vertices[1]
    x2, y2 = vertices[2]
    
    # Compute area of the triangle
    area = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    
    # Compute gradients of lambda_i
    # These are derived from solving the linear system for barycentric coordinates
    # and then taking gradients
    grad_lambda0 = np.array([(y1 - y2) / (2 * area), (x2 - x1) / (2 * area)])
    grad_lambda1 = np.array([(y2 - y0) / (2 * area), (x0 - x2) / (2 * area)])
    grad_lambda2 = np.array([(y0 - y1) / (2 * area), (x1 - x0) / (2 * area)])
    
    return np.array([grad_lambda0, grad_lambda1, grad_lambda2])

def compute_barycentric_coordinates(point, vertices, gradients=None):
    """Compute barycentric coordinates for a point in a triangle.
    
    Args:
        point: (x,y) coordinates
        vertices: 3x2 array of triangle vertex coordinates [(x0,y0), (x1,y1), (x2,y2)]
        gradients: optional precomputed gradients from compute_barycentric_gradients()
        
    Returns:
        Array [lambda0, lambda1, lambda2] of barycentric coordinates
    """
    # Extract point coordinates
    x, y = point
    
    # Extract vertices
    x0, y0 = vertices[0]
    x1, y1 = vertices[1]
    x2, y2 = vertices[2]
    
    # Compute area of the triangle
    area = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    
    # Compute sub-triangle areas
    area0 = 0.5 * ((x1 - x) * (y2 - y) - (x2 - x) * (y1 - y))
    area1 = 0.5 * ((x2 - x) * (y0 - y) - (x0 - x) * (y2 - y))
    area2 = 0.5 * ((x0 - x) * (y1 - y) - (x1 - x) * (y0 - y))
    
    # Compute barycentric coordinates
    lambda0 = area0 / area
    lambda1 = area1 / area
    lambda2 = area2 / area
    
    return np.array([lambda0, lambda1, lambda2])

def whitney_form(point, vertices, i, j, gradients=None):
    """Whitney 1-form lambda_ij = lambda_i * grad(lambda_j) - lambda_j * grad(lambda_i)
    
    Args:
        point: (x,y) coordinates
        vertices: 3x2 array of triangle vertex coordinates
        i, j: indices (0,1,2) of the vertices defining the edge
        gradients: optional precomputed gradients from compute_barycentric_gradients()
        
    Returns:
        2D vector [dx_component, dy_component] of the Whitney form at the point
    """
    if gradients is None:
        gradients = compute_barycentric_gradients(vertices)
    
    # Compute barycentric coordinates
    lambdas = compute_barycentric_coordinates(point, vertices)
    
    # Get the gradients
    grad_i = gradients[i]
    grad_j = gradients[j]
    
    # Compute Whitney form
    whitney_vector = lambdas[i] * grad_j - lambdas[j] * grad_i
    
    return whitney_vector

def plot_whitney_form(vertices, i, j, form_name, filename):
    """Plot a Whitney 1-form as a vector field with magnitude heatmap for an arbitrary triangle.
    
    Args:
        vertices: 3x2 array of triangle vertex coordinates
        i, j: indices (0,1,2) of the vertices defining the edge
        form_name: name for the plot title
        filename: output file name
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Precompute gradients for efficiency
    gradients = compute_barycentric_gradients(vertices)
    
    # Generate triangulation for the triangle
    n_grid = 50
    points = generate_triangle_grid(vertices, n_grid)
    x, y = points[:, 0], points[:, 1]
    
    # Create triangulation
    triang = Triangulation(x, y)
    
    # Calculate vector field at each point
    vectors = np.array([whitney_form([xi, yi], vertices, i, j, gradients) for xi, yi in points])
    u, v = vectors[:, 0], vectors[:, 1]
    magnitude = np.sqrt(u**2 + v**2)
    
    # Plot magnitude as a heatmap using tripcolor
    _im = ax.tripcolor(triang, magnitude, cmap='viridis', shading='gouraud')
    
    # Generate fewer quiver points
    quiver_points = generate_triangle_grid(vertices, 15)
    quiver_vectors = np.array([whitney_form([p[0], p[1]], vertices, i, j, gradients) for p in quiver_points])
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
    triangle_boundary = np.vstack([vertices, vertices[0]])
    ax.plot(triangle_boundary[:, 0], triangle_boundary[:, 1], 'k-', linewidth=1.5)
    
    # Add vertex labels
    #for k, (vx, vy) in enumerate(vertices):
    #    ax.text(vx, vy, f'$v_{k}$', fontsize=14, ha='center', va='center',
    #            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))
    
    fontsize = 20
    ax.set_title(f'{form_name}', fontsize=fontsize)
    ax.set_xlabel('x', fontsize=fontsize)
    ax.set_ylabel('y', fontsize=fontsize)
    ax.set_aspect('equal')
    
    # Set limits with a small margin around the triangle
    min_x, max_x = np.min(vertices[:, 0]), np.max(vertices[:, 0])
    min_y, max_y = np.min(vertices[:, 1]), np.max(vertices[:, 1])
    margin = 0.02 * max(max_x - min_x, max_y - min_y)
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved as '{filename}'")
    plt.close(fig)

def main():
    """Create separate plots for each Whitney 1-form on different triangles."""
   
    # Plot Whitney forms on reference triangle
    ref_triangle = np.array([[0, 0], [1, 0], [0, 1]])
    plot_whitney_form(ref_triangle, 0, 1, "λ₀₁", "out/ref_lambda01.png")
    plot_whitney_form(ref_triangle, 0, 2, "λ₀₂", "out/ref_lambda02.png")
    plot_whitney_form(ref_triangle, 1, 2, "λ₁₂", "out/ref_lambda12.png")
    
    # Plot Whitney forms on equilateral triangle
    eq_triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    plot_whitney_form(eq_triangle, 0, 1, "λ₀₁", "out/eq_lambda01.png")
    plot_whitney_form(eq_triangle, 0, 2, "λ₀₂", "out/eq_lambda02.png")
    plot_whitney_form(eq_triangle, 1, 2, "λ₁₂", "out/eq_lambda12.png")

if __name__ == "__main__":
    main()
