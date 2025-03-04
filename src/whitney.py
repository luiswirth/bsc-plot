import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator

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
            lambda0 = (n_points - i - j) / n_points
            lambda1 = j / n_points
            lambda2 = i / n_points
            
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

def plot_local_whitney_form(vertices, i, j, form_name, filename):
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
    n_grid = 100  # Increased resolution
    points = generate_triangle_grid(vertices, n_grid)
    x, y = points[:, 0], points[:, 1]
    
    # Create triangulation
    triang = Triangulation(x, y)
    
    # Calculate vector field at each point
    vectors = np.array([whitney_form([xi, yi], vertices, i, j, gradients) for xi, yi in points])
    u, v = vectors[:, 0], vectors[:, 1]
    magnitude = np.sqrt(u**2 + v**2)
    
    # Plot magnitude as a heatmap using tripcolor with flat shading for clearer boundaries
    _im = ax.tripcolor(triang, magnitude, cmap='viridis', shading='flat')
    
    # Generate fewer quiver points
    quiver_points = generate_triangle_grid(vertices, 12)
    quiver_vectors = np.array([whitney_form([p[0], p[1]], vertices, i, j, gradients) for p in quiver_points])
    quiver_u, quiver_v = quiver_vectors[:, 0], quiver_vectors[:, 1]
    quiver_mag = np.sqrt(quiver_u**2 + quiver_v**2)
    
    # Normalize for direction
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_u = np.where(quiver_mag > 0, quiver_u / quiver_mag, 0)
        norm_v = np.where(quiver_mag > 0, quiver_v / quiver_mag, 0)
    
    # Plot vector field with thinner arrows
    ax.quiver(quiver_points[:, 0], quiver_points[:, 1], 
              norm_u, norm_v,
              angles='xy', scale_units='xy', scale=20, 
              pivot='tail', color='white', alpha=0.8, 
              width=0.004)  # Thinner arrows
    
    # Plot triangle boundary
    triangle_boundary = np.vstack([vertices, vertices[0]])
    ax.plot(triangle_boundary[:, 0], triangle_boundary[:, 1], 'k-', linewidth=1.5)
    
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

# New functions for global shape functions on a mesh

def create_simple_mesh():
    """Create a simple mesh with 4 triangles.
    
    Returns:
        vertices: Nx2 array of vertex coordinates
        triangles: Mx3 array of triangle indices
        edges: dict mapping global edge (v1,v2) to list of adjacent triangle indices
    """

    vertices = np.array([
        [0, 0], # center left
        [1, 0], # center right
        [0.5, np.sqrt(3)/2], # center top
        [-0.5, np.sqrt(3)/2], # top left
        [1.5, np.sqrt(3)/2], # top right
        [0.5, -np.sqrt(3)/2], # bottom center
    ])
    
    # Define triangles (as indices into the vertices array)
    triangles = np.array([
        [0, 1, 2], # center
        [0, 2, 3], # left
        [1, 4, 2], # right
        [1, 0, 5], # bottom
    ])
    
    # Create edge-to-triangles mapping
    edges = {}
    for t_idx, triangle in enumerate(triangles):
        for i in range(3):
            v1, v2 = triangle[i], triangle[(i+1)%3]
            # Ensure edge is stored with lower index first for consistency
            edge = tuple(sorted([v1, v2]))
            if edge not in edges:
                edges[edge] = []
            edges[edge].append(t_idx)
    
    return vertices, triangles, edges

def compute_whitney_global(point, vertices, triangles, global_edge, triangle_idx=None):
    """Compute global Whitney form for a specific edge at a point.
    
    Args:
        point: (x,y) coordinates
        vertices: global vertex coordinates
        triangles: global triangle indices
        global_edge: tuple (v1,v2) of global vertex indices defining the edge
        triangle_idx: index of triangle containing the point (if None, will find it)
        
    Returns:
        2D vector [dx_component, dy_component] of the Whitney form at the point
        or zeros if point is not in a triangle containing the edge
    """
    # Sort edge vertices for consistency
    global_edge = tuple(sorted(global_edge))
    
    # Find which triangle the point is in, if not provided
    if triangle_idx is None:
        for idx, tri in enumerate(triangles):
            # Get triangle vertices
            tri_vertices = vertices[tri]
            
            # Check if point is inside this triangle
            lambdas = compute_barycentric_coordinates(point, tri_vertices)
            if np.all(lambdas >= -1e-10) and np.all(lambdas <= 1 + 1e-10):
                triangle_idx = idx
                break
        
        if triangle_idx is None:
            # Point is not in any triangle
            return np.zeros(2)
    
    # Get global vertex indices of the current triangle
    global_tri_indices = triangles[triangle_idx]
    
    # Check if the global edge is part of this triangle
    edge_in_triangle = (
        (global_edge[0] in global_tri_indices and global_edge[1] in global_tri_indices)
    )
    
    if not edge_in_triangle:
        # Edge is not part of this triangle
        return np.zeros(2)
    
    # Map global indices to local (0,1,2) indices
    local_indices = {}
    for local_idx, global_idx in enumerate(global_tri_indices):
        local_indices[global_idx] = local_idx
    
    # Get local indices of the edge vertices
    local_i = local_indices[global_edge[0]]
    local_j = local_indices[global_edge[1]]
    
    # Get the triangle vertices
    tri_vertices = vertices[global_tri_indices]
    
    # Calculate Whitney form for this triangle
    return whitney_form(point, tri_vertices, local_i, local_j)

def plot_global_whitney_form(vertices, triangles, edges, global_edge, filename):
    """Plot a global Whitney 1-form across the entire mesh.
    
    Args:
        vertices: global vertex coordinates
        triangles: global triangle indices
        edges: mapping from global edges to adjacent triangles
        global_edge: tuple (v1,v2) of global vertex indices defining the edge to plot
        filename: output file name
    """
    # Sort edge vertices for consistency
    global_edge = tuple(sorted(global_edge))
    
    # Check if edge exists in mesh
    if global_edge not in edges:
        print(f"Edge {global_edge} not found in mesh")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # First plot the triangles with color fill
    for t_idx, triangle in enumerate(triangles):
        tri_vertices = vertices[triangle]
        
        # Check if this triangle contains the global edge
        has_edge = (global_edge[0] in triangle and global_edge[1] in triangle)
        
        # Determine if we should fill this triangle
        if has_edge:
            # Create triangulation for this triangle
            tri_points = generate_triangle_grid(tri_vertices, 50)
            x, y = tri_points[:, 0], tri_points[:, 1]
            triang = Triangulation(x, y)
            
            # Map global indices to local indices
            local_indices = {g_idx: l_idx for l_idx, g_idx in enumerate(triangle)}
            local_i = local_indices[global_edge[0]]
            local_j = local_indices[global_edge[1]]
            
            # Calculate vector field at each point
            vectors = np.array([whitney_form([xi, yi], tri_vertices, local_i, local_j) 
                               for xi, yi in zip(x, y)])
            magnitude = np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2)
            
            # Plot magnitude as heatmap using tripcolor with flat shading
            ax.tripcolor(triang, magnitude, cmap='viridis', shading='flat')
            
            # Add arrows inside this triangle
            quiver_points = generate_triangle_grid(tri_vertices, 8)
            quiver_vectors = np.array([whitney_form([p[0], p[1]], tri_vertices, local_i, local_j) 
                                     for p in quiver_points])
            quiver_u, quiver_v = quiver_vectors[:, 0], quiver_vectors[:, 1]
            quiver_mag = np.sqrt(quiver_u**2 + quiver_v**2)
            
            # Normalize for direction
            with np.errstate(divide='ignore', invalid='ignore'):
                norm_u = np.where(quiver_mag > 0, quiver_u / quiver_mag, 0)
                norm_v = np.where(quiver_mag > 0, quiver_v / quiver_mag, 0)
            
            # Plot vector field with thinner arrows
            ax.quiver(quiver_points[:, 0], quiver_points[:, 1], 
                      norm_u, norm_v,
                      angles='xy', scale_units='xy', scale=25,
                      pivot='tail', color='white', alpha=0.8,
                      width=0.003, headwidth=4, headlength=5)  # Thinner arrows
    
    # Plot mesh edges
    for triangle in triangles:
        tri_vertices = vertices[triangle]
        # Close the loop
        tri_vertices = np.vstack([tri_vertices, tri_vertices[0]])
        ax.plot(tri_vertices[:, 0], tri_vertices[:, 1], 'k-', linewidth=1.0)
    
    # Highlight the edge of interest
    edge_vertices = vertices[list(global_edge)]
    ax.plot(edge_vertices[:, 0], edge_vertices[:, 1], 'r-', linewidth=2.5)
    
    # Add vertex labels
    for i, (x, y) in enumerate(vertices):
        ax.text(x, y, f'${i}$', fontsize=14, ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis')
    sm.set_array([])  # Fake an array for the colorbar
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Magnitude', fontsize=14)
    
    # Set title and labels
    v1, v2 = global_edge
    ax.set_title(f'Global Whitney Form: $\\lambda_{{{v1}{v2}}}$', fontsize=20)
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    ax.set_aspect('equal')
    
    # Set limits with small margin
    min_x, min_y = np.min(vertices, axis=0)
    max_x, max_y = np.max(vertices, axis=0)
    margin = 0.1
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved as '{filename}'")
    plt.close(fig)

def plot_local_whitneys():
    """Plot local Whitney form basis functions on different triangles."""
    # Reference triangle
    ref_triangle = np.array([[0, 0], [1, 0], [0, 1]])
    plot_local_whitney_form(ref_triangle, 0, 1, "λ₀₁", "out/ref_lambda01.png")
    plot_local_whitney_form(ref_triangle, 0, 2, "λ₀₂", "out/ref_lambda02.png")
    plot_local_whitney_form(ref_triangle, 1, 2, "λ₁₂", "out/ref_lambda12.png")
    
    # Equilateral triangle
    eq_triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    plot_local_whitney_form(eq_triangle, 0, 1, "λ₀₁", "out/eq_lambda01.png")
    plot_local_whitney_form(eq_triangle, 0, 2, "λ₀₂", "out/eq_lambda02.png")
    plot_local_whitney_form(eq_triangle, 1, 2, "λ₁₂", "out/eq_lambda12.png")

def plot_global_whitneys():

    # Create simple mesh
    vertices, triangles, edges = create_simple_mesh()
    
    # Plot global Whitney forms for different edges
    plot_global_whitney_form(vertices, triangles, edges, (0, 1), "out/global_lambda01.png")
    plot_global_whitney_form(vertices, triangles, edges, (0, 2), "out/global_lambda02.png")
    plot_global_whitney_form(vertices, triangles, edges, (1, 2), "out/global_lambda12.png")

def main():
    """Main function to create either local or global Whitney form plots."""
    import os
    os.makedirs("out", exist_ok=True)
    
    #plot_local_whitneys()
    plot_global_whitneys()

if __name__ == "__main__":
    main()
