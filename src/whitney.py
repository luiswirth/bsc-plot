import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# --- Barycentric Coordinates and Triangle Utilities ---

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
    grad_lambda0 = np.array([(y1 - y2) / (2 * area), (x2 - x1) / (2 * area)])
    grad_lambda1 = np.array([(y2 - y0) / (2 * area), (x0 - x2) / (2 * area)])
    grad_lambda2 = np.array([(y0 - y1) / (2 * area), (x1 - x0) / (2 * area)])
    
    return np.array([grad_lambda0, grad_lambda1, grad_lambda2])

def compute_barycentric_coordinates(point, vertices):
    """Compute barycentric coordinates for a point in a triangle.
    
    Args:
        point: (x,y) coordinates
        vertices: 3x2 array of triangle vertex coordinates [(x0,y0), (x1,y1), (x2,y2)]
        
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

# --- Whitney Form Functions ---

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

def evaluate_fe_solution(point, vertices, triangle, coefficients, gradients=None):
    """Evaluate a finite element solution at a point inside a triangle.
    
    Args:
        point: (x,y) coordinates
        vertices: global vertex coordinates
        triangle: indices of the triangle vertices
        coefficients: dictionary mapping edge (v1,v2) to coefficient value
        gradients: optional precomputed gradients for the triangle
        
    Returns:
        2D vector [dx_component, dy_component] of the combined vector field
    """
    if gradients is None:
        # Get the triangle vertices
        tri_vertices = vertices[triangle]
        gradients = compute_barycentric_gradients(tri_vertices)
    
    # Initialize result vector
    result = np.zeros(2)
    
    # Get the triangle vertices
    tri_vertices = vertices[triangle]
    
    # Local edges in the triangle (in CCW order)
    local_edges = [
        (0, 1),
        (1, 2),
        (2, 0)
    ]
    
    # For each local edge, get its global edge and corresponding coefficient
    for local_i, local_j in local_edges:
        # Map to global vertex indices
        global_i = triangle[local_i]
        global_j = triangle[local_j]
        
        # Sort to match the dictionary key format
        global_edge = tuple(sorted([global_i, global_j]))
        
        # If this edge has a coefficient, add its contribution
        if global_edge in coefficients:
            coeff = coefficients[global_edge]
            
            # Handle orientation: if the sorted edge reversed the original orientation,
            # we need to negate the Whitney form
            edge_orientation = 1
            if (global_i, global_j) != global_edge:  # If sorting changed the order
                edge_orientation = -1
            
            # Compute Whitney form for this edge and scale by coefficient
            w_vector = whitney_form(point, tri_vertices, local_i, local_j, gradients)
            result += coeff * edge_orientation * w_vector
    
    return result

# --- Mesh Creation and Handling ---

def create_single_triangle_mesh(vertices):
    """Create a mesh with a single triangle.
    
    Args:
        vertices: 3x2 array of triangle vertex coordinates
        
    Returns:
        vertices: same as input
        triangles: 1x3 array with indices [0, 1, 2]
        edges: dict mapping edge (v1,v2) to list of adjacent triangle indices
    """
    triangles = np.array([[0, 1, 2]])  # Single triangle
    
    # Create edge-to-triangles mapping
    edges = {
        (0, 1): [0],
        (1, 2): [0],
        (0, 2): [0]
    }
    
    return vertices, triangles, edges

def create_simple_mesh():
    """Create a simple mesh with 4 triangles.
    
    Returns:
        vertices: Nx2 array of vertex coordinates
        triangles: Mx3 array of triangle indices
        edges: dict mapping global edge (v1,v2) to list of adjacent triangle indices
    """
    vertices = np.array([
        [0, 0],                  # center left
        [1, 0],                  # center right
        [0.5, np.sqrt(3)/2],     # center top
        [-0.5, np.sqrt(3)/2],    # top left
        [1.5, np.sqrt(3)/2],     # top right
        [0.5, -np.sqrt(3)/2],    # bottom center
    ])
    
    # Define triangles (as indices into the vertices array)
    triangles = np.array([
        [0, 1, 2],  # center
        [0, 2, 3],  # left
        [1, 4, 2],  # right
        [0, 1, 5],  # bottom
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

def find_triangle_containing_point(point, vertices, triangles):
    """Find the triangle containing a given point.
    
    Args:
        point: (x,y) coordinates
        vertices: global vertex coordinates
        triangles: triangle indices
        
    Returns:
        Index of the triangle containing the point, or None if not found
    """
    for idx, tri in enumerate(triangles):
        # Get triangle vertices
        tri_vertices = vertices[tri]
        
        # Check if point is inside this triangle
        lambdas = compute_barycentric_coordinates(point, tri_vertices)
        if np.all(lambdas >= -1e-10) and np.all(lambdas <= 1 + 1e-10):
            return idx
    
    return None

# --- Plotting Functions ---

def plot_vector_field(ax, points, vectors, scale=20, width=0.004):
    """Helper function to plot vector field with normalized directions.
    
    Args:
        ax: matplotlib axis
        points: array of points where vectors are located
        vectors: array of vector components [u, v]
        scale: scale factor for quiver plot
        width: width of arrows
    """
    u, v = vectors[:, 0], vectors[:, 1]
    magnitude = np.sqrt(u**2 + v**2)
    
    # Normalize for direction
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_u = np.where(magnitude > 0, u / magnitude, 0)
        norm_v = np.where(magnitude > 0, v / magnitude, 0)
    
    # Plot vector field
    ax.quiver(points[:, 0], points[:, 1], 
              norm_u, norm_v,
              angles='xy', scale_units='xy', scale=scale, 
              pivot='tail', color='white', alpha=0.8, 
              width=width, headwidth=4, headlength=6)

def plot_whitney_form(vertices, triangles, edges, edge, filename, title=None, show_labels=True):
    """Plot a Whitney 1-form across a mesh.
    
    Args:
        vertices: vertex coordinates
        triangles: triangle indices
        edges: mapping from edge (v1,v2) to list of adjacent triangle indices
        edge: tuple (v1,v2) of vertex indices defining the edge to plot
        filename: output file name
        title: optional custom title for the plot
        show_labels: whether to show vertex labels
    """
    # Sort edge vertices for consistency
    edge = tuple(sorted(edge))
    
    # Check if edge exists in mesh
    if edge not in edges:
        print(f"Edge {edge} not found in mesh")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # First plot the triangles with color fill
    for t_idx, triangle in enumerate(triangles):
        tri_vertices = vertices[triangle]
        
        # Check if this triangle contains the edge
        has_edge = (edge[0] in triangle and edge[1] in triangle)
        
        # Only process triangles containing the edge
        if has_edge:
            # Create triangulation for this triangle
            tri_points = generate_triangle_grid(tri_vertices, 50)
            x, y = tri_points[:, 0], tri_points[:, 1]
            triang = Triangulation(x, y)
            
            # Map global indices to local indices
            local_indices = {g_idx: l_idx for l_idx, g_idx in enumerate(triangle)}
            local_i = local_indices[edge[0]]
            local_j = local_indices[edge[1]]
            
            # Calculate vector field at each point
            vectors = np.array([whitney_form([xi, yi], tri_vertices, local_i, local_j) 
                              for xi, yi in zip(x, y)])
            magnitude = np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2)
            
            # Plot magnitude as heatmap using tripcolor with flat shading
            ax.tripcolor(triang, magnitude, cmap='viridis', shading='flat')
            
            # Add arrows inside this triangle
            quiver_points = generate_triangle_grid(tri_vertices, 10)
            quiver_vectors = np.array([whitney_form([p[0], p[1]], tri_vertices, local_i, local_j) 
                                    for p in quiver_points])
            
            # Plot vector field
            plot_vector_field(ax, quiver_points, quiver_vectors, scale=15, width=0.004)
    
    # Plot mesh edges
    for triangle in triangles:
        tri_vertices = vertices[triangle]
        # Close the loop
        tri_vertices = np.vstack([tri_vertices, tri_vertices[0]])
        ax.plot(tri_vertices[:, 0], tri_vertices[:, 1], 'k-', linewidth=1.0)
    
    # Highlight the edge of interest
    edge_vertices = vertices[list(edge)]
    ax.plot(edge_vertices[:, 0], edge_vertices[:, 1], 'r-', linewidth=2.5)
    
    # Add vertex labels if requested
    if show_labels:
        for i, (x, y) in enumerate(vertices):
            ax.text(x, y, f'${i}$', fontsize=14, ha='center', va='center', 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))
    
    # Set title and labels
    v1, v2 = edge
    fontsize = 20
    if title:
        ax.set_title(title, fontsize=fontsize)
    else:
        ax.set_title(f'Whitney Form $\\varphi_{{{v1}{v2}}}$', fontsize=fontsize)
    
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    ax.set_aspect('equal')
    
    # Set limits with small margin
    min_x, min_y = np.min(vertices, axis=0)
    max_x, max_y = np.max(vertices, axis=0)
    margin = 0.1 * max(max_x - min_x, max_y - min_y)
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved as '{filename}'")
    plt.close(fig)

def plot_fe_solution(vertices, triangles, edges, coefficients, filename, title=None, show_labels=True, highlight_edges=None):
    """Plot a finite element solution represented as a linear combination of Whitney forms.
    
    Args:
        vertices: vertex coordinates
        triangles: triangle indices
        edges: mapping from edge (v1,v2) to list of adjacent triangle indices
        coefficients: dictionary mapping edge (v1,v2) to coefficient value
        filename: output file name
        title: optional custom title for the plot
        show_labels: whether to show vertex labels
        highlight_edges: optional list of edges to highlight
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Compute combined vector field and magnitude for each triangle
    max_magnitude = 0  # Track maximum magnitude for colormap normalization
    
    # Plot each triangle
    for t_idx, triangle in enumerate(triangles):
        tri_vertices = vertices[triangle]
        
        # Generate points inside this triangle
        tri_points = generate_triangle_grid(tri_vertices, 50)
        x, y = tri_points[:, 0], tri_points[:, 1]
        triang = Triangulation(x, y)
        
        # Pre-compute gradients for efficiency
        gradients = compute_barycentric_gradients(tri_vertices)
        
        # Calculate vector field at each point
        vectors = np.array([evaluate_fe_solution([xi, yi], vertices, triangle, coefficients, gradients) 
                          for xi, yi in zip(x, y)])
        magnitude = np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2)
        max_magnitude = max(max_magnitude, np.max(magnitude)) if len(magnitude) > 0 else max_magnitude
        
        # Plot magnitude as heatmap using tripcolor with flat shading
        ax.tripcolor(triang, magnitude, cmap='viridis', shading='flat')
        
        # Add arrows inside this triangle
        quiver_points = generate_triangle_grid(tri_vertices, 10)
        quiver_vectors = np.array([evaluate_fe_solution([p[0], p[1]], vertices, triangle, coefficients, gradients) 
                                for p in quiver_points])
        
        # Plot vector field
        plot_vector_field(ax, quiver_points, quiver_vectors, scale=15, width=0.004)
    
    # Plot mesh edges
    for triangle in triangles:
        tri_vertices = vertices[triangle]
        # Close the loop
        tri_vertices = np.vstack([tri_vertices, tri_vertices[0]])
        ax.plot(tri_vertices[:, 0], tri_vertices[:, 1], 'k-', linewidth=1.0)
    
    # Highlight specific edges if requested
    if highlight_edges:
        for edge in highlight_edges:
            edge = tuple(sorted(edge))  # Ensure consistent ordering
            if edge in edges:
                edge_vertices = vertices[list(edge)]
                ax.plot(edge_vertices[:, 0], edge_vertices[:, 1], 'r-', linewidth=2.5)
    
    # Add coefficient values as edge labels
    for edge, coeff in coefficients.items():
        if abs(coeff) > 1e-10:  # Only label non-zero coefficients
            # Edge midpoint for label placement
            v1, v2 = edge
            mid_x = (vertices[v1][0] + vertices[v2][0]) / 2
            mid_y = (vertices[v1][1] + vertices[v2][1]) / 2
            
            # Offset label a bit perpendicular to the edge
            edge_vec = vertices[v2] - vertices[v1]
            norm_vec = np.array([-edge_vec[1], edge_vec[0]])
            norm_vec = 0.05 * norm_vec / np.linalg.norm(norm_vec)
            
            ax.text(mid_x + norm_vec[0], mid_y + norm_vec[1], 
                   f'{coeff:.2f}', fontsize=12, ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))
    
    # Add vertex labels if requested
    if show_labels:
        for i, (x, y) in enumerate(vertices):
            ax.text(x, y, f'${i}$', fontsize=14, ha='center', va='center', 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))
    
    # Set title and labels
    fontsize = 20
    if title:
        ax.set_title(title, fontsize=fontsize)
    else:
        ax.set_title('Whitney Form Solution', fontsize=fontsize)
    
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    ax.set_aspect('equal')
    
    # Set limits with small margin
    min_x, min_y = np.min(vertices, axis=0)
    max_x, max_y = np.max(vertices, axis=0)
    margin = 0.1 * max(max_x - min_x, max_y - min_y)
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved as '{filename}'")
    plt.close(fig)

# --- Main Functions ---

def plot_local_whitneys():
    """Plot local Whitney form basis functions on different triangles."""
    # Reference triangle
    ref_triangle = np.array([[0, 0], [1, 0], [0, 1]])
    ref_vertices, ref_triangles, ref_edges = create_single_triangle_mesh(ref_triangle)
    
    plot_whitney_form(ref_vertices, ref_triangles, ref_edges, (0, 1), 
                      "out/ref_phi01.png", show_labels=False)
    plot_whitney_form(ref_vertices, ref_triangles, ref_edges, (0, 2), 
                      "out/ref_phi02.png", show_labels=False)
    plot_whitney_form(ref_vertices, ref_triangles, ref_edges, (1, 2), 
                      "out/ref_phi12.png", show_labels=False)
    
    # Equilateral triangle
    eq_triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    eq_vertices, eq_triangles, eq_edges = create_single_triangle_mesh(eq_triangle)
    
    plot_whitney_form(eq_vertices, eq_triangles, eq_edges, (0, 1), 
                      "out/eq_phi01.png", show_labels=False)
    plot_whitney_form(eq_vertices, eq_triangles, eq_edges, (0, 2), 
                      "out/eq_phi02.png", show_labels=False)
    plot_whitney_form(eq_vertices, eq_triangles, eq_edges, (1, 2), 
                      "out/eq_phi12.png", show_labels=False)

def plot_global_whitneys():
    """Plot global Whitney form basis functions on a simple mesh."""
    # Create simple mesh
    vertices, triangles, edges = create_simple_mesh()
    
    # Plot global Whitney forms for different edges
    plot_whitney_form(vertices, triangles, edges, (0, 1), "out/mesh_phi01.png")
    plot_whitney_form(vertices, triangles, edges, (0, 2), "out/mesh_phi02.png")
    plot_whitney_form(vertices, triangles, edges, (1, 2), "out/mesh_phi12.png")

def plot_example_solutions():
    """Plot example finite element solutions."""
    # Create simple mesh
    vertices, triangles, edges = create_simple_mesh()
    
    # Example 1: Constant vector field pointing to the right
    coefficients1 = {
        (0, 1): 1.0,    # Horizontal edge in bottom triangle
        (0, 2): 0.5,    # Edge from bottom-left to top
        (1, 2): -0.5,   # Edge from bottom-right to top
        (0, 3): 0.0,    # Left edge
        (2, 3): 0.0,    # Upper left edge
        (1, 4): 0.0,    # Right edge
        (2, 4): 0.0,    # Upper right edge
        (0, 5): 0.0,    # Bottom left edge
        (1, 5): 0.0,    # Bottom right edge
    }
    
    plot_fe_solution(vertices, triangles, edges, coefficients1, 
                    "out/solution_constant.png", 
                    title="Constant Vector Field")
    
    # Example 2: Rotational vector field
    coefficients2 = {
        (0, 1): 0.0,    # Horizontal edge in center triangle
        (0, 2): 1.0,    # Edge from center-left to top
        (1, 2): 1.0,    # Edge from center-right to top
        (0, 3): 1.0,    # Left edge
        (2, 3): -1.0,   # Upper left edge
        (1, 4): -1.0,   # Right edge
        (2, 4): 1.0,    # Upper right edge
        (0, 5): -1.0,   # Bottom left edge
        (1, 5): 1.0,    # Bottom right edge
    }
    
    plot_fe_solution(vertices, triangles, edges, coefficients2, 
                    "out/solution_rotation.png", 
                    title="Rotational Vector Field")
    
    # Example 3: Divergent vector field from center
    coefficients3 = {
        (0, 1): 1.0,    # Horizontal edge in center triangle
        (0, 2): 0.7,    # Edge from center-left to top
        (1, 2): 0.7,    # Edge from center-right to top
        (0, 3): -0.5,   # Left edge
        (2, 3): 0.5,    # Upper left edge
        (1, 4): 0.5,    # Right edge
        (2, 4): 0.5,    # Upper right edge
        (0, 5): 0.7,    # Bottom left edge
        (1, 5): 0.7,    # Bottom right edge
    }
    
    plot_fe_solution(vertices, triangles, edges, coefficients3, 
                    "out/solution_divergent.png", 
                    title="Divergent Vector Field",
                    highlight_edges=[(0, 1), (0, 2), (1, 2)])
    
    # Example 4: Local solution on a single triangle
    eq_triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    eq_vertices, eq_triangles, eq_edges = create_single_triangle_mesh(eq_triangle)
    
    # Simple linear combination of basis functions
    local_coefficients = {
        (0, 1): 1.0,
        (1, 2): 0.5,
        (0, 2): 0.8
    }
    
    plot_fe_solution(eq_vertices, eq_triangles, eq_edges, local_coefficients,
                    "out/solution_local.png",
                    title="Linear Combination on Single Triangle",
                    show_labels=False)

def main():
    """Main function to create local and global Whitney form plots."""
    import os
    os.makedirs("out", exist_ok=True)
    
    plot_local_whitneys()
    print("Local Whitney forms plotted.")
    
    plot_global_whitneys()
    print("Global Whitney forms plotted.")
    
    plot_example_solutions()
    print("Example solutions plotted.")

if __name__ == "__main__":
    main()