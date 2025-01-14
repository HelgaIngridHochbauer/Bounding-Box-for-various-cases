
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import time
from PIL import Image
import gradio as gr
import matplotlib.patches as patches
import random
import io
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon


@dataclass
class Rectangle:
    """Rectangle representation with center, axes, extents, and area."""
    center: np.ndarray
    axis: List[np.ndarray]
    extent: List[float]
    area: float

def generate_random_points(n: int, min_val: float = -10, max_val: float = 10) -> List[Tuple[float, float]]:
    """Generate n random 2D points within the specified range."""
    points = []
    for _ in range(n):
        x = np.random.uniform(min_val, max_val)
        y = np.random.uniform(min_val, max_val)
        points.append((x, y))
    return points

def compute_convex_hull(points: List[np.ndarray]) -> List[np.ndarray]:
    """Compute the convex hull of a set of points using Graham scan."""
    if len(points) < 3:
        return points

    # Find point with lowest y-coordinate (and leftmost if tied)
    pivot = min(points, key=lambda p: (p[1], p[0]))

    # Sort points based on polar angle and distance from pivot
    def sort_key(p):
        if np.array_equal(p, pivot):
            return (-float('inf'), 0)
        angle = np.arctan2(p[1] - pivot[1], p[0] - pivot[0])
        dist = np.sum((p - pivot) ** 2)
        return (angle, dist)

    sorted_points = sorted(points, key=sort_key)

    # Graham scan algorithm
    hull = []
    for point in sorted_points:
        while len(hull) >= 2:
            p1, p2 = hull[-2], hull[-1]
            cross = np.cross(p2 - p1, point - p1)
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(point)

    return hull

def perpendicular(v: np.ndarray) -> np.ndarray:
    """Returns a vector perpendicular to the input vector."""
    return np.array([-v[1], v[0]])

def min_area_rectangle_of_points(points: List[Tuple[float, float]]) -> Rectangle:
    """Compute the minimum-area rectangle containing the points."""
    if len(points) < 3:
        raise ValueError("At least 3 points are required")

    np_points = [np.array(p) for p in points]
    polygon = compute_convex_hull(np_points)

    min_rect = Rectangle(
        center=np.zeros(2),
        axis=[np.zeros(2), np.zeros(2)],
        extent=[0.0, 0.0],
        area=float('inf')
    )

    n = len(polygon)
    for i0 in range(n):
        i1 = (i0 + 1) % n
        origin = polygon[i0]
        U0 = polygon[i1] - origin
        length = np.linalg.norm(U0)

        if length < 1e-10:
            continue

        U0 = U0 / length
        U1 = perpendicular(U0)

        min0, max0 = float('inf'), float('-inf')
        min1, max1 = float('inf'), float('-inf')

        for point in polygon:
            D = point - origin
            dot0 = np.dot(U0, D)
            dot1 = np.dot(U1, D)

            min0 = min(min0, dot0)
            max0 = max(max0, dot0)
            min1 = min(min1, dot1)
            max1 = max(max1, dot1)

        area = (max0 - min0) * (max1 - min1)

        if area < min_rect.area:
            center = origin + ((min0 + max0) / 2) * U0 + ((min1 + max1) / 2) * U1
            min_rect = Rectangle(
                center=center,
                axis=[U0, U1],
                extent=[(max0 - min0) / 2, (max1 - min1) / 2],
                area=area
            )

    return min_rect

def plot_result(points: List[Tuple[float, float]], rect: Rectangle, computation_time: float):
    """Plot the points and minimum area rectangle."""
    points = np.array(points)
    plt.figure(figsize=(10, 10))

    # Plot points
    plt.scatter(points[:, 0], points[:, 1], c='blue', label='Points')

    # Plot rectangle
    center = rect.center
    axis0, axis1 = rect.axis
    extent0, extent1 = rect.extent

    corners = [
        center + extent0 * axis0 + extent1 * axis1,
        center + extent0 * axis0 - extent1 * axis1,
        center - extent0 * axis0 - extent1 * axis1,
        center - extent0 * axis0 + extent1 * axis1,
        center + extent0 * axis0 + extent1 * axis1
    ]

    corners = np.array(corners)
    plt.plot(corners[:, 0], corners[:, 1], 'r-', linewidth=2, label='Minimum Rectangle')
    plt.scatter([center[0]], [center[1]], c='red', marker='x', s=100, label='Rectangle Center')

    # Plot convex hull
    hull_points = compute_convex_hull([np.array(p) for p in points])
    hull_points.append(hull_points[0])  # Close the hull
    hull_points = np.array(hull_points)
    plt.plot(hull_points[:, 0], hull_points[:, 1], 'g--', label='Convex Hull')

    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title(f'Minimum Area Rectangle\nArea: {rect.area:.2f}\nComputation Time: {computation_time:.4f} seconds')
    plt.show()

def generate_random_squares(num_squares, max_center=10, max_side=5):
    return [
        (random.uniform(-max_center, max_center),  # x_center
         random.uniform(-max_center, max_center),  # y_center
         random.uniform(1, max_side))  # side_length
        for _ in range(num_squares)
    ]

def generate_random_discs(num_discs, max_center=10, max_radius=5):
    return [
        (random.uniform(-max_center, max_center),  # x_center
         random.uniform(-max_center, max_center),  # y_center
         random.uniform(1, max_radius))  # radius
        for _ in range(num_discs)
    ]

def bounding_box_for_shapes(squares=None, discs=None):
    x_min_global = float('inf')
    x_max_global = float('-inf')
    y_min_global = float('inf')
    y_max_global = float('-inf')

    if squares:
        for x_center, y_center, side_length in squares:
            half_side = side_length / 2
            x_min = x_center - half_side
            x_max = x_center + half_side
            y_min = y_center - half_side
            y_max = y_center + half_side

            x_min_global = min(x_min_global, x_min)
            x_max_global = max(x_max_global, x_max)
            y_min_global = min(y_min_global, y_min)
            y_max_global = max(y_max_global, y_max)

    if discs:
        for x_center, y_center, radius in discs:
            x_min = x_center - radius
            x_max = x_center + radius
            y_min = y_center - radius
            y_max = y_center + radius

            x_min_global = min(x_min_global, x_min)
            x_max_global = max(x_max_global, x_max)
            y_min_global = min(y_min_global, y_min)
            y_max_global = max(y_max_global, y_max)

    return x_min_global, x_max_global, y_min_global, y_max_global

def plot_bounding_box(squares, discs, bounding_box):
    fig, ax = plt.subplots(figsize=(10, 10))

    if squares:
        for x_center, y_center, side_length in squares:
            half_side = side_length / 2
            rect = patches.Rectangle(
                (x_center - half_side, y_center - half_side),
                side_length, side_length,
                linewidth=1, edgecolor='blue', facecolor='none'
            )
            ax.add_patch(rect)

    if discs:
        for x_center, y_center, radius in discs:
            circle = patches.Circle(
                (x_center, y_center),
                radius,
                linewidth=1, edgecolor='green', facecolor='none'
            )
            ax.add_patch(circle)

    x_min, x_max, y_min, y_max = bounding_box
    bbox_rect = patches.Rectangle(
        (x_min, y_min),
        x_max - x_min, y_max - y_min,
        linewidth=2, edgecolor='red', facecolor='none', label='Bounding Box'
    )
    ax.add_patch(bbox_rect)

    margin = 1
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend()
    ax.set_title('Bounding Box for Shapes')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    plt.grid(True)

def bounding_box_for_imprecise_points(num_squares=0, num_discs=0):
    squares = generate_random_squares(int(num_squares)) if num_squares > 0 else []
    discs = generate_random_discs(int(num_discs)) if num_discs > 0 else []

    start_time = time.time()
    bounding_box = bounding_box_for_shapes(squares, discs)
    end_time = time.time()

    buffer = io.BytesIO()
    plot_bounding_box(squares, discs, bounding_box)
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)

    image = Image.open(buffer)
    return f"Bounding box: {bounding_box}\nTime taken: {end_time - start_time:.6f} seconds", image

def calculate_min_bounding_box_with_animation(points):
    def rotate_points(points, angle):
        """Rotate points by a given angle (in radians)."""
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        return np.dot(points, rotation_matrix.T)

    hull = ConvexHull(points)  # Compute convex hull
    hull_points = points[hull.vertices]
    num_hull_points = len(hull_points)
    min_area = float('inf')
    min_box = None
    frames = []

    for i in range(num_hull_points):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % num_hull_points]
        edge_vector = p2 - p1
        edge_angle = np.arctan2(edge_vector[1], edge_vector[0])  # Calculate edge angle

        rotated_points = rotate_points(hull_points, -edge_angle)  # Rotate points to align edge horizontally
        min_x, max_x = np.min(rotated_points[:, 0]), np.max(rotated_points[:, 0])
        min_y, max_y = np.min(rotated_points[:, 1]), np.max(rotated_points[:, 1])
        width, height = max_x - min_x, max_y - min_y
        area = width * height

        if area < min_area:  # Update minimum bounding box
            min_area = area
            min_box = np.array([
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y],
            ])
            min_box = rotate_points(min_box, edge_angle)  # Rotate box back to original space

        # Save frame for animation
        current_box = np.array([
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
        ])
        frames.append((hull_points.copy(), rotate_points(current_box, edge_angle)))

    return min_box, min_area, frames


def plot_rotating_calipers_animation(points, frames):
    """
    Creates an animation showing the Rotating Calipers method.

    Args:
        points (ndarray): Original set of points.
        frames (list): List of frames containing hull points and bounding box at each step.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    def update(frame):
        ax.clear()
        hull_points, bounding_box = frame

        # Scatter points
        ax.scatter(points[:, 0], points[:, 1], color="blue", label="Points")
        # Plot convex hull
        ax.plot(hull_points[:, 0], hull_points[:, 1], 'r--', label="Convex Hull")
        ax.plot([hull_points[-1, 0], hull_points[0, 0]],
                [hull_points[-1, 1], hull_points[0, 1]], 'r--')  # Close hull shape

        # Display bounding box
        bounding_polygon = Polygon(bounding_box, closed=True, edgecolor="green", facecolor="none", linewidth=2)
        ax.add_patch(bounding_polygon)

        # Set plot limits and properties
        ax.set_xlim(points[:, 0].min() - 5, points[:, 0].max() + 5)
        ax.set_ylim(points[:, 1].min() - 5, points[:, 1].max() + 5)
        ax.set_title("Rotating Calipers - Bounding Box")
        ax.set_aspect('equal', adjustable='box')
        ax.legend()

    anim = FuncAnimation(fig, update, frames=frames, interval=800, repeat=False)

    # Save as a GIF file
    writer = PillowWriter(fps=2)
    gif_path = "/tmp/rotating_calipers.gif"  # Save the GIF to a file
    anim.save(gif_path, writer=writer)

    return gif_path


def input_polygon_points():
    n_points = int(input("Enter the number of points for the polygon: "))
    points = []
    for i in range(n_points):
        x = float(input(f"Enter x-coordinate for point {i + 1}: "))
        y = float(input(f"Enter y-coordinate for point {i + 1}: "))
        points.append([x, y])
    return np.array(points)

# Function to find the convex hull
def convex_hull(points):
    hull = ConvexHull(points)
    return points[hull.vertices]

# Exhaustive Search Method (simple approach)
def bounding_box_exhaustive(points):
    min_x = min(points[:, 0])
    max_x = max(points[:, 0])
    min_y = min(points[:, 1])
    max_y = max(points[:, 1])
    return min_x, min_y, max_x, max_y

# Rotate points by a given angle (in radians)
def rotate_points(points, angle):
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return np.dot(points, rotation_matrix.T)

# Full Rotating Calipers Implementation
def rotating_calipers(points):
    hull = ConvexHull(points)  # Compute convex hull
    hull_points = points[hull.vertices]
    num_hull_points = len(hull_points)
    min_area = float('inf')
    min_box = None

    # Loop through each edge of the convex hull
    for i in range(num_hull_points):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % num_hull_points]
        edge_vector = p2 - p1
        edge_angle = np.arctan2(edge_vector[1], edge_vector[0])  # Calculate edge angle

        # Rotate points so that edge aligns horizontally
        rotated_points = rotate_points(hull_points, -edge_angle)
        min_x, max_x = np.min(rotated_points[:, 0]), np.max(rotated_points[:, 0])
        min_y, max_y = np.min(rotated_points[:, 1]), np.max(rotated_points[:, 1])
        width, height = max_x - min_x, max_y - min_y
        area = width * height

        if area < min_area:  # Update minimum bounding box
            min_area = area
            min_box = np.array([
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y],
            ])
            # Rotate bounding box back to original space
            min_box = rotate_points(min_box, edge_angle)

    return min_box, min_area

# Hybrid Approach
def hybrid_bounding_box(points):
    # Compute convex hull
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    # Proper convexity check
    def is_convex(polygon_points):
        n = len(polygon_points)
        for i in range(n):
            p1, p2, p3 = polygon_points[i], polygon_points[(i + 1) % n], polygon_points[(i + 2) % n]
            cross_product = np.cross(p2 - p1, p3 - p2)
            if cross_product < 0:  # If any cross product is negative, not convex
                return False
        return True

    if is_convex(points):
        # Convex polygon - Use Rotating Calipers
        print("Using Rotating Calipers algorithm.")
        return rotating_calipers(hull_points)
    else:
        # Non-convex polygon - Use Exhaustive Search on Convex Hull
        print("Using Exhaustive Search algorithm.")
        min_x, min_y, max_x, max_y = bounding_box_exhaustive(hull_points)
        bounding_box = np.array([
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
        ])
        return bounding_box, (max_x - min_x) * (max_y - min_y)

def process_hybrid_bounding_box(num_points, coordinates_str):
    try:
        # Parse coordinates
        points = []
        if coordinates_str.strip():
            for pair in coordinates_str.split(";"):
                x, y = map(float, pair.split(","))
                points.append([x, y])
            points = np.array(points)
        else:
            raise ValueError("Coordinates cannot be empty.")

        # Validate the number of points
        if len(points) != num_points:
            raise ValueError("Number of points doesn't match the provided coordinates.")

        # Get bounding box using hybrid approach
        min_box, min_area = hybrid_bounding_box(points)

        # Create a visualization
        buffer = io.BytesIO()
        plt.figure(figsize=(7, 7))
        fig, ax = plt.subplots()

        # Ensure the polygon visually closes by appending the first point at the end
        polygon_points_closed = np.vstack([points, points[0]])
        ax.plot(polygon_points_closed[:, 0], polygon_points_closed[:, 1], color='r', linestyle='-', linewidth=2)

        # Handling bounding box output dynamically
        if isinstance(min_box[0], np.ndarray):
            # Rotating Calipers case
            x_coords = [point[0] for point in min_box]
            y_coords = [point[1] for point in min_box]
            min_x = min(x_coords)
            max_x = max(x_coords)
            min_y = min(y_coords)
            max_y = max(y_coords)
        else:
            # Exhaustive search case
            min_x, min_y, max_x, max_y = min_box

        # Drawing the bounding box
        plt.plot([min_x, max_x], [min_y, min_y], color='b')
        plt.plot([min_x, max_x], [max_y, max_y], color='b')
        plt.plot([min_x, min_x], [min_y, max_y], color='b')
        plt.plot([max_x, max_x], [min_y, max_y], color='b')

        # Dynamically adjust plot limits
        all_x = points[:, 0]
        all_y = points[:, 1]
        padding = 1  # Optional padding
        plt.xlim(min(all_x) - padding, max(all_x) + padding)
        plt.ylim(min(all_y) - padding, max(all_y) + padding)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("Polygon and Bounding Box Visualization")
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)
        image = Image.open(buffer)

        # Prepare results
        result_text = f"Minimum bounding box area: {min_area:.2f}\nBounding Box Coordinates: {min_box.tolist()}"
        return result_text, image

    except Exception as e:
        return str(e), None


# Gradio integration
def rotating_calipers_algorithm(n_points: int):
    points = np.random.rand(n_points, 2) * 100
    min_box, min_area, animation_frames = calculate_min_bounding_box_with_animation(points)
    gif_path = plot_rotating_calipers_animation(points, animation_frames)

    result_text = f"Minimum bounding box area: {min_area:.2f}\n"

    # Return the result text and the file path to the GIF
    return result_text, gif_path

def exhaustive_search(n_points: int):
    try:
        # Generate points
        points = [np.array(p) for p in generate_random_points(n_points)]

        # Compute the minimum area rectangle
        start_time = time.time()
        rect = min_area_rectangle_of_points(points)
        search_time = time.time() - start_time

        # Prepare result text
        result_text = (
            f"Number of points: {n_points}\n"
            f"Rectangle center: ({rect.center[0]:.2f}, {rect.center[1]:.2f})\n"
            f"Rectangle area: {rect.area:.2f}\n"
            f"Rectangle extents: [{rect.extent[0]:.2f}, {rect.extent[1]:.2f}]\n"
            f"Exhaustive search time: {search_time:.4f} seconds"
        )

        # Plot and save the result
        plot_result(points, rect, search_time)
        plot_path = "result_plot.png"
        plt.savefig(plot_path)
        plt.close()

        return result_text, plot_path
    except Exception as e:
        # Handle errors gracefully
        return f"Error: {str(e)}", None



with gr.Blocks() as interface:
    gr.Markdown("## Bounding Box Algorithms")

    with gr.Tab("Exhaustive search"):
        n_points_input = gr.Number(label="Number of Points", value=10, minimum=3)
        result_text = gr.Text(label="Results")
        result_plot = gr.Image(label="Plot")
        submit_button = gr.Button("Run")
        submit_button.click(
            fn=exhaustive_search,
            inputs=[n_points_input],
            outputs=[result_text, result_plot],
        )

    with gr.Tab("Rotating Calipers"):
        n_points_input_rotating = gr.Number(label="Number of Points", value=20, minimum=3)
        result_text_rotating = gr.Text(label="Results")
        animation_output = gr.Image(label="Rotating Calipers Animation", type="filepath")
        submit_button_rotating = gr.Button("Run")
        submit_button_rotating.click(
            fn=rotating_calipers_algorithm,
            inputs=[n_points_input_rotating],
            outputs=[result_text_rotating, animation_output],
        )

    with gr.Tab("Bounding Box for Imprecise Points"):
      num_squares_input = gr.Number(label="Number of Squares")
      num_discs_input = gr.Number(label="Number of Discs")
      results_text_output = gr.Text(label="Results")
      bounding_box_image_output = gr.Image(label="Bounding Box Visualization")

      def bounding_box_fn(num_squares, num_discs):
          return bounding_box_for_imprecise_points(num_squares, num_discs)

      # Add a new submit button for this tab
      submit_button_bounding_box = gr.Button("Run")
      submit_button_bounding_box.click(
          fn=bounding_box_fn,
          inputs=[num_squares_input, num_discs_input],
          outputs=[results_text_output, bounding_box_image_output],
    )

    with gr.Tab("Hybrid Bounding Box"):
      with gr.Row():
        num_points_input = gr.Number(
            label="Number of Points of the Polygon",
            value=5,
            precision=0,
            interactive=True,
            info="Enter the number of points for the polygon."
        )

        coordinates_input = gr.Textbox(
            label="Coordinates",
            placeholder="Enter coordinates as x1,y1;x2,y2;... (e.g., 0,0;1,2;3,4)",
            interactive=True,
            info="Provide coordinates separated by semicolons."
        )

        submit_button_bounding_box = gr.Button("Run")

        # Define output components
        output_text = gr.Text(label="Results")
        output_image = gr.Image(label="Plot")

        # Link the button click to the processing function
        submit_button_bounding_box.click(
            fn=process_hybrid_bounding_box,  # Reference the function, do not call it here
            inputs=[num_points_input, coordinates_input],  # Inputs for the function
            outputs=[output_text, output_image]  # Outputs from the function
        )

interface.launch()

