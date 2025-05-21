import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rotate_vector(vec, axis, angle_deg):
    angle_rad = np.radians(angle_deg)
    axis = axis / np.linalg.norm(axis)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    ux, uy, uz = axis
    rot_matrix = np.array([
        [cos_a + ux**2*(1-cos_a),     ux*uy*(1-cos_a) - uz*sin_a, ux*uz*(1-cos_a) + uy*sin_a],
        [uy*ux*(1-cos_a) + uz*sin_a,  cos_a + uy**2*(1-cos_a),    uy*uz*(1-cos_a) - ux*sin_a],
        [uz*ux*(1-cos_a) - uy*sin_a,  uz*uy*(1-cos_a) + ux*sin_a, cos_a + uz**2*(1-cos_a)]
    ])
    return rot_matrix @ vec

def generate_koch_3d(p0, p1, iterations, ratios, rotation_axes, angles_deg):
    if iterations == 0:
        return [p0, p1]

    base_vec = p1 - p0
    base_len = np.linalg.norm(base_vec)
    base_dir = base_vec / base_len

    current_point = p0
    points = [p0]
    for r, axis, angle in zip(ratios, rotation_axes, angles_deg):
        segment_len = base_len * r
        new_dir = rotate_vector(base_dir, axis, angle)
        next_point = current_point + new_dir * segment_len
        subcurve = generate_koch_3d(current_point, next_point, iterations - 1, ratios, rotation_axes, angles_deg)
        points.extend(subcurve[1:])
        current_point = next_point

    return points

def compute_curve_length(points):
    return sum(np.linalg.norm(points[i+1] - points[i]) for i in range(len(points)-1))

def estimate_fractal_dimension(n_segments, scaling_ratio):
    if n_segments <= 1 or scaling_ratio <= 0:
        return 1.0
    return np.log(n_segments) / np.log(1 / scaling_ratio)

def plot_koch_3d_with_info(points, title, iterations, scaling_ratio=None):
    length = compute_curve_length(points)
    num_points = len(points)
    theoretical_D = None

    if scaling_ratio:
        theoretical_D = estimate_fractal_dimension(len(points) - 1, scaling_ratio)

    xs, ys, zs = zip(*points)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, ys, zs, color='navy')

    ax.set_title(f"{title} (iteration: {iterations})", fontsize=14)
    ax.set_axis_off()

    info = f"Iterations: {iterations}\nPoints: {num_points}\nLength: {length:.5f}"
    if theoretical_D:
        info += f"\nFractal dimension: {theoretical_D:.5f}"

    ax.text2D(0.01, 0.99, info, transform=ax.transAxes,
              fontsize=10, va='top', ha='left',
              bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

    plt.tight_layout()
    plt.show()

iterations = 4
ratios = [1/3, 1/3, 1/3]
rotation_axes = [
    np.array([0, 0, 1]),   
    np.array([1, 0, 0]),   
    np.array([0, 1, 0])    
]
angles_deg = [0, 60, -60]

p0 = np.array([0.0, 0.0, 0.0])
p1 = np.array([1.0, 0.0, 0.0])
curve3d = generate_koch_3d(p0, p1, iterations, ratios, rotation_axes, angles_deg)

plot_koch_3d_with_info(curve3d, title="3D Koch curve", iterations=iterations, scaling_ratio=ratios[0])