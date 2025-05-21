import matplotlib.pyplot as plt
import numpy as np

def generate_generalized_koch(p0, p1, iterations, ratios, angles_deg):
    if iterations == 0:
        return [p0, p1]

    base_vec = p1 - p0
    base_len = np.linalg.norm(base_vec)
    base_dir = base_vec / base_len
    base_angle = np.arctan2(base_dir[1], base_dir[0])

    current_point = p0
    points = [p0]
    for r, angle in zip(ratios, angles_deg):
        segment_len = base_len * r
        total_angle = base_angle + np.radians(angle)
        direction = np.array([np.cos(total_angle), np.sin(total_angle)])
        next_point = current_point + direction * segment_len
        subcurve = generate_generalized_koch(current_point, next_point, iterations - 1, ratios, angles_deg)
        points.extend(subcurve[1:])
        current_point = next_point

    return points

def estimate_fractal_dimension(n_segments, scaling_ratio):
    if n_segments <= 1 or scaling_ratio <= 0:
        return 1.0  
    return np.log(n_segments) / np.log(1 / scaling_ratio)

def compute_curve_length(points):
    return sum(np.linalg.norm(np.array(points[i+1]) - np.array(points[i])) for i in range(len(points)-1))

def analyze_and_plot(title, ratios, angles_deg, iterations):
    start = np.array([0.0, 0.0])
    end = np.array([1.0, 0.0])
    curve = generate_generalized_koch(start, end, iterations, ratios, angles_deg)
    x_vals, y_vals = zip(*curve)

    length = compute_curve_length(curve)
    num_points = len(curve)

    theoretical_D = None
    if len(set(ratios)) == 1:
        r = ratios[0]
        n = len(ratios)
        theoretical_D = estimate_fractal_dimension(n, r)

    plt.figure(figsize=(12, 4))
    plt.plot(x_vals, y_vals, color='darkblue')
    plt.title(f'{title} (iteration: {iterations})', fontsize=14)
    plt.axis('equal')
    plt.axis('off')

    info_text = f"Iterations: {iterations}\nPoints: {num_points}\nLength: {length:.5f}"
    if theoretical_D:
        info_text += f"\nFractal dimension: {theoretical_D:.5f}"

    plt.gca().text(
        0.01, 0.99, info_text,
        transform=plt.gca().transAxes,  
        fontsize=10,
        ha='left', va='top',
        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round')
    )

    plt.show()


iterations = 6

fractals = [
    {
        "title": "Classical Koch curve",
        "ratios": [1/3, 1/3, 1/3, 1/3],
        "angles_deg": [0, 60, -60, 0]
    },
    {
        "title": "Square Koch curve",
        "ratios": [1/4]*8,
        "angles_deg": [0, 90, -90, 0, 0, -90, 90, 0]
    },
    {
        "title": "Levy C-curve 🌿",
        "ratios": [np.sqrt(2)/2, np.sqrt(2)/2],
        "angles_deg": [45, -45]
    },
    {
        "title": "Zigzag fractal",
        "ratios": [0.5, 0.5],
        "angles_deg": [45, -45]
    },
    {
        "title": "Wavy line fractal",
        "ratios": [1/4, 1/4, 1/4, 1/4],
        "angles_deg": [0, 30, -30, 0]
    },
]

for fractal in fractals:
    analyze_and_plot(fractal["title"], fractal["ratios"], fractal["angles_deg"], iterations)