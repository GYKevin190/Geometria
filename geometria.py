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

def plot_fractal(title, ratios, angles_deg, iterations):
    start = np.array([0.0, 0.0])
    end = np.array([1.0, 0.0])
    curve = generate_generalized_koch(start, end, iterations, ratios, angles_deg)
    x_vals, y_vals = zip(*curve)

    plt.figure(figsize=(12, 3))
    plt.plot(x_vals, y_vals, color='darkblue')
    plt.title(f'{title} (iteráció: {iterations})')
    plt.axis('equal')
    plt.axis('off')
    plt.show()

# Globális iterációszám
iterations = 6

# Fraktál definíciók
fraktalok = [
    {
        "title": "Klasszikus Koch-görbe ❄️",
        "ratios": [1/3, 1/3, 1/3, 1/3],
        "angles_deg": [0, 60, -60, 0]
    },
    {
        "title": "Négyzetes Koch-görbe 🟦",
        "ratios": [1/4]*8,
        "angles_deg": [0, 90, -90, 0, 0, -90, 90, 0]
    },
    {
        "title": "Levy C-görbe 🌿",
        "ratios": [np.sqrt(2)/2, np.sqrt(2)/2],
        "angles_deg": [45, -45]
    },
    {
        "title": "Recés / cikkcakk fraktál 🔀",
        "ratios": [0.5, 0.5],
        "angles_deg": [45, -45]
    },
    {
        "title": "Hullámvonal fraktál 🌊",
        "ratios": [1/4, 1/4, 1/4, 1/4],
        "angles_deg": [0, 30, -30, 0]
    },
]

# Fraktálok generálása
for fraktal in fraktalok:
    plot_fractal(fraktal["title"], fraktal["ratios"], fraktal["angles_deg"], iterations)
