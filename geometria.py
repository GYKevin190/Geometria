import numpy as np
import matplotlib.pyplot as plt

class KochCurve2D:
    """Computational layer for generating 2D Koch curves."""
    def __init__(self, ratios, angles_deg):
        self.ratios = ratios
        self.angles_deg = angles_deg

    def generate_curve(self, p0, p1, iterations):
        """Generate fractal points recursively."""
        if iterations == 0:
            return [p0, p1]

        base_vec = p1 - p0
        base_len = np.linalg.norm(base_vec)
        base_dir = base_vec / base_len
        base_angle = np.arctan2(base_dir[1], base_dir[0])

        current_point = p0
        points = [p0]
        for r, angle in zip(self.ratios, self.angles_deg):
            segment_len = base_len * r
            total_angle = base_angle + np.radians(angle)
            direction = np.array([np.cos(total_angle), np.sin(total_angle)])
            next_point = current_point + direction * segment_len
            subcurve = self.generate_curve(current_point, next_point, iterations - 1)
            points.extend(subcurve[1:])
            current_point = next_point
        return points

    @staticmethod
    def calculate_length(points):
        """Calculate total curve length."""
        return sum(np.linalg.norm(np.array(points[i+1]) - np.array(points[i])) for i in range(len(points)-1))

    def estimate_dimension(self):
        """Estimate fractal dimension (for uniform ratios)."""
        if len(set(self.ratios)) == 1:
            return np.log(len(self.ratios)) / np.log(1 / self.ratios[0])
        return None

class FractalVisualizer2D:
    """Visualization layer for 2D fractals."""
    @staticmethod
    def plot(fractal, title, iterations, curve_points):
        length = KochCurve2D.calculate_length(curve_points)
        num_points = len(curve_points)
        theoretical_D = fractal.estimate_dimension()

        plt.figure(figsize=(12, 4))
        x_vals, y_vals = zip(*curve_points)
        plt.plot(x_vals, y_vals, color='darkblue')
        plt.title(f'{title} (Iteration: {iterations})')
        plt.axis('equal')
        plt.axis('off')

        info_text = f"Iterations: {iterations}\nPoints: {num_points}\nLength: {length:.5f}"
        if theoretical_D:
            info_text += f"\nFractal Dimension: {theoretical_D:.5f}"

        plt.gca().text(0.01, 0.99, info_text, transform=plt.gca().transAxes,
                      fontsize=10, ha='left', va='top',
                      bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        plt.show()


if __name__ == "__main__":
    fractals = [
        {"title": "Classical Koch Curve", "ratios": [1/3, 1/3, 1/3, 1/3], "angles_deg": [0, 60, -60, 0]},
        {"title": "Square Koch Curve", "ratios": [1/4]*8, "angles_deg": [0, 90, -90, 0, 0, -90, 90, 0]},
        {"title": "Levy C-Curve", "ratios": [np.sqrt(2)/2, np.sqrt(2)/2], "angles_deg": [45, -45]},
        {"title": "Zigzag Fractal", "ratios": [0.5, 0.5], "angles_deg": [45, -45]},
        {"title": "Wavy Fractal", "ratios": [1/4, 1/4, 1/4, 1/4], "angles_deg": [0, 30, -30, 0]}
    ]

    for fractal in fractals:
        generator = KochCurve2D(fractal["ratios"], fractal["angles_deg"])
        curve = generator.generate_curve(np.array([0.0, 0.0]), np.array([1.0, 0.0]), iterations=5)
        FractalVisualizer2D.plot(generator, fractal["title"], iterations=5, curve_points=curve)