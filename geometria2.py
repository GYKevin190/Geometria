import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class KochCurve3D:
    """Computational layer for 3D Koch curves."""
    def __init__(self, ratios, rotation_axes, angles_deg):
        self.ratios = ratios
        self.rotation_axes = rotation_axes
        self.angles_deg = angles_deg

    def generate_curve(self, p0, p1, iterations):
        """Generate 3D fractal points."""
        if iterations == 0:
            return [p0, p1]

        base_vec = p1 - p0
        base_len = np.linalg.norm(base_vec)
        base_dir = base_vec / base_len

        current_point = p0
        points = [p0]
        for r, axis, angle in zip(self.ratios, self.rotation_axes, self.angles_deg):
            segment_len = base_len * r
            new_dir = self.rotate_vector(base_dir, axis, angle)
            next_point = current_point + new_dir * segment_len
            subcurve = self.generate_curve(current_point, next_point, iterations - 1)
            points.extend(subcurve[1:])
            current_point = next_point
        return points

    @staticmethod
    def rotate_vector(vec, axis, angle_deg):
        """Rotate vector in 3D space."""
        angle_rad = np.radians(angle_deg)
        axis = axis / np.linalg.norm(axis)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        ux, uy, uz = axis
        rot_matrix = np.array([
            [cos_a + ux**2*(1-cos_a), ux*uy*(1-cos_a) - uz*sin_a, ux*uz*(1-cos_a) + uy*sin_a],
            [uy*ux*(1-cos_a) + uz*sin_a, cos_a + uy**2*(1-cos_a), uy*uz*(1-cos_a) - ux*sin_a],
            [uz*ux*(1-cos_a) - uy*sin_a, uz*uy*(1-cos_a) + ux*sin_a, cos_a + uz**2*(1-cos_a)]
        ])
        return rot_matrix @ vec

    @staticmethod
    def calculate_length(points):
        """Calculate curve length."""
        return sum(np.linalg.norm(points[i+1] - points[i]) for i in range(len(points)-1))

    def estimate_dimension(self):
        """Estimate fractal dimension."""
        if len(set(self.ratios)) == 1:
            return np.log(len(self.ratios)) / np.log(1 / self.ratios[0])
        return None

class FractalVisualizer3D:
    """Visualization layer for 3D fractals."""
    @staticmethod
    def plot(fractal, title, iterations, curve_points):
        length = KochCurve3D.calculate_length(curve_points)
        num_points = len(curve_points)
        theoretical_D = fractal.estimate_dimension()

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        xs, ys, zs = zip(*curve_points)
        ax.plot(xs, ys, zs, color='navy')
        ax.set_title(f"{title} (Iteration: {iterations})")
        ax.set_axis_off()

        info_text = f"Iterations: {iterations}\nPoints: {num_points}\nLength: {length:.5f}"
        if theoretical_D:
            info_text += f"\nFractal Dimension: {theoretical_D:.5f}"

        ax.text2D(0.01, 0.99, info_text, transform=ax.transAxes,
                fontsize=10, va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    generator = KochCurve3D(
        ratios=[1/3, 1/3, 1/3],
        rotation_axes=[np.array([0, 0, 1]), np.array([1, 0, 0]), np.array([0, 1, 0])],
        angles_deg=[0, 60, -60]
    )
    curve = generator.generate_curve(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), iterations=4)
    FractalVisualizer3D.plot(generator, "3D Koch Fractal", iterations=4, curve_points=curve)