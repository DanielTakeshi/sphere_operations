import numpy as np
import matplotlib.pyplot as plt


def sample_sphere_surface(center_sphere, radius, n_points=1):
    """Sample about a sphere surface.

    Samples IID standard Gaussians:
        [
            x(1), y(1), z(1)
            x(2), y(2), z(2)
            ...
            x(N), y(N), z(N)
        ]
    Then normalize and multiply each by the radius.
    Suggest keeping n_points quite high to see a difference.
    """
    assert radius > 0
    xyz_N3 = np.random.normal(loc=0.0, scale=1.0, size=(n_points, 3))
    xyz_N3 = xyz_N3 * (radius / np.linalg.norm(xyz_N3, axis=1, keepdims=True))
    return center_sphere + xyz_N3


def visualize(points):
    """Visualize points with 3D scatter plot."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2])
    plt.show()


if __name__ == "__main__":
    center_sphere = np.array([0, 0, 0])
    points = sample_sphere_surface(center_sphere, radius=1.0, n_points=10000)
    visualize(points)