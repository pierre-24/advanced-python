"""
Test ``ball_search()``.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from neighbour_search.kdtree_ns import KDTreeNeighbourSearch as NeighbourSearch

RADIUS = .25
N = 100

def plot_p(points: np.ndarray, mouse_pos: tuple[float, float]):
    ax.clear()

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    in_circle = ns.ball_search(np.array(mouse_pos), RADIUS)

    ax.plot(points[:, 0], points[:, 1], 'o')
    ax.plot(points[in_circle, 0], points[in_circle, 1], 'o')

    ax.add_patch(Circle(mouse_pos, RADIUS, fill=False, ec='black', lw=1))

    figure.canvas.draw()
    figure.canvas.flush_events()


if __name__ == '__main__':
    points = 2 * (np.random.random((N, 2)) - [.5, .5])
    ns = NeighbourSearch(points)

    figure = plt.figure(figsize=(5, 5))
    ax = figure.subplots()
    plot_p(points, (0, 0))

    def on_move_event(event):
        if event.inaxes:
            plot_p(points, (event.xdata, event.ydata))

    plt.connect('motion_notify_event', on_move_event)

    plt.show()
