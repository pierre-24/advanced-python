"""
Functional test for ``ball_search()``.
Display a graph with points.
Mouse allows you to move a circle (the "ball") with your mouse.
All points withing ball should be highlighted.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from neighbour_search.kdtree_ns import KDTreeNeighbourSearch as NeighbourSearch
from neighbour_search.kdtree_ns import KDTreeNode, KDTreeLeaf


class BallSearchTest:
    """
    Test the ``ball_search()`` function.

    Display a [-1, 1] x [-1, 1] graph containing ``N`` points.
    Move the mouse to highlight the one within ``radius`` of the mouse position.
    """

    def __init__(self, radius: float = .25, N: int = 100, min_leaf: int = 3):
        self.radius = radius
        self.points = 2 * (np.random.random((N, 2)) - [.5, .5])

        self.ns = NeighbourSearch(self.points, max_leaf=min_leaf)

        self.figure = plt.figure(figsize=(8, 8))
        self.ax = self.figure.subplots()

        self.mouse_position = (.0, .0)

    def main(self):
        """Main loop"""
        def on_move_event(event):
            if event.inaxes:
                self.mouse_position = (event.xdata, event.ydata)
                self.replot()

        plt.connect('motion_notify_event', on_move_event)

        self.replot()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_ns_node(
            ax, node: KDTreeNode | KDTreeLeaf | None,
            xr: tuple[float, float] = (-1, 1), yr: tuple[float, float] = (-1, 1)
    ):
        """Plot a representation of a ``KDTreeNode`` using lines"""
        if not isinstance(node, KDTreeNode):
            return

        if node.depth % 2 == 0:
            ax.vlines(node.midpoint, *yr, color='gray', lw=2.0 - node.depth * .3)
            BallSearchTest.plot_ns_node(ax, node.left, (xr[0], node.midpoint), yr)
            BallSearchTest.plot_ns_node(ax, node.right, (node.midpoint, xr[1]), yr)
        else:
            ax.hlines(node.midpoint, *xr, color='gray', lw=2.0 - node.depth * .3)
            BallSearchTest.plot_ns_node(ax, node.left, xr, (yr[0], node.midpoint))
            BallSearchTest.plot_ns_node(ax, node.right, xr, (node.midpoint, yr[1]))

    def replot(self):
        """(re)Plot everything."""
        self.ax.clear()

        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)

        self.plot_ns_node(self.ax, self.ns.root)

        in_circle = self.ns.ball_search(np.array(self.mouse_position), self.radius)

        self.ax.plot(self.points[:, 0], self.points[:, 1], 'o', ms=4)
        self.ax.plot(self.points[in_circle, 0], self.points[in_circle, 1], 'o', ms=5)

        self.ax.add_patch(Circle(self.mouse_position, self.radius, fill=False, ec='black', lw=1))

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=100)
    parser.add_argument('-r', type=float, default=.25)
    parser.add_argument('-m', type=int, default=5)

    args = parser.parse_args()

    BallSearchTest(args.r, args.N, args.m).main()
