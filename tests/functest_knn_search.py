"""
Functional test for ``knn_search()``.
Display a graph with points.
Mouse allows you to select the closest point with your mouse.
Its ``k`` closest neighbour should be highlighted.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

# from neighbour_search.naive_ns import NaiveNeighbourSearch as NeighbourSearch
from neighbour_search.kdtree_ns import KDTreeNeighbourSearch as NeighbourSearch
from neighbour_search.kdtree_ns import KDTreeNode, KDTreeLeaf


class KNNSearchTest:
    """
    Test the ``knn_search()`` function.

    Display a [-1, 1] x [-1, 1] graph containing ``N`` points.
    Move the mouse to highlight a point and its ``k`` closest neighbour.
    """

    def __init__(self, k: int = 5, N: int = 100, min_leaf: int = 3):
        self.k = k
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
            KNNSearchTest.plot_ns_node(ax, node.left, (xr[0], node.midpoint), yr)
            KNNSearchTest.plot_ns_node(ax, node.right, (node.midpoint, xr[1]), yr)
        else:
            ax.hlines(node.midpoint, *xr, color='gray', lw=2.0 - node.depth * .3)
            KNNSearchTest.plot_ns_node(ax, node.left, xr, (yr[0], node.midpoint))
            KNNSearchTest.plot_ns_node(ax, node.right, xr, (node.midpoint, yr[1]))

    def replot(self):
        """(re)Plot everything."""
        self.ax.clear()

        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)

        self.plot_ns_node(self.ax, self.ns.root)

        # select closest point
        in_circle = self.ns.ball_search(np.array(self.mouse_position), .25)
        look_for = np.argmin(np.linalg.norm(self.points[in_circle] - self.mouse_position, axis=1))
        look_for = in_circle[look_for]

        # select `self.k` neighbours
        in_circle = self.ns.knn_search(look_for, self.k)

        self.ax.plot(self.points[:, 0], self.points[:, 1], 'o', ms=4)
        self.ax.plot(self.points[in_circle, 0], self.points[in_circle, 1], 'o', ms=5)
        self.ax.plot(self.points[look_for, 0], self.points[look_for, 1], 'o', ms=5)

        self.ax.add_patch(
            Circle(
                self.points[look_for],
                np.max(np.linalg.norm(self.points[in_circle] - self.points[look_for], axis=1)),
                fill=False, ec='black', lw=1
            ))

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=100)
    parser.add_argument('-k', type=int, default=5)
    parser.add_argument('-m', type=int, default=5)

    args = parser.parse_args()

    KNNSearchTest(args.k, args.N, args.m).main()
