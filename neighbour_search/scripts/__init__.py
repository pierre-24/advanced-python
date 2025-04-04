"""
This contains a set of functional tests/example
"""

from matplotlib.patches import Circle
import numpy as np
import matplotlib.pyplot as plt

from neighbour_search.naive_ns import NaiveNeighbourSearch
from neighbour_search.kdtree_ns import KDTreeNeighbourSearch, KDTreeNode, KDTreeLeaf
from neighbour_search.balltree_ns import BallTreeNeighbourSearch, BallTreeNode, BallTreeLeaf


class BaseFunctionalTest:

    NeighbourSearch = None

    radius = .25

    def __init__(self, N: int = 100, show_ns: bool = False):
        self.show_ns = show_ns
        self.points = 2 * (np.random.random((N, 2)) - [.5, .5])

        self.ns = self.NeighbourSearch(self.points)

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

    def replot(self):
        """(re)Plot everything."""
        self.ax.clear()

        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)

        if self.show_ns:
            self.replot_ns()

        self.replot_test()

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def replot_test(self):
        raise NotImplementedError()

    def replot_ns(self):
        raise NotImplementedError()


class BaseFunctionalTestWithNaiveImpl(BaseFunctionalTest):
    NeighbourSearch = NaiveNeighbourSearch

    def replot_ns(self):
        pass


class BaseFunctionalTestWithKDTreeImpl(BaseFunctionalTest):
    NeighbourSearch = KDTreeNeighbourSearch

    def replot_ns(self):
        self.plot_ns_node(self.ns.root)

    def plot_ns_node(
            self,
            node: KDTreeNode | KDTreeLeaf | None,
            xr: tuple[float, float] = (-1, 1), yr: tuple[float, float] = (-1, 1)
    ):
        """Plot a representation of a ``KDTreeNode`` using lines"""
        if not isinstance(node, KDTreeNode):
            return

        if node.depth % 2 == 0:
            self.ax.vlines(node.midpoint, *yr, color='gray', lw=2.0 - node.depth * .3)
            self.plot_ns_node(node.left, (xr[0], node.midpoint), yr)
            self.plot_ns_node(node.right, (node.midpoint, xr[1]), yr)
        else:
            self.ax.hlines(node.midpoint, *xr, color='gray', lw=2.0 - node.depth * .3)
            self.plot_ns_node(node.left, xr, (yr[0], node.midpoint))
            self.plot_ns_node(node.right, xr, (node.midpoint, yr[1]))


class BaseFunctionalTestWithBallTreeImpl(BaseFunctionalTest):
    NeighbourSearch = BallTreeNeighbourSearch

    def replot_ns(self):
        self.plot_ns_node(self.ns.root)

    def plot_ns_node(self, node: BallTreeNode | BallTreeLeaf | None, depth: int = 0):
        """Plot a representation of a ``BallTreeNode`` using a circle"""
        if node is None:
            return

        if isinstance(node, BallTreeLeaf):
            isin = ((node.center - self.mouse_position)**2).sum() < (node.radius + self.radius) ** 2

            self.ax.add_patch(Circle(
                node.center,
                node.radius,
                fill=False,
                ec='blue' if isin else 'grey',
                lw=2.0 if isin else 1.0,
            ))

        if isinstance(node, BallTreeNode):
            self.plot_ns_node(node.left, depth + 1)
            self.plot_ns_node(node.right, depth + 1)
