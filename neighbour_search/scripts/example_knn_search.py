"""
Functional test for ``knn_search()``.
Display a graph with points.
Mouse allows you to select the closest point with your mouse.
Its ``k`` closest neighbour should be highlighted.
"""
import argparse

import numpy as np
from matplotlib.patches import Circle

from neighbour_search.scripts import BaseFunctionalTestWithNaiveImpl, BaseFunctionalTestWithKDTreeImpl, \
    BaseFunctionalTestWithBallTreeImpl


class KNNTestMixin:
    """
    Test the ``knn_search()`` function.
    Move the mouse to highlight a point and its ``k`` closest neighbour.
    """

    k = 5

    def replot_test(self):
        # select closest point
        in_circle = self.ns.ball_search(np.array(self.mouse_position), .25)
        look_for = np.argmin(np.linalg.norm(self.points[in_circle] - self.mouse_position, axis=1))
        look_for = in_circle[look_for]

        # select `self.k` neighbours to `look_for`
        in_circle = self.ns.knn_search(look_for, self.k)

        self.ax.plot(self.points[:, 0], self.points[:, 1], 'o', ms=4)
        self.ax.plot(self.points[in_circle, 0], self.points[in_circle, 1], 'o', ms=5)
        self.ax.plot(self.points[look_for, 0], self.points[look_for, 1], 'o', ms=5)

        self.radius = np.max(np.linalg.norm(self.points[in_circle] - self.points[look_for], axis=1))

        self.ax.add_patch(
            Circle(
                self.points[look_for],
                self.radius,
                fill=False, ec='black', lw=1
            ))


class NaiveTest(KNNTestMixin, BaseFunctionalTestWithNaiveImpl):
    def __init__(self, k: int, N: int, show_ns: bool = False):
        super().__init__(N, show_ns)
        self.k = k


class KDTreeTest(KNNTestMixin, BaseFunctionalTestWithKDTreeImpl):
    def __init__(self, k: int, N: int, show_ns: bool = False):
        super().__init__(N, show_ns)
        self.k = k


class BallTreeTest(KNNTestMixin, BaseFunctionalTestWithBallTreeImpl):
    def __init__(self, k: int, N: int, show_ns: bool = False):
        super().__init__(N, show_ns)
        self.k = k


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-N', type=int, default=100, help='number of points')
    parser.add_argument('-s', action='store_true', help='show the tree structure')
    parser.add_argument('-t', choices=['naive', 'ball', 'kd'], default='naive', help='NS method')

    parser.add_argument('-k', type=int, default=5, help='number of neighbours')

    args = parser.parse_args()

    if args.t == 'naive':
        NaiveTest(args.k, args.N, args.s).main()
    elif args.t == 'ball':
        BallTreeTest(args.k, args.N, args.s).main()
    else:
        KDTreeTest(args.k, args.N, args.s).main()
