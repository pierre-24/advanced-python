"""
Functional test for ``ball_search()``.
Display a graph with points.
Mouse allows you to move a circle (the "ball") with your mouse.
All points withing ball should be highlighted.
"""
import argparse

import numpy as np
from matplotlib.patches import Circle

from neighbour_search.scripts import BaseFunctionalTestWithNaiveImpl, BaseFunctionalTestWithKDTreeImpl, \
    BaseFunctionalTestWithBallTreeImpl


class BallSearchTestMixin:
    """
    Test the ``ball_search()`` function.
    Move the mouse to highlight the one within ``radius`` of the mouse position.
    """

    radius = .25

    def replot_test(self):

        in_circle = self.ns.ball_search(np.array(self.mouse_position), self.radius)

        self.ax.plot(self.points[:, 0], self.points[:, 1], 'o', ms=4)
        self.ax.plot(self.points[in_circle, 0], self.points[in_circle, 1], 'o', ms=5)

        self.ax.add_patch(Circle(self.mouse_position, self.radius, fill=False, ec='black', lw=1))


class NaiveTest(BallSearchTestMixin, BaseFunctionalTestWithNaiveImpl):
    def __init__(self, radius: float, N: int, show_ns: bool = False):
        super().__init__(N, show_ns)
        self.radius = radius


class KDTreeTest(BallSearchTestMixin, BaseFunctionalTestWithKDTreeImpl):
    def __init__(self, radius: float, N: int, show_ns: bool = False):
        super().__init__(N, show_ns)
        self.radius = radius


class BallTreeTest(BallSearchTestMixin, BaseFunctionalTestWithBallTreeImpl):
    def __init__(self, radius: float, N: int, show_ns: bool = False):
        super().__init__(N, show_ns)
        self.radius = radius


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-N', type=int, default=100, help='number of points')
    parser.add_argument('-s', action='store_true', help='show the tree structure')
    parser.add_argument('-t', choices=['naive', 'ball', 'kd'], default='naive', help='NS method')

    parser.add_argument('-r', type=float, default=.25, help='radius of the ball')

    args = parser.parse_args()

    if args.t == 'naive':
        NaiveTest(args.r, args.N, args.s).main()
    elif args.t == 'ball':
        BallTreeTest(args.r, args.N, args.s).main()
    else:
        KDTreeTest(args.r, args.N, args.s).main()
