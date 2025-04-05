"""
`Ball tree <https://en.wikipedia.org/wiki/K-d_tree>`_ neighbour search implementation.
Heavily relies on the `triangle inequality <https://en.wikipedia.org/wiki/Triangle_inequality>`_.

+ Creation is :math:`\\mathcal{O}(n\\log n)`.
+ ``knn_search()`` is :math:`\\mathcal{O}(\\log n)` in the average case.
+ ``ball_search()`` is :math:`\\mathcal{O}(\\log n)` in the average case.
"""
import numpy as np

from typing import Optional, Self

from neighbour_search import AbstractNeighbourSearch, PriorityQueue


class BallTreeLeaf:
    """
    A balltree leaf.

    Invariant:
        All points lies within ``self.sq_radius`` from ``self.center``.
    """

    def __init__(self, points: list[tuple[int, np.ndarray]]):
        """
        Create a leaf containing ``points``

        Args:
            points: the points, as a ``(id, position)`` list.
        """
        self.points = points

        self.center = np.mean([p[1] for p in points], axis=0)
        self.radius = np.sqrt(np.max([((p[1] - self.center) ** 2).sum() for p in points]))

    def __len__(self) -> int:
        return len(self.points)

    def ball_search(self, position: np.typing.NDArray, distance: float) -> list[int]:
        """Get the points that are below or equal to a ``distance`` from ``position`` within this leaf.

        Args:
            position: A (2,) array
            distance: distance from ``position`` at which point will be selected, should be >0
        Returns:
            A list of indices, the points that are at a ``radius`` distance from ``position``.
        """

        # triangle inequality
        if ((self.center - position)**2).sum() > (distance + self.radius) ** 2:
            return []

        # create queue
        queue = []

        # put item in queue
        for i, p in self.points:
            d = ((p - position)**2).sum()
            if d <= distance ** 2:
                queue.append(i)

        # return the result
        return queue

    def knn_search(
            self,
            target_position: np.typing.NDArray,
            k: int, queue: PriorityQueue,
            max_dist: float = np.inf
    ) -> float:
        """Search within leaf for the k nearest neighbors.

        Args:
            target_position: A (2,) array, the position of the node of interest.
            k: number of neighbours to return, so ``k >= 0``.
            queue: (max-first) priority queue used for searching. It is modified to contain the final result.
            max_dist: the largest distance between a point in ``queue`` and ``target_position``.
                If ``queue`` is empty, should be positive infinite.
        Returns:
            The maximum distance between a point in ``queue`` and ``target_position``.
        """

        if len(queue) < k or ((self.center - target_position)**2).sum() < max_dist + self.radius:  # triangle inequality
            for i, p in self.points:
                d = (p[0] - target_position[0]) ** 2 + (p[1] - target_position[1]) ** 2
                if d == 0:  # do not select the target point
                    continue

                queue.push(i, -d)

            for i in range(len(queue) - k):
                queue.pop()

            if len(queue) > 0:
                return np.sqrt(-queue.queue[0][0])

        return max_dist


class BallTreeNode:
    """A balltree node.

    Invariant:
        Each node in the tree defines the smallest ball that contains all data points in its subtree.
    """

    def __init__(self, center: np.typing.NDArray, radius: float):
        """Create an empty node.

        Args:
            center: central point
            radius: largest distance from ``center`` to any of the points it contains.
        """

        self.center = center
        self.radius = radius

        self.left: Optional[BallTreeNode | BallTreeLeaf] = None
        self.right: Optional[BallTreeNode | BallTreeLeaf] = None

    @staticmethod
    def _farthest_from(i: int | np.integer, points: list[tuple[int, np.ndarray]]) -> np.integer:
        """
        Find the farthest point from `i` in ``points``.

        Args:
             i: index of the point
        Returns:
            the index of the point further away.
        """
        dists = [((p[1] - points[i][1]) ** 2).sum() for p in points]
        a_ = np.argmax(dists)
        return a_

    @classmethod
    def from_points(cls, points: list[tuple[int, np.ndarray]], leaf_size: int = 3) -> Self:
        """
        Build the node from a list of points.

        Use the so-called "bouncing bubble" algorithm:

        1. Pick ``a``, the point farthest from ``points[0]``
        2. Pick ``b``, the point farthest from ``a``.
        3. The center is the midpoint between those two points
        4. Split in two based on the proximity of each point with ``a`` and ``b``.

        Args:
            points: the points, as a ``(id, position)`` list.
            leaf_size: maximum leaf size, must be >0.
        """

        a = cls._farthest_from(0, points)
        b = cls._farthest_from(a, points)

        center = (points[b][1] + points[a][1]) / 2

        sqradius = np.max([((p[1] - center) ** 2).sum() for p in points])
        da = np.array([((p[1] - points[a][1]) ** 2).sum() for p in points])
        db = np.array([((p[1] - points[b][1]) ** 2).sum() for p in points])

        to_left = np.where(da <= db)[0]
        to_right = np.where(db < da)[0]

        node = cls(center, np.sqrt(sqradius))

        if len(to_left) > leaf_size:
            node.left = BallTreeNode.from_points([points[i] for i in to_left], leaf_size)
        elif len(to_left) > 0:
            node.left = BallTreeLeaf([points[i] for i in to_left])

        if len(to_right) > leaf_size:
            node.right = BallTreeNode.from_points([points[i] for i in to_right], leaf_size)
        elif len(to_right) > 0:
            node.right = BallTreeLeaf([points[i] for i in to_right])

        return node

    def __len__(self) -> int:
        return (len(self.left) if self.left is not None else 0) + (len(self.right) if self.right is not None else 0)

    def ball_search(self, position: np.typing.NDArray, distance: float) -> list[int]:
        """Get the points that are below or equal to a ``distance`` from ``position`` within this node.

        Example:
            >>> from neighbour_search.balltree_ns import BallTreeNode
            >>> root = BallTreeNode.from_points([(0, np.array([0, 0])), (1, np.array([3, 2]))])
            >>> root.ball_search(np.array([2, 2]), 2)
            [1]

        Args:
            position: A (2,) array
            distance: distance from ``position`` at which point will be selected, should be >0
        Returns:
            A list of indices, the points that are at a ``radius`` distance from ``position``.
        """

        # triangle inequality
        if ((self.center - position)**2).sum() > (distance + self.radius) ** 2:
            return []

        # ok, it might be in this node :)
        queue = []
        if self.left is not None:
            queue += self.left.ball_search(position, distance)

        if self.right is not None:
            queue += self.right.ball_search(position, distance)

        return queue

    def knn_search(
            self,
            target_position: np.typing.NDArray,
            k: int, queue: PriorityQueue,
            max_dist: float = np.inf
    ) -> float:
        """
        Search within node for the k nearest neighbors.
        To achieve good performances, the node that is the closest to ``target_position`` is preferentially visited.

        Example:
            >>> from neighbour_search import PriorityQueue
            >>> from neighbour_search.balltree_ns import BallTreeNode
            >>> root = BallTreeNode.from_points([(0, np.array([0, 0])), (1, np.array([3, 2]))])
            >>> queue = PriorityQueue()
            >>> _ = root.knn_search(np.array([0, 0]), 1, queue)
            >>> queue.pop()
            1

        Args:
            target_position: A (2,) array, the position of the node of interest.
            k: number of neighbours to return, so ``k >= 0``.
            queue: (max-first) priority queue used for searching. It is modified to contain the final result.
            max_dist: the largest distance between a point in ``queue`` and ``target_position``.
                If ``queue`` is empty, should be positive infinite.
        Returns:
            The maximum distance between a point in ``queue`` and ``target_position``.
        """

        # triangle inequality
        if len(queue) < k or ((self.center - target_position)**2).sum() < (max_dist + self.radius) ** 2:
            node_visit = []

            if self.left is not None:
                node_visit.append(self.left)

            if self.right is not None:
                node_visit.append(self.right)

            # go to the node closest to `target_point` first
            node_visit.sort(key=lambda n: ((n.center - target_position)**2).sum())

            for node in node_visit:
                max_dist = node.knn_search(target_position, k, queue, max_dist)

        return max_dist


class BallTreeNeighbourSearch(AbstractNeighbourSearch):
    """
    Create a balltree for neighbour search.

    Example:
        >>> import numpy as np
        >>> from neighbour_search.balltree_ns import BallTreeNeighbourSearch as NeighbourSearch
        >>> neighbour_search = NeighbourSearch(np.array([[0, 0], [1, 1]]))
        >>> neighbour_search.knn_search(0, 1)
        [1]
        >>> neighbour_search.ball_search(np.array([0, 0]), 5)
        [0, 1]
    """

    def __init__(self, positions: np.typing.NDArray, leaf_size: int = 5):
        """
        Create a new balltree, containing the root node.

        Args:
            positions: the points
            leaf_size: maximum leaf size, must be >0.
        """
        assert leaf_size > 0

        super().__init__(positions)

        if positions.shape[0] > leaf_size:
            self.root: BallTreeNode | BallTreeLeaf = BallTreeNode.from_points(
                [(i, positions[i]) for i in range(positions.shape[0])], leaf_size)
        else:
            self.root = BallTreeLeaf([(i, positions[i]) for i in range(positions.shape[0])])

    def knn_search(self, target: int, k: int) -> list[int]:
        assert 0 <= target < len(self)
        assert 0 <= k < len(self)

        queue = PriorityQueue()
        self.root.knn_search(self.positions[target], k, queue)

        # get the ``k`` first ones
        return [queue.pop() for _ in range(k)]

    def ball_search(self, position: np.typing.NDArray, distance: float) -> list[int]:
        assert distance >= 0

        return self.root.ball_search(position, distance)
