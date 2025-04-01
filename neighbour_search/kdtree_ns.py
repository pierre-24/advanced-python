"""
`k-d tree <https://en.wikipedia.org/wiki/K-d_tree>`_ neighbour search implementation.

+ Creation is :math:`\\mathcal{O}(n\\log n)`.
+ ``knn_search()`` is :math:`\\mathcal{O}(\\log n)` in the average case.
+ ``ball_search()`` is :math:`\\mathcal{O}(\\log n)` in the average case.
"""
import numpy as np
import math

from typing import Optional, Self

from neighbour_search import AbstractNeighbourSearch, PriorityQueue


class KDTreeLeaf:
    """
    A k-d tree leaf, which contains points that are in (relative) proximity.
    """

    def __init__(self, points: list[tuple[int, np.ndarray]]):
        """
        Create a leaf containing `points`

        Args:
            points: the points, as a ``(id, position)`` list.
        """
        self.points = points

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

        # create queue
        queue = []

        # put item in queue
        for i, p in self.points:
            d = (p[0] - position[0]) ** 2 + (p[1] - position[1]) ** 2
            if d <= distance ** 2:
                queue.append(i)

        # return the result
        return queue

    def knn_search(
            self,
            target_position: np.typing.NDArray,
            k: int, queue: PriorityQueue,
            max_dist: float = math.inf
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

        for i, p in self.points:
            d = (p[0] - target_position[0]) ** 2 + (p[1] - target_position[1]) ** 2
            if d == 0:  # do not select the target point
                continue

            queue.push(i, -d)

        for i in range(len(queue) - k):
            queue.pop()

        if len(queue) > 0:
            return np.sqrt(-queue.queue[0][0])
        else:
            return max_dist


class KDTreeNode:
    """A k-d tree node.

    Invariant:
        The left subtree contains points whose value for component `c` is lower than (or equal to) `midpoint`,
        while the right subtree contains points which are higher.
    """

    def __init__(self, depth: int, midpoint: float):
        """Create an empty node.

        Args:
            depth: depth of this node ``depth >= 0``.
            midpoint: midpoint value
        """

        self.depth = depth
        self.midpoint = midpoint
        self.left: Optional[KDTreeNode | KDTreeLeaf] = None
        self.right: Optional[KDTreeNode | KDTreeLeaf] = None

    @classmethod
    def from_points(cls, points: list[tuple[int, np.ndarray]], depth: int = 0, max_leaf: int = 3) -> Self:
        """
        Build the node from a list of points.

        Args:
            points: the points, as a ``(id, position)`` list.
            depth: depth of this node, so ``depth >= 0``.
            max_leaf: maximum leaf size, must be >0.
        """

        midpoint = np.median([p[1][depth % 2] for p in points])

        node = cls(depth, midpoint)

        to_left = []
        to_right = []

        for p in points:
            if p[1][depth % 2] <= midpoint:
                to_left.append(p)
            else:
                to_right.append(p)

        if len(to_left) > max_leaf:
            node.left = KDTreeNode.from_points(to_left, depth + 1, max_leaf)
        elif len(to_left) > 0:
            node.left = KDTreeLeaf(to_left)

        if len(to_right) > max_leaf:
            node.right = KDTreeNode.from_points(to_right, depth + 1, max_leaf)
        elif len(to_right) > 0:
            node.right = KDTreeLeaf(to_right)

        return node

    def __len__(self) -> int:
        return (len(self.left) if self.left is not None else 0) + (len(self.right) if self.right is not None else 0)

    def ball_search(self, position: np.typing.NDArray, distance: float) -> list[int]:
        """Get the points that are below or equal to a ``distance`` from ``position`` within this node.

        Example:
            >>> from neighbour_search.kdtree_ns import KDTreeNode
            >>> root = KDTreeNode.from_points([(0, np.array([0, 0])), (1, np.array([3, 2]))])
            >>> root.ball_search(np.array([2, 2]), 2)
            [1]

        Args:
            position: A (2,) array
            distance: distance from ``position`` at which point will be selected, should be >0
        Returns:
            A list of indices, the points that are at a ``radius`` distance from ``position``.
        """

        queue = []
        if self.left is not None and position[self.depth % 2] - distance <= self.midpoint:
            queue += self.left.ball_search(position, distance)

        if self.right is not None and position[self.depth % 2] + distance > self.midpoint:
            queue += self.right.ball_search(position, distance)

        return queue

    def knn_search(
            self,
            target_position: np.typing.NDArray,
            k: int, queue: PriorityQueue,
            max_dist: float = math.inf
    ) -> float:
        """Search within node for the k nearest neighbors.

        Example:
            >>> from neighbour_search import PriorityQueue
            >>> from neighbour_search.kdtree_ns import KDTreeNode
            >>> root = KDTreeNode.from_points([(0, np.array([0, 0])), (1, np.array([3, 2]))])
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

        if self.left is not None:
            if len(queue) < k:
                max_dist = self.left.knn_search(target_position, k, queue, max_dist)
            elif target_position[self.depth % 2] - max_dist < self.midpoint:
                max_dist = self.left.knn_search(target_position, k, queue, max_dist)

        if self.right is not None:
            if len(queue) < k:
                max_dist = self.right.knn_search(target_position, k, queue, max_dist)
            elif target_position[self.depth % 2] + max_dist > self.midpoint:
                max_dist = self.right.knn_search(target_position, k, queue, max_dist)

        return max_dist


class KDTreeNeighbourSearch(AbstractNeighbourSearch):
    """
    Create a k-d tree for neighbour search.

    Example:
        >>> import numpy as np
        >>> from neighbour_search.kdtree_ns import KDTreeNeighbourSearch as NeighbourSearch
        >>> neighbour_search = NeighbourSearch(np.array([[0, 0], [1, 1]]))
        >>> neighbour_search.knn_search(0, 1)
        [1]
        >>> neighbour_search.ball_search(np.array([0, 0]), 5)
        [0, 1]
    """

    def __init__(self, positions: np.typing.NDArray, max_leaf: int = 3):
        """
        Create a new k-d tree, containing the root node.

        Args:
            positions: the points
            max_leaf: maximum leaf size, must be >0.
        """
        assert max_leaf > 0

        super().__init__(positions)
        self.root = KDTreeNode.from_points([(i, positions[i]) for i in range(positions.shape[0])], 0, max_leaf)

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

    def __len__(self):
        return len(self.root)
