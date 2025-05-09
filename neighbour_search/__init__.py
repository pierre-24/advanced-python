"""
Basic objects
"""

import numpy as np
import heapq

__version__ = '0.1.0'


class PriorityQueue:
    """Define a priority queue, which store items by priority.

    Invariant:
        The first element of the queue is the one with the lowest priority.

    Example:
        >>> from neighbour_search import PriorityQueue
        >>> pq = PriorityQueue()
        >>> pq.push('a', 5)
        >>> pq.push('b', 3)
        >>> pq.push('c', 2)
        >>> pq.push('d', 7)
        >>> pq.pop()
        'c'
        >>> pq.pop()
        'b'
    """

    def __init__(self):
        """Create a new (empty) priority queue
        """
        self.queue: list = []

    def push(self, item: object, priority: int | float):
        """Push an item into the queue, with a given priority.

        Args:
            item: item
            priority: priority of ``item``
        """
        heapq.heappush(self.queue, (priority, item))

    def pop(self):
        """
        Pop the first item (the one with the lowest priority).

        Returns:
            The item with the lowest priority in the queue.
        Raises:
             KeyError: if empty.
        """
        if len(self.queue) == 0:
            raise IndexError('Priority queue is empty')

        return heapq.heappop(self.queue)[1]

    def __len__(self) -> int:
        return len(self.queue)


class AbstractNeighbourSearch:
    """
    Define a data Structure that stores a list of 2D points, and provide ways to find neighbours.

    .. note::

       This is an abstract base class, which does not implement ``knn_search()`` nor ``ball_search()``.
    """

    def __init__(self, positions: np.typing.NDArray):
        """
        Create a new neighbour search object.

        Args:
            positions: A (N,2) array of points, where N is the number of points.
        """
        self.positions = positions

    def ball_search(self, position: np.typing.NDArray, distance: float) -> list[int]:
        """Get the points that are below or equal to a ``distance`` from ``position``.

        Args:
            position: A (2,) array
            distance: distance from ``position`` at which point will be selected, should be >= 0
        Returns:
            A list of indices, the points that are at a ``radius`` distance from ``position``.
        """

        raise NotImplementedError()

    def knn_search(self, target: int, k: int) -> list[int]:
        """Find the ``k`` nearest neighbours of point ``target``.

        Args:
            target: a valid index, so ``0 <= target < len(self)``.
            k: number of neighbours to return, so ``0 <= k < len(self)``.
        Returns:
            a list of indices, the ``k`` nearest neighbours of ``target``.
        """

        raise NotImplementedError()

    def __len__(self) -> int:
        """
        Returns:
            The number of points in the data structure.
        """

        return self.positions.shape[0]
