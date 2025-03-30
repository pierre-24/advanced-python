"""
Search for neighbours in a list of 2D points
"""

import numpy as np
import heapq

__version__ = '0.1.0'


class PriorityQueue:
    """Define a priority queue, which store items by priority.

    Invariant: the items are sorted by priority, from low to high.
    """

    def __init__(self):
        """Create a new empty priority queue
        """
        self.queue = []

    def put(self, item: object, priority: int):
        """Put an item into the queue, with a given priority.

        Args:
            item: item
            priority: priority of `item`
        """
        heapq.heappush(self.queue, (priority, item))

    def pop(self):
        """
        Pop the item with the lowest priority.

        Raises:
             KeyError: if empty.
        """
        if len(self.queue) == 0:
            raise IndexError('Priority queue is empty')

        return heapq.heappop(self.queue)[1]


class AbstractNeighbourSearch:
    """
    Define a Data Structure that stores a list of 2D points, and provide ways to find neighbours.
    """

    def __init__(self, positions: np.ndarray[float]):
        """
        Create a new neighbour search object.

        Args:
            positions: A (N,2) array of points, where N is the number of points.
        """
        self.positions = positions

    def ball_search(self, position: np.ndarray[float], distance: float) -> list[int]:
        """Get the points that are below or equal to a `distance` from `position`.

        Args:
            position: A (2,) array
            distance: distance from `position` at which point will be selected, should be >0
        Returns:
            A list of indices, the points that are at a `radius` distance from `position`.
        """

        raise NotImplementedError()

    def knn_search(self, target: int, k: int) -> list[int]:
        """Find the `k` nearest neighbours of point `target`.

        Args:
            target: a valid index, `0 <= target < len(self)`.
            k: number of neighbours to return. `0 <= k < len(self)`.
        Returns:
            a list of indices, the `k` nearest neighbours of point `point_index`.
        """

        raise NotImplementedError()

    def __len__(self) -> int:
        """
        Returns:
            The number of points in the data structure.
        """

        return self.positions.shape[0]
